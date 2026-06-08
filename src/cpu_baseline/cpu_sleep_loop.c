#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "timing.h"
#include "logging.h"
#include "signal_markers.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEFAULT_ITERATIONS 120
#define DEFAULT_SLEEP_MS 500
#define DEFAULT_READY_FILE "/tmp/criu-experiment-cpu.ready"
#define DEFAULT_PROGRESS_FILE "/tmp/criu-experiment-cpu.progress"
#define DEFAULT_DONE_FILE "/tmp/criu-experiment-cpu.done"
#define DEFAULT_OUTPUT_JSON "/tmp/criu-experiment-cpu.json"

typedef struct Options {
    int iterations;
    int sleep_ms;
    int clear_markers;

    const char *ready_file;
    const char *progress_file;
    const char *done_file;
    const char *output_json;
} Options;

typedef struct Metrics {
    int completed_iterations;
    int interrupted;
    int passed;

    double total_program_ms;
    double loop_total_ms;
} Metrics;

static volatile sig_atomic_t g_stop_requested = 0;

static void handle_signal(int signo)
{
    (void)signo;
    g_stop_requested = 1;
}

static void install_signal_handlers(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));

    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);

    if (sigaction(SIGTERM, &sa, NULL) != 0) {
        CRIU_EXP_LOG_ERRNO("sigaction(SIGTERM) failed");
    }

    if (sigaction(SIGINT, &sa, NULL) != 0) {
        CRIU_EXP_LOG_ERRNO("sigaction(SIGINT) failed");
    }
}

static void print_usage(const char *program)
{
    fprintf(
        stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --iterations N       Number of loop iterations, default: %d\n"
        "  --sleep-ms N         Sleep duration per iteration, default: %d\n"
        "  --output-json PATH   Output JSON path, default: %s\n"
        "  --ready-file PATH    Ready marker path, default: %s\n"
        "  --progress-file PATH Progress marker path, default: %s\n"
        "  --done-file PATH     Done marker path, default: %s\n"
        "  --clear-markers 0|1  Remove old marker files at startup, default: 1\n"
        "  --help               Show this message\n"
        "\n",
        program,
        DEFAULT_ITERATIONS,
        DEFAULT_SLEEP_MS,
        DEFAULT_OUTPUT_JSON,
        DEFAULT_READY_FILE,
        DEFAULT_PROGRESS_FILE,
        DEFAULT_DONE_FILE
    );
}

static int parse_int_arg(const char *name, const char *value)
{
    if (value == NULL) {
        CRIU_EXP_LOG_ERROR("Missing value for argument %s", name);
        exit(EXIT_FAILURE);
    }

    char *end = NULL;
    long parsed = strtol(value, &end, 10);

    if (end == value || *end != '\0') {
        CRIU_EXP_LOG_ERROR("Invalid integer for %s: %s", name, value);
        exit(EXIT_FAILURE);
    }

    if (parsed < -2147483647L || parsed > 2147483647L) {
        CRIU_EXP_LOG_ERROR("Integer out of range for %s: %s", name, value);
        exit(EXIT_FAILURE);
    }

    return (int)parsed;
}

static Options parse_options(int argc, char **argv)
{
    Options opts;

    opts.iterations = DEFAULT_ITERATIONS;
    opts.sleep_ms = DEFAULT_SLEEP_MS;
    opts.clear_markers = 1;
    opts.ready_file = DEFAULT_READY_FILE;
    opts.progress_file = DEFAULT_PROGRESS_FILE;
    opts.done_file = DEFAULT_DONE_FILE;
    opts.output_json = DEFAULT_OUTPUT_JSON;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if (strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(arg, "--iterations") == 0) {
            opts.iterations = parse_int_arg(arg, argv[++i]);
        } else if (strcmp(arg, "--sleep-ms") == 0) {
            opts.sleep_ms = parse_int_arg(arg, argv[++i]);
        } else if (strcmp(arg, "--output-json") == 0) {
            opts.output_json = argv[++i];
        } else if (strcmp(arg, "--ready-file") == 0) {
            opts.ready_file = argv[++i];
        } else if (strcmp(arg, "--progress-file") == 0) {
            opts.progress_file = argv[++i];
        } else if (strcmp(arg, "--done-file") == 0) {
            opts.done_file = argv[++i];
        } else if (strcmp(arg, "--clear-markers") == 0) {
            opts.clear_markers = parse_int_arg(arg, argv[++i]);
        } else {
            CRIU_EXP_LOG_ERROR("Unknown argument: %s", arg);
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return opts;
}

static void validate_options(const Options *opts)
{
    if (opts->iterations <= 0) {
        CRIU_EXP_LOG_ERROR("iterations must be > 0");
        exit(EXIT_FAILURE);
    }

    if (opts->sleep_ms < 0) {
        CRIU_EXP_LOG_ERROR("sleep_ms must be >= 0");
        exit(EXIT_FAILURE);
    }

    if (opts->ready_file == NULL ||
        opts->progress_file == NULL ||
        opts->done_file == NULL ||
        opts->output_json == NULL) {
        CRIU_EXP_LOG_ERROR("marker paths and output_json must not be NULL");
        exit(EXIT_FAILURE);
    }
}

static const char *bool_json(int value)
{
    return value ? "true" : "false";
}

static void write_json_result(
    const char *path,
    const Options *opts,
    const Metrics *metrics,
    int exit_code
)
{
    FILE *fp = fopen(path, "w");

    if (fp == NULL) {
        CRIU_EXP_LOG_ERRNO("Failed to open CPU output JSON file");
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"program\": \"cpu_sleep_loop\",\n");
    fprintf(fp, "  \"pid\": %d,\n", (int)getpid());
    fprintf(fp, "  \"exit_code\": %d,\n", exit_code);
    fprintf(fp, "  \"passed\": %s,\n", bool_json(metrics->passed));
    fprintf(fp, "  \"interrupted\": %s,\n", bool_json(metrics->interrupted));
    fprintf(fp, "  \"iterations\": %d,\n", opts->iterations);
    fprintf(fp, "  \"completed_iterations\": %d,\n", metrics->completed_iterations);
    fprintf(fp, "  \"sleep_ms\": %d,\n", opts->sleep_ms);
    fprintf(fp, "  \"timing_ms\": {\n");
    fprintf(fp, "    \"total_program_ms\": %.6f,\n", metrics->total_program_ms);
    fprintf(fp, "    \"loop_total_ms\": %.6f\n", metrics->loop_total_ms);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"markers\": {\n");
    fprintf(fp, "    \"ready_file\": \"%s\",\n", opts->ready_file);
    fprintf(fp, "    \"progress_file\": \"%s\",\n", opts->progress_file);
    fprintf(fp, "    \"done_file\": \"%s\"\n", opts->done_file);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    fclose(fp);
}

int main(int argc, char **argv)
{
    int64_t total_start_ns = criu_exp_now_ns();

    Options opts = parse_options(argc, argv);
    validate_options(&opts);

    Metrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    int exit_code = EXIT_FAILURE;

    install_signal_handlers();

    CRIU_EXP_LOG_INFO("Starting CPU CRIU baseline loop");
    CRIU_EXP_LOG_INFO(
        "pid=%d iterations=%d sleep_ms=%d",
        (int)getpid(),
        opts.iterations,
        opts.sleep_ms
    );

    if (opts.clear_markers) {
        if (criu_exp_clear_markers(opts.ready_file, opts.progress_file, opts.done_file) != 0) {
            CRIU_EXP_LOG_WARN("Failed to clear one or more CPU marker files");
        }
    }

    if (criu_exp_write_progress(opts.progress_file, 0) != 0) {
        CRIU_EXP_LOG_WARN("Failed to write initial CPU progress marker");
    }

    if (criu_exp_write_marker(opts.ready_file) != 0) {
        CRIU_EXP_LOG_WARN("Failed to write CPU ready marker");
    } else {
        CRIU_EXP_LOG_INFO("CPU ready marker written: %s", opts.ready_file);
    }

    int64_t loop_start_ns = criu_exp_now_ns();

    for (int iter = 1; iter <= opts.iterations; ++iter) {
        if (g_stop_requested) {
            CRIU_EXP_LOG_WARN("Stop requested at iteration %d", iter);
            metrics.interrupted = 1;
            break;
        }

        criu_exp_sleep_ms(opts.sleep_ms);

        metrics.completed_iterations = iter;

        if (criu_exp_write_progress(opts.progress_file, iter) != 0) {
            CRIU_EXP_LOG_WARN("Failed to write CPU progress marker for iteration %d", iter);
        }

        if (iter == 1 || iter == opts.iterations || iter % 10 == 0) {
            CRIU_EXP_LOG_INFO("CPU loop completed iteration %d/%d", iter, opts.iterations);
        }
    }

    int64_t loop_end_ns = criu_exp_now_ns();
    metrics.loop_total_ms = criu_exp_ms_between(loop_start_ns, loop_end_ns);

    if (!metrics.interrupted && metrics.completed_iterations == opts.iterations) {
        metrics.passed = 1;
        exit_code = EXIT_SUCCESS;
    } else {
        metrics.passed = 0;
        exit_code = EXIT_FAILURE;
    }

    metrics.total_program_ms = criu_exp_elapsed_ms_since(total_start_ns);

    write_json_result(opts.output_json, &opts, &metrics, exit_code);

    if (criu_exp_write_marker(opts.done_file) != 0) {
        CRIU_EXP_LOG_WARN("Failed to write CPU done marker");
    }

    if (metrics.passed) {
        CRIU_EXP_LOG_INFO("CPU CRIU baseline loop completed successfully");
    } else {
        CRIU_EXP_LOG_ERROR("CPU CRIU baseline loop failed or was interrupted");
    }

    return exit_code;
}