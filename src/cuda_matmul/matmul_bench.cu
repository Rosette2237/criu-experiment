#include "matmul_config.h"
#include "timing.h"
#include "logging.h"
#include "signal_markers.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>

struct Options {
    int matrix_size = CRIU_EXP_DEFAULT_MATRIX_SIZE;
    int iterations = CRIU_EXP_DEFAULT_ITERATIONS;
    int block_size = CRIU_EXP_DEFAULT_BLOCK_SIZE;
    int device_id = CRIU_EXP_DEFAULT_DEVICE_ID;
    int sleep_ms_between_iterations = CRIU_EXP_DEFAULT_SLEEP_MS_BETWEEN_ITERATIONS;
    int verify = CRIU_EXP_DEFAULT_VERIFY;
    int verify_samples = CRIU_EXP_DEFAULT_VERIFY_SAMPLES;
    int clear_markers = 1;

    const char *ready_file = CRIU_EXP_DEFAULT_READY_FILE;
    const char *progress_file = CRIU_EXP_DEFAULT_PROGRESS_FILE;
    const char *done_file = CRIU_EXP_DEFAULT_DONE_FILE;
    const char *output_json = CRIU_EXP_DEFAULT_OUTPUT_JSON;
};

struct Metrics {
    double total_program_ms = -1.0;
    double cuda_set_device_ms = -1.0;
    double cuda_init_ms = -1.0;
    double host_allocation_ms = -1.0;
    double host_initialization_ms = -1.0;
    double device_allocation_ms = -1.0;
    double h2d_copy_ms = -1.0;
    double kernel_total_ms = -1.0;
    double d2h_copy_ms = -1.0;
    double verification_ms = -1.0;
    double cleanup_ms = -1.0;

    int completed_iterations = 0;
    int passed = 0;
    int verification_failures = 0;
};

static void print_usage(const char *program)
{
    std::fprintf(
        stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --matrix-size N                     Square matrix size, default: %d\n"
        "  --iterations N                      Number of repeated matmul iterations, default: %d\n"
        "  --block-size N                      CUDA block width/height, default: %d\n"
        "  --device N                          CUDA device ID, default: %d\n"
        "  --sleep-ms-between-iterations N     Sleep between iterations, default: %d\n"
        "  --verify 0|1                        Enable sampled correctness check, default: %d\n"
        "  --verify-samples N                  Number of sampled output elements, default: %d\n"
        "  --output-json PATH                  Output JSON path, default: %s\n"
        "  --ready-file PATH                   Ready marker path, default: %s\n"
        "  --progress-file PATH                Progress marker path, default: %s\n"
        "  --done-file PATH                    Done marker path, default: %s\n"
        "  --clear-markers 0|1                 Remove old marker files at startup, default: 1\n"
        "  --help                              Show this message\n"
        "\n",
        program,
        CRIU_EXP_DEFAULT_MATRIX_SIZE,
        CRIU_EXP_DEFAULT_ITERATIONS,
        CRIU_EXP_DEFAULT_BLOCK_SIZE,
        CRIU_EXP_DEFAULT_DEVICE_ID,
        CRIU_EXP_DEFAULT_SLEEP_MS_BETWEEN_ITERATIONS,
        CRIU_EXP_DEFAULT_VERIFY,
        CRIU_EXP_DEFAULT_VERIFY_SAMPLES,
        CRIU_EXP_DEFAULT_OUTPUT_JSON,
        CRIU_EXP_DEFAULT_READY_FILE,
        CRIU_EXP_DEFAULT_PROGRESS_FILE,
        CRIU_EXP_DEFAULT_DONE_FILE
    );
}

static int parse_int_arg(const char *name, const char *value)
{
    if (value == nullptr) {
        CRIU_EXP_LOG_ERROR("Missing value for argument %s", name);
        std::exit(EXIT_FAILURE);
    }

    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);

    if (end == value || *end != '\0') {
        CRIU_EXP_LOG_ERROR("Invalid integer for %s: %s", name, value);
        std::exit(EXIT_FAILURE);
    }

    if (parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        CRIU_EXP_LOG_ERROR("Integer out of range for %s: %s", name, value);
        std::exit(EXIT_FAILURE);
    }

    return static_cast<int>(parsed);
}

static Options parse_options(int argc, char **argv)
{
    Options opts;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if (std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (std::strcmp(arg, "--matrix-size") == 0) {
            opts.matrix_size = parse_int_arg(arg, argv[++i]);
        } else if (std::strcmp(arg, "--iterations") == 0) {
            opts.iterations = parse_int_arg(arg, argv[++i]);
        } else if (std::strcmp(arg, "--block-size") == 0) {
            opts.block_size = parse_int_arg(arg, argv[++i]);
        } else if (std::strcmp(arg, "--device") == 0) {
            opts.device_id = parse_int_arg(arg, argv[++i]);
        } else if (std::strcmp(arg, "--sleep-ms-between-iterations") == 0) {
            opts.sleep_ms_between_iterations = parse_int_arg(arg, argv[++i]);
        } else if (std::strcmp(arg, "--verify") == 0) {
            opts.verify = parse_int_arg(arg, argv[++i]);
        } else if (std::strcmp(arg, "--verify-samples") == 0) {
            opts.verify_samples = parse_int_arg(arg, argv[++i]);
        } else if (std::strcmp(arg, "--output-json") == 0) {
            opts.output_json = argv[++i];
        } else if (std::strcmp(arg, "--ready-file") == 0) {
            opts.ready_file = argv[++i];
        } else if (std::strcmp(arg, "--progress-file") == 0) {
            opts.progress_file = argv[++i];
        } else if (std::strcmp(arg, "--done-file") == 0) {
            opts.done_file = argv[++i];
        } else if (std::strcmp(arg, "--clear-markers") == 0) {
            opts.clear_markers = parse_int_arg(arg, argv[++i]);
        } else {
            CRIU_EXP_LOG_ERROR("Unknown argument: %s", arg);
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    return opts;
}

static void validate_options(const Options &opts)
{
    if (opts.matrix_size < CRIU_EXP_MIN_MATRIX_SIZE ||
        opts.matrix_size > CRIU_EXP_MAX_MATRIX_SIZE) {
        CRIU_EXP_LOG_ERROR(
            "matrix_size must be between %d and %d",
            CRIU_EXP_MIN_MATRIX_SIZE,
            CRIU_EXP_MAX_MATRIX_SIZE
        );
        std::exit(EXIT_FAILURE);
    }

    if (opts.iterations < CRIU_EXP_MIN_ITERATIONS ||
        opts.iterations > CRIU_EXP_MAX_ITERATIONS) {
        CRIU_EXP_LOG_ERROR(
            "iterations must be between %d and %d",
            CRIU_EXP_MIN_ITERATIONS,
            CRIU_EXP_MAX_ITERATIONS
        );
        std::exit(EXIT_FAILURE);
    }

    if (opts.block_size < CRIU_EXP_MIN_BLOCK_SIZE ||
        opts.block_size > CRIU_EXP_MAX_BLOCK_SIZE) {
        CRIU_EXP_LOG_ERROR(
            "block_size must be between %d and %d",
            CRIU_EXP_MIN_BLOCK_SIZE,
            CRIU_EXP_MAX_BLOCK_SIZE
        );
        std::exit(EXIT_FAILURE);
    }

    if (opts.verify_samples < CRIU_EXP_MIN_VERIFY_SAMPLES ||
        opts.verify_samples > CRIU_EXP_MAX_VERIFY_SAMPLES) {
        CRIU_EXP_LOG_ERROR(
            "verify_samples must be between %d and %d",
            CRIU_EXP_MIN_VERIFY_SAMPLES,
            CRIU_EXP_MAX_VERIFY_SAMPLES
        );
        std::exit(EXIT_FAILURE);
    }

    if (opts.sleep_ms_between_iterations < 0) {
        CRIU_EXP_LOG_ERROR("sleep_ms_between_iterations must be >= 0");
        std::exit(EXIT_FAILURE);
    }

    if (opts.ready_file == nullptr ||
        opts.progress_file == nullptr ||
        opts.done_file == nullptr ||
        opts.output_json == nullptr) {
        CRIU_EXP_LOG_ERROR("marker paths and output_json must not be null");
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _status = (call);                                          \
        if (_status != cudaSuccess) {                                          \
            CRIU_EXP_LOG_ERROR(                                                \
                "CUDA error at %s:%d: %s failed: %s",                          \
                __FILE__,                                                      \
                __LINE__,                                                      \
                #call,                                                         \
                cudaGetErrorString(_status)                                    \
            );                                                                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void matmul_kernel(
    const float *A,
    const float *B,
    float *C,
    int N
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

static float matrix_a_value(int row, int col)
{
    int value = (row + col) % CRIU_EXP_MATRIX_A_MODULUS;
    return static_cast<float>(value) * CRIU_EXP_MATRIX_A_SCALE;
}

static float matrix_b_value(int row, int col)
{
    int value = (row * 3 + col * 2) % CRIU_EXP_MATRIX_B_MODULUS;
    return static_cast<float>(value) * CRIU_EXP_MATRIX_B_SCALE;
}

static void initialize_matrices(float *A, float *B, float *C, int N)
{
    size_t total_elements = static_cast<size_t>(N) * static_cast<size_t>(N);

    for (size_t idx = 0; idx < total_elements; ++idx) {
        int row = static_cast<int>(idx / static_cast<size_t>(N));
        int col = static_cast<int>(idx % static_cast<size_t>(N));

        A[idx] = matrix_a_value(row, col);
        B[idx] = matrix_b_value(row, col);
        C[idx] = 0.0f;
    }
}

static float reference_element(int row, int col, int N)
{
    float sum = 0.0f;

    for (int k = 0; k < N; ++k) {
        sum += matrix_a_value(row, k) * matrix_b_value(k, col);
    }

    return sum;
}

static int verify_sampled_result(
    const float *C,
    int N,
    int samples,
    int *failures_out
)
{
    int failures = 0;

    if (samples <= 0) {
        samples = 1;
    }

    for (int sample = 0; sample < samples; ++sample) {
        int row = (sample * 97 + 13) % N;
        int col = (sample * 193 + 29) % N;

        float actual = C[row * N + col];
        float expected = reference_element(row, col, N);

        float abs_diff = std::fabs(actual - expected);
        float rel_denom = std::fmax(std::fabs(expected), 1.0f);
        float rel_diff = abs_diff / rel_denom;

        if (abs_diff > CRIU_EXP_VERIFY_ABS_TOLERANCE &&
            rel_diff > CRIU_EXP_VERIFY_REL_TOLERANCE) {
            ++failures;

            if (failures <= 8) {
                CRIU_EXP_LOG_ERROR(
                    "Verification mismatch sample=%d row=%d col=%d actual=%f expected=%f abs_diff=%f rel_diff=%f",
                    sample,
                    row,
                    col,
                    actual,
                    expected,
                    abs_diff,
                    rel_diff
                );
            }
        }
    }

    if (failures_out != nullptr) {
        *failures_out = failures;
    }

    return failures == 0;
}

static const char *bool_json(int value)
{
    return value ? "true" : "false";
}

static void write_json_result(
    const char *path,
    const Options &opts,
    const Metrics &metrics,
    size_t matrix_bytes,
    size_t gpu_memory_allocated_bytes,
    size_t host_memory_allocated_bytes,
    int exit_code
)
{
    FILE *fp = std::fopen(path, "w");

    if (fp == nullptr) {
        CRIU_EXP_LOG_ERRNO("Failed to open output JSON file");
        return;
    }

    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"program\": \"matmul_bench\",\n");
    std::fprintf(fp, "  \"exit_code\": %d,\n", exit_code);
    std::fprintf(fp, "  \"passed\": %s,\n", bool_json(metrics.passed));
    std::fprintf(fp, "  \"matrix_size\": %d,\n", opts.matrix_size);
    std::fprintf(fp, "  \"iterations\": %d,\n", opts.iterations);
    std::fprintf(fp, "  \"completed_iterations\": %d,\n", metrics.completed_iterations);
    std::fprintf(fp, "  \"block_size\": %d,\n", opts.block_size);
    std::fprintf(fp, "  \"device_id\": %d,\n", opts.device_id);
    std::fprintf(fp, "  \"sleep_ms_between_iterations\": %d,\n", opts.sleep_ms_between_iterations);
    std::fprintf(fp, "  \"verify\": %s,\n", bool_json(opts.verify));
    std::fprintf(fp, "  \"verify_samples\": %d,\n", opts.verify_samples);
    std::fprintf(fp, "  \"verification_failures\": %d,\n", metrics.verification_failures);
    std::fprintf(fp, "  \"matrix_bytes\": %zu,\n", matrix_bytes);
    std::fprintf(fp, "  \"gpu_memory_allocated_bytes\": %zu,\n", gpu_memory_allocated_bytes);
    std::fprintf(fp, "  \"host_memory_allocated_bytes\": %zu,\n", host_memory_allocated_bytes);
    std::fprintf(fp, "  \"timing_ms\": {\n");
    std::fprintf(fp, "    \"total_program_ms\": %.6f,\n", metrics.total_program_ms);
    std::fprintf(fp, "    \"cuda_set_device_ms\": %.6f,\n", metrics.cuda_set_device_ms);
    std::fprintf(fp, "    \"cuda_init_ms\": %.6f,\n", metrics.cuda_init_ms);
    std::fprintf(fp, "    \"host_allocation_ms\": %.6f,\n", metrics.host_allocation_ms);
    std::fprintf(fp, "    \"host_initialization_ms\": %.6f,\n", metrics.host_initialization_ms);
    std::fprintf(fp, "    \"device_allocation_ms\": %.6f,\n", metrics.device_allocation_ms);
    std::fprintf(fp, "    \"h2d_copy_ms\": %.6f,\n", metrics.h2d_copy_ms);
    std::fprintf(fp, "    \"kernel_total_ms\": %.6f,\n", metrics.kernel_total_ms);
    std::fprintf(fp, "    \"d2h_copy_ms\": %.6f,\n", metrics.d2h_copy_ms);
    std::fprintf(fp, "    \"verification_ms\": %.6f,\n", metrics.verification_ms);
    std::fprintf(fp, "    \"cleanup_ms\": %.6f\n", metrics.cleanup_ms);
    std::fprintf(fp, "  },\n");
    std::fprintf(fp, "  \"markers\": {\n");
    std::fprintf(fp, "    \"ready_file\": \"%s\",\n", opts.ready_file);
    std::fprintf(fp, "    \"progress_file\": \"%s\",\n", opts.progress_file);
    std::fprintf(fp, "    \"done_file\": \"%s\"\n", opts.done_file);
    std::fprintf(fp, "  }\n");
    std::fprintf(fp, "}\n");

    std::fclose(fp);
}

int main(int argc, char **argv)
{
    int64_t total_start_ns = criu_exp_now_ns();

    Options opts = parse_options(argc, argv);
    validate_options(opts);

    Metrics metrics;

    CRIU_EXP_LOG_INFO("Starting CUDA matrix multiplication benchmark");
    CRIU_EXP_LOG_INFO("matrix_size=%d iterations=%d block_size=%d device=%d",
                      opts.matrix_size,
                      opts.iterations,
                      opts.block_size,
                      opts.device_id);

    if (opts.clear_markers) {
        if (criu_exp_clear_markers(opts.ready_file, opts.progress_file, opts.done_file) != 0) {
            CRIU_EXP_LOG_WARN("Failed to clear one or more marker files");
        }
    }

    int N = opts.matrix_size;
    size_t total_elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    size_t matrix_bytes = total_elements * sizeof(float);
    size_t gpu_memory_allocated_bytes = matrix_bytes * 3;
    size_t host_memory_allocated_bytes = matrix_bytes * 3;

    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    int exit_code = EXIT_FAILURE;

    int64_t t0 = 0;
    int64_t t1 = 0;

    t0 = criu_exp_now_ns();
    CUDA_CHECK(cudaSetDevice(opts.device_id));
    t1 = criu_exp_now_ns();
    metrics.cuda_set_device_ms = criu_exp_ms_between(t0, t1);

    t0 = criu_exp_now_ns();
    CUDA_CHECK(cudaFree(nullptr));
    t1 = criu_exp_now_ns();
    metrics.cuda_init_ms = criu_exp_ms_between(t0, t1);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device_id));

    CRIU_EXP_LOG_INFO("Using CUDA device %d: %s", opts.device_id, prop.name);
    CRIU_EXP_LOG_INFO("Matrix bytes per matrix: %zu", matrix_bytes);
    CRIU_EXP_LOG_INFO("Total explicit GPU matrix allocation bytes: %zu", gpu_memory_allocated_bytes);

    t0 = criu_exp_now_ns();
    h_A = static_cast<float *>(std::malloc(matrix_bytes));
    h_B = static_cast<float *>(std::malloc(matrix_bytes));
    h_C = static_cast<float *>(std::malloc(matrix_bytes));
    t1 = criu_exp_now_ns();
    metrics.host_allocation_ms = criu_exp_ms_between(t0, t1);

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        CRIU_EXP_LOG_ERROR("Host memory allocation failed");
        goto cleanup;
    }

    t0 = criu_exp_now_ns();
    initialize_matrices(h_A, h_B, h_C, N);
    t1 = criu_exp_now_ns();
    metrics.host_initialization_ms = criu_exp_ms_between(t0, t1);

    t0 = criu_exp_now_ns();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), matrix_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), matrix_bytes));
    t1 = criu_exp_now_ns();
    metrics.device_allocation_ms = criu_exp_ms_between(t0, t1);

    t0 = criu_exp_now_ns();
    CUDA_CHECK(cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, matrix_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = criu_exp_now_ns();
    metrics.h2d_copy_ms = criu_exp_ms_between(t0, t1);

    if (criu_exp_write_progress(opts.progress_file, 0) != 0) {
        CRIU_EXP_LOG_WARN("Failed to write initial progress marker");
    }

    if (criu_exp_write_marker(opts.ready_file) != 0) {
        CRIU_EXP_LOG_WARN("Failed to write ready marker");
    } else {
        CRIU_EXP_LOG_INFO("Ready marker written: %s", opts.ready_file);
    }

    dim3 block(opts.block_size, opts.block_size);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    CRIU_EXP_LOG_INFO(
        "Launching repeated kernels: grid=(%u,%u) block=(%u,%u)",
        grid.x,
        grid.y,
        block.x,
        block.y
    );

    t0 = criu_exp_now_ns();

    for (int iter = 1; iter <= opts.iterations; ++iter) {
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        metrics.completed_iterations = iter;

        if (criu_exp_write_progress(opts.progress_file, iter) != 0) {
            CRIU_EXP_LOG_WARN("Failed to write progress marker for iteration %d", iter);
        }

        if (iter == 1 || iter == opts.iterations || iter % 10 == 0) {
            CRIU_EXP_LOG_INFO("Completed iteration %d/%d", iter, opts.iterations);
        }

        if (opts.sleep_ms_between_iterations > 0) {
            criu_exp_sleep_ms(opts.sleep_ms_between_iterations);
        }
    }

    t1 = criu_exp_now_ns();
    metrics.kernel_total_ms = criu_exp_ms_between(t0, t1);

    t0 = criu_exp_now_ns();
    CUDA_CHECK(cudaMemcpy(h_C, d_C, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = criu_exp_now_ns();
    metrics.d2h_copy_ms = criu_exp_ms_between(t0, t1);

    if (opts.verify) {
        t0 = criu_exp_now_ns();

        int verify_passed = verify_sampled_result(
            h_C,
            N,
            opts.verify_samples,
            &metrics.verification_failures
        );

        t1 = criu_exp_now_ns();
        metrics.verification_ms = criu_exp_ms_between(t0, t1);

        if (!verify_passed) {
            CRIU_EXP_LOG_ERROR("Verification failed with %d sampled mismatches",
                               metrics.verification_failures);
            goto cleanup;
        }
    } else {
        metrics.verification_ms = 0.0;
        metrics.verification_failures = 0;
    }

    metrics.passed = 1;
    exit_code = EXIT_SUCCESS;

cleanup:
    t0 = criu_exp_now_ns();

    if (d_A != nullptr) {
        cudaFree(d_A);
    }

    if (d_B != nullptr) {
        cudaFree(d_B);
    }

    if (d_C != nullptr) {
        cudaFree(d_C);
    }

    if (h_A != nullptr) {
        std::free(h_A);
    }

    if (h_B != nullptr) {
        std::free(h_B);
    }

    if (h_C != nullptr) {
        std::free(h_C);
    }

    cudaDeviceSynchronize();

    t1 = criu_exp_now_ns();
    metrics.cleanup_ms = criu_exp_ms_between(t0, t1);

    metrics.total_program_ms = criu_exp_elapsed_ms_since(total_start_ns);

    write_json_result(
        opts.output_json,
        opts,
        metrics,
        matrix_bytes,
        gpu_memory_allocated_bytes,
        host_memory_allocated_bytes,
        exit_code
    );

    if (criu_exp_write_marker(opts.done_file) != 0) {
        CRIU_EXP_LOG_WARN("Failed to write done marker");
    }

    if (metrics.passed) {
        CRIU_EXP_LOG_INFO("CUDA matrix multiplication benchmark completed successfully");
    } else {
        CRIU_EXP_LOG_ERROR("CUDA matrix multiplication benchmark failed");
    }

    return exit_code;
}