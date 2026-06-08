#ifndef CRIU_EXPERIMENT_LOGGING_H
#define CRIU_EXPERIMENT_LOGGING_H

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lightweight logging helpers shared by the CUDA benchmark and CPU baseline.
 *
 * Logs are intentionally written to stderr so that experiment scripts can
 * redirect program output into:
 *
 *   results/raw/<run-id>/*_stdout.log
 *   results/raw/<run-id>/*_stderr.log
 */

static inline void criu_exp_timestamp(char *buffer, size_t buffer_size)
{
    if (buffer == NULL || buffer_size == 0) {
        return;
    }

    time_t now = time(NULL);
    struct tm tm_value;

#if defined(_POSIX_VERSION)
    localtime_r(&now, &tm_value);
#else
    struct tm *tmp = localtime(&now);
    if (tmp == NULL) {
        snprintf(buffer, buffer_size, "unknown-time");
        return;
    }
    tm_value = *tmp;
#endif

    strftime(buffer, buffer_size, "%Y-%m-%d %H:%M:%S", &tm_value);
}

static inline void criu_exp_log_impl(
    const char *level,
    const char *file,
    int line,
    const char *fmt,
    ...
)
{
    char timestamp[64];
    criu_exp_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr, "[%s] [%s] [%s:%d] ", timestamp, level, file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
}

#define CRIU_EXP_LOG_INFO(...) \
    criu_exp_log_impl("INFO", __FILE__, __LINE__, __VA_ARGS__)

#define CRIU_EXP_LOG_WARN(...) \
    criu_exp_log_impl("WARN", __FILE__, __LINE__, __VA_ARGS__)

#define CRIU_EXP_LOG_ERROR(...) \
    criu_exp_log_impl("ERROR", __FILE__, __LINE__, __VA_ARGS__)

#define CRIU_EXP_LOG_ERRNO(message) \
    criu_exp_log_impl( \
        "ERROR", \
        __FILE__, \
        __LINE__, \
        "%s: errno=%d (%s)", \
        (message), \
        errno, \
        strerror(errno) \
    )

static inline void criu_exp_die(const char *message)
{
    CRIU_EXP_LOG_ERROR("%s", message);
    exit(EXIT_FAILURE);
}

static inline void criu_exp_die_errno(const char *message)
{
    CRIU_EXP_LOG_ERRNO(message);
    exit(EXIT_FAILURE);
}

#ifdef __cplusplus
}
#endif

#endif /* CRIU_EXPERIMENT_LOGGING_H */