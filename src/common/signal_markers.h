#ifndef CRIU_EXPERIMENT_SIGNAL_MARKERS_H
#define CRIU_EXPERIMENT_SIGNAL_MARKERS_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Marker-file helpers.
 *
 * The benchmark programs use marker files so that shell scripts can coordinate
 * checkpoint timing without relying on arbitrary sleep durations.
 *
 * Typical CUDA marker files:
 *
 *   /tmp/criu-experiment-matmul.ready
 *   /tmp/criu-experiment-matmul.progress
 *   /tmp/criu-experiment-matmul.done
 *
 * Typical CPU marker files:
 *
 *   /tmp/criu-experiment-cpu.ready
 *   /tmp/criu-experiment-cpu.progress
 *   /tmp/criu-experiment-cpu.done
 */

static inline int criu_exp_write_text_file(const char *path, const char *text)
{
    if (path == NULL || text == NULL) {
        errno = EINVAL;
        return -1;
    }

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }

    if (fputs(text, fp) == EOF) {
        int saved_errno = errno;
        fclose(fp);
        errno = saved_errno;
        return -1;
    }

    if (fflush(fp) != 0) {
        int saved_errno = errno;
        fclose(fp);
        errno = saved_errno;
        return -1;
    }

    if (fsync(fileno(fp)) != 0) {
        int saved_errno = errno;
        fclose(fp);
        errno = saved_errno;
        return -1;
    }

    if (fclose(fp) != 0) {
        return -1;
    }

    return 0;
}

static inline int criu_exp_write_marker(const char *path)
{
    return criu_exp_write_text_file(path, "1\n");
}

static inline int criu_exp_write_progress(const char *path, int iteration)
{
    if (path == NULL) {
        errno = EINVAL;
        return -1;
    }

    char buffer[64];
    int written = snprintf(buffer, sizeof(buffer), "%d\n", iteration);

    if (written < 0 || (size_t)written >= sizeof(buffer)) {
        errno = EOVERFLOW;
        return -1;
    }

    return criu_exp_write_text_file(path, buffer);
}

static inline int criu_exp_read_progress(const char *path, int *iteration_out)
{
    if (path == NULL || iteration_out == NULL) {
        errno = EINVAL;
        return -1;
    }

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }

    int value = -1;
    int scanned = fscanf(fp, "%d", &value);

    if (fclose(fp) != 0) {
        return -1;
    }

    if (scanned != 1) {
        errno = EINVAL;
        return -1;
    }

    *iteration_out = value;
    return 0;
}

static inline int criu_exp_file_exists(const char *path)
{
    if (path == NULL) {
        return 0;
    }

    return access(path, F_OK) == 0;
}

static inline int criu_exp_remove_file_if_exists(const char *path)
{
    if (path == NULL) {
        errno = EINVAL;
        return -1;
    }

    if (unlink(path) == 0) {
        return 0;
    }

    if (errno == ENOENT) {
        return 0;
    }

    return -1;
}

static inline int criu_exp_clear_markers(
    const char *ready_path,
    const char *progress_path,
    const char *done_path
)
{
    int status = 0;

    if (ready_path != NULL && criu_exp_remove_file_if_exists(ready_path) != 0) {
        status = -1;
    }

    if (progress_path != NULL && criu_exp_remove_file_if_exists(progress_path) != 0) {
        status = -1;
    }

    if (done_path != NULL && criu_exp_remove_file_if_exists(done_path) != 0) {
        status = -1;
    }

    return status;
}

#ifdef __cplusplus
}
#endif

#endif /* CRIU_EXPERIMENT_SIGNAL_MARKERS_H */