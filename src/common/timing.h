#ifndef CRIU_EXPERIMENT_TIMING_H
#define CRIU_EXPERIMENT_TIMING_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Return monotonic wall-clock time in nanoseconds.
 *
 * This is intended for benchmark wall-clock measurements around:
 *   - CUDA initialization
 *   - memory allocation
 *   - host-to-device copies
 *   - kernel execution
 *   - device-to-host copies
 *   - correctness verification
 *
 * CLOCK_MONOTONIC is used instead of CLOCK_REALTIME because benchmark timing
 * should not be affected by wall-clock time adjustments.
 */
static inline int64_t criu_exp_now_ns(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return -1;
    }

    return ((int64_t)ts.tv_sec * 1000000000LL) + (int64_t)ts.tv_nsec;
}

static inline double criu_exp_ns_to_ms(int64_t ns)
{
    return (double)ns / 1000000.0;
}

static inline double criu_exp_ms_between(int64_t start_ns, int64_t end_ns)
{
    if (start_ns < 0 || end_ns < 0 || end_ns < start_ns) {
        return -1.0;
    }

    return criu_exp_ns_to_ms(end_ns - start_ns);
}

static inline double criu_exp_elapsed_ms_since(int64_t start_ns)
{
    int64_t end_ns = criu_exp_now_ns();
    return criu_exp_ms_between(start_ns, end_ns);
}

static inline void criu_exp_sleep_ms(int milliseconds)
{
    if (milliseconds <= 0) {
        return;
    }

    struct timespec req;
    req.tv_sec = milliseconds / 1000;
    req.tv_nsec = (long)(milliseconds % 1000) * 1000000L;

    while (nanosleep(&req, &req) != 0) {
        /*
         * If interrupted by a signal, nanosleep updates req with the remaining
         * time. Continue until the requested sleep duration is completed.
         */
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CRIU_EXPERIMENT_TIMING_H */