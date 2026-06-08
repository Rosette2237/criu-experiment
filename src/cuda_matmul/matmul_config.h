#ifndef CRIU_EXPERIMENT_MATMUL_CONFIG_H
#define CRIU_EXPERIMENT_MATMUL_CONFIG_H

/*
 * Default CUDA matrix multiplication benchmark configuration.
 *
 * These defaults are intentionally conservative so the first checkpoint/restore
 * attempt has a high chance of reaching the checkpoint point before exhausting
 * GPU memory or taking too long.
 *
 * Runtime values can be overridden by command-line arguments in matmul_bench.cu.
 */

#define CRIU_EXP_DEFAULT_MATRIX_SIZE 1024
#define CRIU_EXP_DEFAULT_ITERATIONS 60
#define CRIU_EXP_DEFAULT_BLOCK_SIZE 16
#define CRIU_EXP_DEFAULT_DEVICE_ID 0
#define CRIU_EXP_DEFAULT_SLEEP_MS_BETWEEN_ITERATIONS 100
#define CRIU_EXP_DEFAULT_VERIFY 1

/*
 * Default marker paths.
 *
 * The shell scripts wait on these files to coordinate checkpoint timing:
 *
 *   ready    -> CUDA context and GPU allocations are complete
 *   progress -> latest completed iteration number
 *   done     -> benchmark completed successfully or failed cleanly
 */

#define CRIU_EXP_DEFAULT_READY_FILE "/tmp/criu-experiment-matmul.ready"
#define CRIU_EXP_DEFAULT_PROGRESS_FILE "/tmp/criu-experiment-matmul.progress"
#define CRIU_EXP_DEFAULT_DONE_FILE "/tmp/criu-experiment-matmul.done"

/*
 * Default output path.
 *
 * The scripts normally override this so each run writes into:
 *
 *   results/raw/<run-id>/cuda_baseline.json
 *   results/raw/<run-id>/checkpoint_restore.json
 */

#define CRIU_EXP_DEFAULT_OUTPUT_JSON "/tmp/criu-experiment-matmul.json"

/*
 * Verification configuration.
 *
 * Full CPU reference matrix multiplication is expensive for large matrices.
 * The benchmark uses deterministic inputs and verifies a sampled set of output
 * elements instead.
 */

#define CRIU_EXP_DEFAULT_VERIFY_SAMPLES 64
#define CRIU_EXP_VERIFY_ABS_TOLERANCE 1.0e-2f
#define CRIU_EXP_VERIFY_REL_TOLERANCE 1.0e-3f

/*
 * Matrix initialization constants.
 *
 * The input matrices are initialized with deterministic formulas. Keeping the
 * values small helps avoid large floating-point error accumulation in the
 * simple matrix multiplication kernel.
 */

#define CRIU_EXP_MATRIX_A_MODULUS 17
#define CRIU_EXP_MATRIX_B_MODULUS 13
#define CRIU_EXP_MATRIX_A_SCALE 0.01f
#define CRIU_EXP_MATRIX_B_SCALE 0.02f

/*
 * Hard safety limits.
 *
 * These are not hardware limits. They are sanity guards to catch accidental
 * invalid arguments before trying to allocate huge buffers.
 */

#define CRIU_EXP_MIN_MATRIX_SIZE 16
#define CRIU_EXP_MAX_MATRIX_SIZE 32768

#define CRIU_EXP_MIN_ITERATIONS 1
#define CRIU_EXP_MAX_ITERATIONS 1000000

#define CRIU_EXP_MIN_BLOCK_SIZE 1
#define CRIU_EXP_MAX_BLOCK_SIZE 32

#define CRIU_EXP_MIN_VERIFY_SAMPLES 1
#define CRIU_EXP_MAX_VERIFY_SAMPLES 4096

#endif /* CRIU_EXPERIMENT_MATMUL_CONFIG_H */