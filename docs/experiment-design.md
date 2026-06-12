# Experiment Design

This document describes the benchmark design for testing CRIU checkpoint/restore behavior with a CUDA matrix multiplication workload.

## Goal

The goal is to evaluate whether a CUDA process can be checkpointed and restored under CRIU with NVIDIA GPU checkpoint support, and to measure the overhead of checkpointing and restoring a CUDA workload.

The experiment should answer:

1. Does the CUDA process checkpoint successfully?
2. Does the CUDA process restore successfully?
3. Does the restored CUDA process continue execution correctly?
4. How long does checkpointing take?
5. How long does restoring take?
6. How large are the CRIU checkpoint images?
7. How does overhead scale with matrix size and GPU memory allocation?

## Main workload

The main workload is a simple CUDA matrix multiplication benchmark.

The program performs repeated matrix multiplication:

```text
C = A x B
```

## Benchmark phases
Each CUDA checkpoint/restore experiment has the following phases.

### Phase 1: Environment capture

Before the benchmark starts, the script records:
```text
hostname
kernel version
OS release
NVIDIA driver version
CUDA version
CRIU version
GPU model
GPU memory
selected CUDA_VISIBLE_DEVICES value
```

This metadata is written to:

```text
results/raw/<run-id>/env.json
```

### Phase 2: Start CUDA workload

The script starts the CUDA matrix multiplication benchmark in the background.

The benchmark should:

1. Select the requested CUDA device
2. Allocate host matrices
3. Allocate GPU matrices
4. Copy inputs to GPU memory
5. Write a ready marker
6. Begin repeated matrix multiplication iterations

The ready marker tells the external script that the process has initialized CUDA and is safe to checkpoint.

Expected marker:

```text
/tmp/criu-experiment-matmul.ready
```

### Phase 3: Wait for checkpoint point

The script waits until the benchmark has completed a configurable number of iterations.

The benchmark periodically writes progress information to:

```text
/tmp/criu-experiment-matmul.progress
```

The progress file should contain the most recent completed iteration number.

Checkpointing should happen after:
```text
completed_iteration >= CHECKPOINT_AFTER_ITERATION
```

This is better than using a fixed sleep because it makes the checkpoint moment more reproducible.

### Phase 4: CUDA-aware checkpoint

The script checkpoints the running CUDA process.

The checkpoint operation should produce:
```text
results/raw/<run-id>/criu_dump.log
results/raw/<run-id>/checkpoint_size.txt
checkpoint-images/<run-id>/
```

The script should measure wall-clock dump time around the checkpoint operation.

### Phase 5: Restore

The script restores the checkpointed process.

The restore operation should produce:

```text
results/raw/<run-id>/criu_restore.log
```

The script should measure wall-clock restore time around the restore operation.

### Phase 6: Completion and correctness check

After restore, the CUDA benchmark should continue running until the requested number of iterations is complete.

At the end, the benchmark should:

1. Copy the final output matrix from GPU to host
2. Verify correctness using a deterministic reference check
3. Write benchmark timing and status information to JSON
4. Write a done marker

Expected marker:

```text
/tmp/criu-experiment-matmul.done
```

Expected JSON output:
```text
results/raw/<run-id>/checkpoint_restore.json
```

## Baseline experiments
The checkpoint/restore experiment should not be the first thing run.

Use the following order.

### CUDA-only baseline

Run the CUDA matrix multiplication benchmark without CRIU.

Purpose:

- Verify CUDA works
- Verify benchmark correctness
- Measure normal workload runtime

Expected result:
```text
passed = true
```

### CPU-only CRIU baseline

Run a CPU-only long-running process and checkpoint/restore it with CRIU.

Purpose:

- Verify CRIU works independently of CUDA
- Confirm permissions and kernel features are sufficient
- Produce a small known-good checkpoint/restore test

Expected result:
```text
restore_success = true
```

### CUDA CRIU single-GPU checkpoint/restore

Run the CUDA matrix multiplication benchmark, checkpoint it mid-execution, restore it, and verify final correctness.

Purpose:

- Test the actual target feature
- Measure checkpoint and restore overhead
- Capture failure logs if CUDA checkpointing does not work

Expected result:
```text
restore_success = true
cuda_correctness = true
```

### CUDA memory-scaling sweep

Repeat the CUDA checkpoint/restore experiment with multiple matrix sizes.

Purpose:

- Measure how dump and restore time scale with GPU allocation size
- Measure checkpoint image size
- Identify maximum practical matrix size for the VM

Example sizes:
```text
512
1024
2048
4096
```

The exact maximum size depends on available GPU memory.


## Metrics
Each run should collect the following metrics.

### Environment metrics
```text
hostname
timestamp_utc
kernel
os_release
nvidia_driver_version
cuda_version
criu_version
gpu_name
gpu_total_memory_mb
selected_gpu
```

### CUDA workload metrics
```text
matrix_size
iterations
block_size
gpu_memory_allocated_bytes
host_memory_allocated_bytes
cuda_init_ms
allocation_ms
h2d_copy_ms
kernel_total_ms
d2h_copy_ms
verification_ms
total_program_ms
passed
```

### Checkpoint/restore metrics
```text
process_pid
checkpoint_after_iteration
dump_wall_ms
restore_wall_ms
checkpoint_image_bytes
dump_exit_code
restore_exit_code
```

### Final derived metrics
```text
baseline_total_ms
checkpointed_total_ms
checkpoint_overhead_ms
restore_overhead_ms
total_overhead_ms
overhead_percent
```