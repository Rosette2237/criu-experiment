# Experiment 003: CUDA CRIU Single GPU

## Goal

Run CRIU checkpoint/restore on the CUDA matrix multiplication benchmark on a single GPU.

This is the first CUDA-aware checkpoint/restore experiment. It verifies that a live CUDA process can be dumped mid-execution and restored to completion with correct output.

## What this experiment tests

This experiment checks:

- CUDA process checkpoint mid-execution
- CUDA state suspension before dump
- CRIU dump of a running CUDA process
- CRIU restore of the dumped CUDA process
- CUDA state resumption after restore
- Continued kernel execution and correctness verification after restore
- Dump and restore wall time measurement
- CUDA suspend and resume wall time measurement
- Checkpoint image size

## Command

From the repository root:

```bash
./scripts/04_run_cuda_checkpoint_restore.sh
```

### Conservative first run

```bash
CUDA_VISIBLE_DEVICES=0 \
MATRIX_SIZE=512 \
ITERATIONS=30 \
CHECKPOINT_AFTER_ITERATION=10 \
BLOCK_SIZE=16 \
SLEEP_MS_BETWEEN_ITERATIONS=100 \
./scripts/04_run_cuda_checkpoint_restore.sh
```

### Recommended normal run

```bash
CUDA_VISIBLE_DEVICES=0 \
MATRIX_SIZE=1024 \
ITERATIONS=60 \
CHECKPOINT_AFTER_ITERATION=20 \
BLOCK_SIZE=16 \
SLEEP_MS_BETWEEN_ITERATIONS=100 \
./scripts/04_run_cuda_checkpoint_restore.sh
```

## Expected result

The run should finish successfully and produce:

- `results/raw/<run-id>/checkpoint_restore.json`
- `results/raw/<run-id>/cuda_program_stdout.log`
- `results/raw/<run-id>/cuda_program_stderr.log`
- `results/raw/<run-id>/criu_dump.log`
- `results/raw/<run-id>/criu_restore.log`
- `results/raw/<run-id>/checkpoint_timing.env`
- `results/raw/<run-id>/checkpoint_size.txt`
- `results/raw/<run-id>/nvidia_smi_before.txt`
- `results/raw/<run-id>/nvidia_smi_during.txt`
- `results/raw/<run-id>/nvidia_smi_after.txt`
- `results/raw/<run-id>/env.json`

The JSON file should contain:

```json
{
  "program": "matmul_bench",
  "passed": true,
  "completed_iterations": 60
}
```

The exact number of completed iterations depends on the `ITERATIONS` value.

## Important metrics

The most important metrics are:

- `dump_wall_ms`
- `restore_wall_ms`
- `cuda_suspend_wall_ms`
- `cuda_resume_wall_ms`
- `cuda_program_total_wall_ms`
- `checkpoint_image_bytes`
- `completed_iterations`

## Success criteria

This experiment is successful if:

- The script exits with code 0
- `checkpoint_restore.json` exists
- `passed` is `true`
- `completed_iterations` equals `ITERATIONS`
- No CUDA errors appear in `cuda_program_stderr.log`
- No errors appear in `criu_dump.log` or `criu_restore.log`

## Failure debugging

If this experiment fails, inspect:

- `results/raw/<run-id>/env.json`
- `results/raw/<run-id>/cuda_program_stdout.log`
- `results/raw/<run-id>/cuda_program_stderr.log`
- `results/raw/<run-id>/criu_dump.log`
- `results/raw/<run-id>/criu_restore.log`
- `results/raw/<run-id>/nvidia_smi_during.txt`

Useful commands:

```bash
nvidia-smi
which cuda-checkpoint
cuda-checkpoint --help
sudo criu check --all
```

## Common failure causes

### cuda-checkpoint missing

`cuda-checkpoint` is not installed or not in PATH. The helper script will report this during environment checking.

### CRIU CUDA plugin unavailable

Inspect `criu_dump.log` for errors related to CUDA mappings or unsupported memory regions. Run:

```bash
sudo criu check --all
```

### Dump fails after CUDA initialization

This is the most important target failure class for this experiment. The CUDA process launched but CRIU could not dump it.

Try the smallest possible test:

```bash
MATRIX_SIZE=512 \
ITERATIONS=30 \
CHECKPOINT_AFTER_ITERATION=10 \
./scripts/04_run_cuda_checkpoint_restore.sh
```

### Restore fails after successful dump

Inspect `criu_restore.log` and `checkpoint_restore.json`. Confirm the restore runs on the same VM with the same visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### Restored program hangs

Check whether the process is alive and whether the GPU is active:

```bash
ps aux | grep matmul_bench | grep -v grep || true
nvidia-smi
```

### Correctness fails after restore

The process completed but output verification failed. Retry with a smaller matrix size and fewer iterations to isolate the failure.