# Experiment 004: CUDA CRIU Memory Scaling

## Goal

Run CUDA CRIU checkpoint/restore across a range of matrix sizes to measure how checkpoint/restore overhead scales with GPU memory footprint.

This experiment builds directly on experiment 003. It should only be attempted after a single-GPU checkpoint/restore succeeds for at least `MATRIX_SIZE=1024`.

## What this experiment tests

This experiment checks:

- Checkpoint/restore correctness at matrix sizes 512, 1024, 2048, and 4096
- Dump wall time as a function of matrix size
- Restore wall time as a function of matrix size
- CUDA suspend and resume wall time as a function of matrix size
- Checkpoint image size as a function of matrix size
- Maximum matrix size that can be checkpointed and restored successfully

## Command

From the repository root:

```bash
./scripts/05_run_matrix_sweep.sh
```

### Conservative first run

```bash
MATRIX_SIZES="512 1024" \
SWEEP_ITERATIONS=30 \
SWEEP_CHECKPOINT_AFTER_ITERATION=10 \
./scripts/05_run_matrix_sweep.sh
```

### Recommended normal run

```bash
MATRIX_SIZES="512 1024 2048 4096" \
SWEEP_ITERATIONS=60 \
SWEEP_CHECKPOINT_AFTER_ITERATION=20 \
./scripts/05_run_matrix_sweep.sh
```

## Expected result

The sweep should finish successfully and produce a sweep run directory:

```
results/raw/<sweep-run-id>/
```

Key output files:

- `results/raw/<sweep-run-id>/sweep_summary.csv`
- `results/raw/<sweep-run-id>/sweep.log`
- `results/raw/<sweep-run-id>/matrix_<N>_stdout.log`
- `results/raw/<sweep-run-id>/matrix_<N>_stderr.log`

Each matrix size also produces its own child run directory under `results/raw/`.

The CSV file contains one row per matrix size with columns:

- `sweep_run_id`, `matrix_size`, `iterations`, `checkpoint_after_iteration`
- `status`, `child_run_id`, `child_run_dir`
- `dump_wall_ms`, `restore_wall_ms`
- `cuda_suspend_wall_ms`, `cuda_resume_wall_ms`
- `cuda_program_total_wall_ms`, `checkpoint_image_bytes`
- `completed_iterations`, `passed`

## Important metrics

The most important metrics are:

- `dump_wall_ms` per matrix size
- `restore_wall_ms` per matrix size
- `cuda_suspend_wall_ms` per matrix size
- `cuda_resume_wall_ms` per matrix size
- `checkpoint_image_bytes` per matrix size

## GPU memory estimate

For square matrix multiplication with three single-precision matrices, approximate GPU matrix allocation:

| N    | Approximate matrix allocation |
|------|-------------------------------|
| 512  | ~3 MiB                        |
| 1024 | ~12 MiB                       |
| 2048 | ~48 MiB                       |
| 4096 | ~192 MiB                      |

Actual GPU memory usage is higher due to CUDA context, runtime, and driver overhead.

## Success criteria

This experiment is successful if:

- The script exits with code 0
- `sweep_summary.csv` contains one row per matrix size
- All rows show `status=success` and `passed=true`
- `completed_iterations` equals `SWEEP_ITERATIONS` for every row
- No errors appear in any `matrix_<N>_stderr.log`

## Failure debugging

If a specific matrix size fails, inspect the per-size logs in the sweep run directory:

```
results/raw/<sweep-run-id>/matrix_<N>_stdout.log
results/raw/<sweep-run-id>/matrix_<N>_stderr.log
```

Then inspect the child run directory for that size:

```
results/raw/<child-run-id>/criu_dump.log
results/raw/<child-run-id>/criu_restore.log
results/raw/<child-run-id>/cuda_program_stderr.log
results/raw/<child-run-id>/nvidia_smi_before.txt
```

Useful commands:

```bash
nvidia-smi
du -sh checkpoint-images/*
cat results/raw/<sweep-run-id>/sweep_summary.csv
```

## Common failure causes

### GPU memory exhausted at large matrix size

Reduce the sweep range and work up incrementally:

```bash
MATRIX_SIZES="512 1024" ./scripts/05_run_matrix_sweep.sh
```

Then add `2048` only after `1024` succeeds, and so on.

### Stale checkpoint images from a previous run

Old checkpoint images can interfere with a new run. Clean them before rerunning:

```bash
rm -rf checkpoint-images/*
```

### Stale marker files from a previous run

Remove marker files before rerunning:

```bash
rm -f /tmp/criu-experiment-*.ready
rm -f /tmp/criu-experiment-*.progress
rm -f /tmp/criu-experiment-*.done
rm -f /tmp/criu-experiment-*.json.tmp
```

### One size fails mid-sweep

The sweep script continues past failures and records each result in the CSV. Check `sweep_summary.csv` to identify which sizes passed and which failed, then inspect the corresponding per-size logs.