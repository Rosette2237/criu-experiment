# Experiment 002: CPU CRIU Baseline

## Goal

Run CRIU checkpoint/restore on a CPU-only process before testing CUDA-aware checkpoint/restore.

This experiment verifies that CRIU works correctly on the VM in isolation from any GPU complexity.

## What this experiment tests

This experiment checks:

- CRIU availability and basic feature check
- Process dump of a running CPU-only loop
- CRIU restore of the dumped process
- Marker file communication (ready, progress, done)
- Checkpoint image generation and sizing
- JSON result generation after restore

## Command

From the repository root:

```bash
./scripts/03_run_cpu_criu_baseline.sh
```

### Conservative first run

```bash
CPU_ITERATIONS=60 \
CPU_SLEEP_MS=500 \
CHECKPOINT_AFTER_ITERATION=10 \
./scripts/03_run_cpu_criu_baseline.sh
```

### Recommended normal run

```bash
CPU_ITERATIONS=120 \
CPU_SLEEP_MS=500 \
CHECKPOINT_AFTER_ITERATION=20 \
./scripts/03_run_cpu_criu_baseline.sh
```

## Expected result

The run should finish successfully and produce:

- `results/raw/<run-id>/cpu_criu_baseline.json`
- `results/raw/<run-id>/cpu_program_stdout.log`
- `results/raw/<run-id>/cpu_program_stderr.log`
- `results/raw/<run-id>/criu_dump.log`
- `results/raw/<run-id>/criu_restore.log`
- `results/raw/<run-id>/checkpoint_timing.env`
- `results/raw/<run-id>/checkpoint_size.txt`
- `results/raw/<run-id>/env.json`

The JSON file should contain:

```json
{
  "program": "cpu_sleep_loop",
  "passed": true,
  "completed_iterations": 120
}
```

The exact number of completed iterations depends on the `CPU_ITERATIONS` value.

## Important metrics

The most important metrics are:

- `dump_wall_ms`
- `restore_wall_ms`
- `checkpoint_image_bytes`
- `total_program_ms`
- `completed_iterations`

## Success criteria

This experiment is successful if:

- The script exits with code 0
- `cpu_criu_baseline.json` exists
- `passed` is `true`
- `completed_iterations` equals `CPU_ITERATIONS`
- No errors appear in `criu_dump.log` or `criu_restore.log`

## Failure debugging

If this experiment fails, inspect:

- `results/raw/<run-id>/env.json`
- `results/raw/<run-id>/cpu_program_stdout.log`
- `results/raw/<run-id>/cpu_program_stderr.log`
- `results/raw/<run-id>/criu_dump.log`
- `results/raw/<run-id>/criu_restore.log`

Useful commands:

```bash
which criu
criu --version
sudo criu check
sudo criu check --all
```

## Common failure causes

### criu missing

CRIU is not installed or not in PATH. Install CRIU before continuing.

### criu check fails

Run:

```bash
sudo criu check --all
```

Common causes: insufficient kernel features, LSM restrictions, or missing permissions.

### Dump fails with permission denied

CRIU requires elevated permissions. Confirm `sudo` works:

```bash
sudo true
```

All CRIU commands in this repository run through `sudo` by default.

### Restore fails immediately

Inspect:

```
results/raw/<run-id>/criu_restore.log
```

Common causes: wrong checkpoint image directory, original process exited before dump completed, or a permission issue.