# Experiment 001: CUDA Baseline

## Goal

Run the CUDA matrix multiplication benchmark without CRIU.

This experiment verifies that the CUDA workload works correctly before adding checkpoint/restore complexity.

## What this experiment tests

This experiment checks:

- CUDA device visibility
- CUDA runtime initialization
- GPU memory allocation
- Host-to-device copies
- Matrix multiplication kernel execution
- Device-to-host copies
- Correctness verification
- JSON result generation

## Command

From the repository root:

```bash
./scripts/02_run_cuda_baseline.sh
```

### Conservative first run

```bash
CUDA_VISIBLE_DEVICES=0 \
MATRIX_SIZE=512 \
ITERATIONS=30 \
BLOCK_SIZE=16 \
SLEEP_MS_BETWEEN_ITERATIONS=100 \
./scripts/02_run_cuda_baseline.sh
```

### Recommended normal run

```bash
CUDA_VISIBLE_DEVICES=0 \
MATRIX_SIZE=1024 \
ITERATIONS=60 \
BLOCK_SIZE=16 \
SLEEP_MS_BETWEEN_ITERATIONS=100 \
./scripts/02_run_cuda_baseline.sh
```

## Expected result

The run should finish successfully and produce:

- `results/raw/<run-id>/cuda_baseline.json`
- `results/raw/<run-id>/cuda_program_stdout.log`
- `results/raw/<run-id>/cuda_program_stderr.log`
- `results/raw/<run-id>/checkpoint_timing.env`
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

- `total_program_ms`
- `kernel_total_ms`
- `cuda_init_ms`
- `device_allocation_ms`
- `h2d_copy_ms`
- `d2h_copy_ms`
- `verification_ms`
- `gpu_memory_allocated_bytes`
- `host_memory_allocated_bytes`

## Success criteria

This experiment is successful if:

- The script exits with code 0
- `cuda_baseline.json` exists
- `passed` is `true`
- `completed_iterations` equals `ITERATIONS`
- No CUDA errors appear in `cuda_program_stderr.log`

## Failure debugging

If this experiment fails, inspect:

- `results/raw/<run-id>/env.json`
- `results/raw/<run-id>/cuda_program_stdout.log`
- `results/raw/<run-id>/cuda_program_stderr.log`
- `results/raw/<run-id>/nvidia_smi_before.txt`
- `results/raw/<run-id>/nvidia_smi_after.txt`

Useful commands:

```bash
nvidia-smi
nvcc --version
make cuda
```

## Common failure causes

### nvcc missing

The CUDA toolkit is not installed or not in PATH.

### cudaSetDevice failed

The selected CUDA device is invalid.

Try:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE=0
```

### cudaMalloc failed

The matrix size is too large or GPU memory is occupied.

Try:

```bash
MATRIX_SIZE=512 ./scripts/02_run_cuda_baseline.sh
```

### Correctness failed

This indicates a benchmark or CUDA execution problem. Do not proceed to CRIU tests until this is fixed.