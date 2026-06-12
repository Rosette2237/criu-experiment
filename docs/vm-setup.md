# VM Setup

This document describes the expected VM setup for running the CRIU CUDA checkpoint/restore experiment.

The repository assumes the VM already has an NVIDIA GPU available and that the NVIDIA driver has been updated to a version that supports CUDA checkpoint/restore integration.

## Target VM assumptions

The initial target environment is a Linux VM with:

- NVIDIA GPU visible through `nvidia-smi`
- NVIDIA driver 570 or newer
- CUDA toolkit installed
- `nvcc` available
- CRIU installed
- CRIU CUDA plugin support available
- Bash shell
- Python 3
- Sudo access

The first experiment target is same-machine checkpoint/restore.

This means the CUDA process is checkpointed and restored on the same VM using the same GPU environment.

## Expected GPU environment

### Check GPU visibility

```bash
nvidia-smi
```

Expected result: at least one NVIDIA GPU should be listed.

For the provided VM class, the GPU list may look similar to:

```
GRID A100X-10C
```

The first experiments should use only one GPU.

Recommended first setting:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not start with multi-GPU testing. Multi-GPU testing should only be attempted after single-GPU checkpoint/restore works.

### Check CUDA compiler

Check whether `nvcc` is available:

```bash
which nvcc
nvcc --version
```

If `nvcc` is missing, install the CUDA toolkit that matches the VM's driver compatibility.

The benchmark is compiled with:

```bash
make cuda
```

The default CUDA architecture in this repository is `sm_80`. This is appropriate for A100-class GPUs.

### Check CRIU

Check whether CRIU is available:

```bash
which criu
criu --version
```

Run CRIU's basic feature check:

```bash
sudo criu check
```

For more verbose diagnostics:

```bash
sudo criu check --all
```

If this fails, fix the CRIU environment before attempting CUDA checkpoint/restore.

### Check CUDA checkpoint utility

Some NVIDIA CUDA checkpoint workflows use the `cuda-checkpoint` command-line utility.

Check whether it exists:

```bash
which cuda-checkpoint
cuda-checkpoint --help
```

If `cuda-checkpoint` is not found, the CUDA checkpoint helper scripts in this repository will report it during environment checking.

## Kernel and permission checks

CRIU commonly requires elevated permissions. Most CRIU commands in this repository are expected to run through `sudo`.

Check the kernel version:

```bash
uname -a
```

Check the current user:

```bash
whoami
id
```

Check whether `sudo` works:

```bash
sudo true
```

## Recommended package checks

Check required tools:

```bash
bash --version
python3 --version
make --version
gcc --version
```

Optional but useful tools:

```bash
which jq || true
which bc || true
which awk
which sed
which grep
which date
```

The scripts are written to avoid requiring too many optional dependencies, but `jq` and `bc` are useful when available.

## Repository setup

From inside the repository root:

```bash
cd criu-experiment
```

Create the runtime directories if they do not already exist:

```bash
mkdir -p build/bin
mkdir -p results/raw
mkdir -p results/parsed
mkdir -p results/logs
mkdir -p results/figures
mkdir -p checkpoint-images
mkdir -p tmp
```

Build the benchmarks:

```bash
make all
```

## First run order

Run the experiments in this order:

```bash
./scripts/00_env_check.sh
./scripts/01_build.sh
./scripts/02_run_cuda_baseline.sh
./scripts/03_run_cpu_criu_baseline.sh
./scripts/04_run_cuda_checkpoint_restore.sh
```

Do not start directly with the CUDA checkpoint/restore test.

The CPU-only CRIU baseline is important because it separates general CRIU problems from CUDA-specific problems.

## Recommended initial configuration

For the first CUDA baseline run:

```bash
export CUDA_VISIBLE_DEVICES=0
export MATRIX_SIZE=1024
export ITERATIONS=60
export BLOCK_SIZE=16
```

For the first CUDA checkpoint/restore run:

```bash
export CUDA_VISIBLE_DEVICES=0
export MATRIX_SIZE=1024
export ITERATIONS=60
export CHECKPOINT_AFTER_ITERATION=20
export BLOCK_SIZE=16
```

These values are intentionally conservative.

After the first successful restore, increase the matrix size.

Suggested sweep:

- 512
- 1024
- 2048
- 4096

## GPU memory estimate

For square matrix multiplication with three single-precision matrices:

- A: N × N × 4 bytes
- B: N × N × 4 bytes
- C: N × N × 4 bytes

Approximate GPU memory for matrix data: `3 × N × N × 4 bytes`

| N    | Approximate memory |
|------|--------------------|
| 1024 | ~12 MiB            |
| 2048 | ~48 MiB            |
| 4096 | ~192 MiB           |
| 8192 | ~768 MiB           |

Actual memory usage can be higher due to CUDA context, runtime allocations, page tables, and driver overhead.

## Files to inspect after failure

If an experiment fails, inspect:

```
results/raw/<run-id>/env.json
results/raw/<run-id>/cuda_program_stdout.log
results/raw/<run-id>/cuda_program_stderr.log
results/raw/<run-id>/criu_dump.log
results/raw/<run-id>/criu_restore.log
results/raw/<run-id>/checkpoint_size.txt
```

Also check the GPU state:

```bash
nvidia-smi
```

## Clean up

Remove build artifacts:

```bash
make clean
```

Remove build artifacts, result files, and checkpoint images:

```bash
make distclean
```

Remove temporary marker files:

```bash
rm -f /tmp/criu-experiment-*.ready
rm -f /tmp/criu-experiment-*.progress
rm -f /tmp/criu-experiment-*.done
rm -f /tmp/criu-experiment-*.json.tmp
```
