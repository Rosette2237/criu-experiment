# Troubleshooting

This document lists common failure modes for the CRIU CUDA checkpoint/restore experiment and the files or commands to inspect.

## General debugging order

Use this order when debugging:

```bash
./scripts/00_env_check.sh
./scripts/01_build.sh
./scripts/02_run_cuda_baseline.sh
./scripts/03_run_cpu_criu_baseline.sh
./scripts/04_run_cuda_checkpoint_restore.sh
```

Do not debug CUDA checkpoint/restore first.

The correct dependency order is:

1. Linux VM works
2. NVIDIA driver works
3. CUDA compiler works
4. CUDA benchmark works
5. CRIU works on CPU-only process
6. CRIU CUDA checkpoint/restore works

## Important output files

After a failed run, inspect the run directory:

```
results/raw/<run-id>/
```

Useful files:

- `env.json`
- `cuda_program_stdout.log`
- `cuda_program_stderr.log`
- `cpu_program_stdout.log`
- `cpu_program_stderr.log`
- `criu_dump.log`
- `criu_restore.log`
- `checkpoint_restore.json`
- `cuda_baseline.json`
- `cpu_criu_baseline.json`
- `checkpoint_size.txt`
- `nvidia_smi_before.txt`
- `nvidia_smi_after.txt`

## Environment check failures

### nvidia-smi is missing

Check:

```bash
which nvidia-smi
nvidia-smi
```

Possible causes:

- NVIDIA driver is not installed
- NVIDIA driver is not loaded
- VM does not have GPU attached
- PATH does not include NVIDIA utility directory

### No GPU appears in nvidia-smi

Check:

```bash
lspci | grep -i nvidia || true
nvidia-smi
```

Possible causes:

- GPU was not attached to the VM
- NVIDIA driver failed to load
- VM flavor does not expose GPU
- Cloud allocation issue

### nvcc is missing

Check:

```bash
which nvcc
nvcc --version
```

Possible causes:

- CUDA toolkit is not installed
- Only NVIDIA driver is installed
- PATH does not include CUDA bin directory

Common PATH fix:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

### CUDA version mismatch warning

`nvidia-smi` reports the maximum CUDA version supported by the driver.

`nvcc --version` reports the installed CUDA toolkit version.

These do not need to be identical, but the toolkit version must be compatible with the installed driver.

## Build failures

### make cuda fails with nvcc: command not found

Check:

```bash
which nvcc
echo "$PATH"
```

Fix the CUDA toolkit installation or update PATH.

### Unsupported GPU architecture

If compilation fails because of the CUDA architecture setting, override the default:

```bash
make cuda CUDA_ARCH=sm_80
```

For A100-class GPUs, use `sm_80`.

### Missing source files

If Make reports missing files, confirm the repo structure:

```bash
find . -maxdepth 3 -type f | sort
```

The required source files include:

- `src/cuda_matmul/matmul_bench.cu`
- `src/cuda_matmul/matmul_config.h`
- `src/cpu_baseline/cpu_sleep_loop.c`
- `src/common/timing.h`
- `src/common/logging.h`
- `src/common/signal_markers.h`

## CUDA baseline failures

### cudaSetDevice fails

Check:

```bash
nvidia-smi
echo "$CUDA_VISIBLE_DEVICES"
```

Try forcing a single visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### cudaMalloc fails

Possible causes:

- Matrix size is too large
- Another process is using GPU memory
- CUDA context overhead leaves less memory than expected

Check GPU memory:

```bash
nvidia-smi
```

Reduce matrix size and retry:

```bash
export MATRIX_SIZE=1024
./scripts/02_run_cuda_baseline.sh
```

### Kernel launch fails

Check:

```
results/raw/<run-id>/cuda_program_stderr.log
```

Possible causes:

- Invalid block size
- Invalid grid size
- Earlier CUDA API error
- GPU reset or driver issue

Try conservative settings:

```bash
export MATRIX_SIZE=1024
export BLOCK_SIZE=16
export ITERATIONS=20
```

### Correctness verification fails

Possible causes:

- Kernel bug
- Floating-point tolerance too strict
- Memory copy failed
- Output buffer was corrupted
- Restore resumed incorrectly

For the first benchmark, correctness should use deterministic inputs and a lightweight sampled verification method.

## CRIU CPU baseline failures

### criu is missing

Check:

```bash
which criu
criu --version
```

Install CRIU before continuing.

### criu check fails

Run:

```bash
sudo criu check
sudo criu check --all
```

Common causes:

- Insufficient kernel feature support
- Missing permissions
- LSM restrictions
- Unsupported namespace configuration

### Dump fails with permission denied

Use `sudo` for CRIU commands:

```bash
sudo criu dump ...
```

Confirm `sudo` works:

```bash
sudo true
```

### Restore fails immediately

Inspect:

```
results/raw/<run-id>/criu_restore.log
```

Common causes:

- Wrong checkpoint image directory
- Original process did not dump successfully
- Files or directories expected by the process disappeared
- Permission issue

## CUDA checkpoint/restore failures

### cuda-checkpoint is missing

Check:

```bash
which cuda-checkpoint
cuda-checkpoint --help
```

If it is missing, either:

- The installed NVIDIA stack does not provide it
- The binary is not in PATH
- The driver/toolkit package is incomplete

The helper script should detect this and print a clear error.

### CRIU CUDA plugin is unavailable

Symptoms may include CRIU dump errors related to CUDA, GPU memory, or unsupported mappings.

Check:

```bash
criu --version
sudo criu check --all
```

Also inspect:

```
results/raw/<run-id>/criu_dump.log
```

### Dump fails after CUDA initialization

This is the most important target failure class.

Inspect:

```
results/raw/<run-id>/criu_dump.log
results/raw/<run-id>/cuda_program_stdout.log
results/raw/<run-id>/cuda_program_stderr.log
```

Possible causes:

- Driver does not support CUDA checkpointing
- CRIU version does not include CUDA plugin support
- CUDA process was not prepared for checkpoint correctly
- `cuda-checkpoint` command failed
- Unsupported CUDA feature was used
- File descriptor or memory mapping unsupported by CRIU

Try the smallest CUDA checkpoint test:

```bash
export MATRIX_SIZE=512
export ITERATIONS=30
export CHECKPOINT_AFTER_ITERATION=10
export BLOCK_SIZE=16
./scripts/04_run_cuda_checkpoint_restore.sh
```

### Restore fails after successful dump

Inspect:

```
results/raw/<run-id>/criu_restore.log
results/raw/<run-id>/checkpoint_restore.json
```

Possible causes:

- CUDA state could not be recreated
- GPU changed between dump and restore
- Device ordering changed
- `CUDA_VISIBLE_DEVICES` changed
- NVIDIA driver/runtime mismatch
- Checkpoint image directory is incomplete

Make sure restore is done on the same VM with the same visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

### Restored program hangs

Check whether the process exists:

```bash
ps aux | grep matmul_bench | grep -v grep || true
```

Check GPU activity:

```bash
nvidia-smi
```

Check progress marker:

```bash
cat /tmp/criu-experiment-matmul.progress || true
```

Possible causes:

- Process restored but CUDA context did not resume correctly
- Program is stuck at CUDA synchronization
- Output redirection or terminal handling issue
- CRIU restored process but parent script lost track of it

### Program completes but correctness fails after restore

This means the restore path worked enough to continue execution, but data integrity may have failed.

Inspect:

```
results/raw/<run-id>/checkpoint_restore.json
results/raw/<run-id>/cuda_program_stdout.log
results/raw/<run-id>/cuda_program_stderr.log
```

Possible causes:

- GPU memory was not restored correctly
- Host-side benchmark state changed
- Progress or iteration state was inconsistent
- Kernel output was overwritten
- Verification method exposed a real data mismatch

Retry with smaller matrix size:

```bash
export MATRIX_SIZE=512
export ITERATIONS=30
export CHECKPOINT_AFTER_ITERATION=10
./scripts/04_run_cuda_checkpoint_restore.sh
```

## Checkpoint image size issues

If checkpoint images are unexpectedly large, inspect:

```bash
du -sh checkpoint-images/*
```

Possible causes:

- Large host memory allocation
- GPU memory copied back into checkpointable memory
- Large logs or temporary files included
- Multiple stale checkpoint directories

Clean old checkpoint images:

```bash
rm -rf checkpoint-images/*
```

## Stale marker files

Stale marker files can confuse scripts. Remove them before rerunning:

```bash
rm -f /tmp/criu-experiment-*.ready
rm -f /tmp/criu-experiment-*.progress
rm -f /tmp/criu-experiment-*.done
rm -f /tmp/criu-experiment-*.json.tmp
```

## Stale processes

Check for old benchmark processes:

```bash
ps aux | grep -E "matmul_bench|cpu_sleep_loop" | grep -v grep || true
```

Terminate stale benchmark processes if needed:

```bash
pkill -f matmul_bench || true
pkill -f cpu_sleep_loop || true
```

## GPU reset symptoms

If CUDA commands start failing after a bad restore, inspect:

```bash
nvidia-smi
dmesg | tail -100
```

On a shared VM or vGPU environment, you may not be able to reset the GPU manually.

If allowed, reset only when no other process is using the GPU:

```bash
sudo nvidia-smi --gpu-reset -i 0
```

If reset is not permitted, restart the VM.

## Minimum known-good test values

Use these values for the first successful CUDA checkpoint/restore attempt:

```bash
export CUDA_VISIBLE_DEVICES=0
export MATRIX_SIZE=512
export ITERATIONS=30
export CHECKPOINT_AFTER_ITERATION=10
export BLOCK_SIZE=16
./scripts/04_run_cuda_checkpoint_restore.sh
```

Then increase to:

```bash
export MATRIX_SIZE=1024
export ITERATIONS=60
export CHECKPOINT_AFTER_ITERATION=20
```

Then run the sweep.

## Reporting failures

When recording a failure, capture:

- Date and time
- VM type
- Kernel version
- NVIDIA driver version
- CUDA toolkit version
- CRIU version
- GPU model
- Exact command run
- Run directory
- `criu_dump.log`
- `criu_restore.log`
- `cuda_program_stdout.log`
- `cuda_program_stderr.log`

The most useful single file for sharing a failed run is:

```
results/raw/<run-id>/env.json
```