# CRIU CUDA GPU Checkpoint Experiment

This repository contains a small CUDA benchmark and shell-based experiment harness for testing checkpoint/restore behavior with CRIU and NVIDIA GPU checkpoint support.

The first target workload is a simple CUDA matrix multiplication benchmark. The benchmark is designed to run long enough for an external script to checkpoint the process in the middle of execution, restore it, and verify that the CUDA workload can continue correctly.

## Repository goal

The goal is to measure and debug:

1. CUDA-only baseline runtime
2. CPU-only CRIU checkpoint/restore sanity check
3. CUDA process checkpoint time
4. CUDA process restore time
5. Total checkpoint/restore overhead
6. Checkpoint image size
7. Post-restore CUDA correctness

## Expected environment

The intended environment is a Linux VM with:

- NVIDIA GPU available through `nvidia-smi`
- NVIDIA driver with CUDA checkpoint support
- CUDA compiler `nvcc`
- CRIU with CUDA plugin support
- `cuda-checkpoint` utility available, if required by the installed NVIDIA stack
- Bash
- Python 3 for result parsing

The current experiment target is single-node, same-machine checkpoint/restore.

## Initial experiment order

Run the experiments in this order:

```bash
./scripts/00_env_check.sh
./scripts/01_build.sh
./scripts/02_run_cuda_baseline.sh
./scripts/03_run_cpu_criu_baseline.sh
./scripts/04_run_cuda_checkpoint_restore.sh
./scripts/05_run_matrix_sweep.sh
./scripts/06_collect_results.sh