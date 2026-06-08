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
