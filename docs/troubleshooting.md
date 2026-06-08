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