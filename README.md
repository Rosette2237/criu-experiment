# CRIU CPU-GPU Checkpointing Experiment

This repository contains a minimal experiment for comparing:

1. CPU-only execution with no checkpointing
2. CUDA GPU execution with no checkpointing
3. CPU-only CRIU checkpoint/restore
4. GPU CRIUgpu checkpoint/restore

The CPU workload is compiled with `g++` and never initializes CUDA. The GPU workload is compiled with `nvcc`, uses `cudaMalloc`, and avoids CUDA Unified Virtual Memory, CUDA IPC, and NCCL.

## Build

```bash
make
```
## Preflight

```bash
bash scripts/preflight.sh
```

## Baselines
```bash
bash scripts/run_cpu_baseline.sh
GPUS=0,1,2 bash scripts/run_gpu_baseline.sh
```

## CPU-only CRIU test
```bash
bash scripts/run_cpu_criu.sh
```

## GPU CRIUgpu tests
```bash
GPUS=0 bash scripts/run_gpu_criugpu.sh
GPUS=0,1 bash scripts/run_gpu_criugpu.sh
GPUS=0,1,2 bash scripts/run_gpu_criugpu.sh
```

## Run all
```bash
bash scripts/run_all.sh
```

## Useful knobs
```bash
N=1024
GPU_EXTRA_MB=2048
CPU_EXTRA_MB=512
PRE_SECONDS=10
POST_SECONDS=10
GPUS=0,1,2
```

For stronger GPU checkpoint pressure, increase `GPU_EXTRA_MB`, but keep per-GPU memory below the available 10 GiB device memory.


## First commands to run on the VM

```bash
cd criu-experiment
chmod +x scripts/*.sh
bash scripts/preflight.sh
make
bash scripts/run_cpu_baseline.sh
GPUS=0,1,2 bash scripts/run_gpu_baseline.sh
bash scripts/run_cpu_criu.sh
GPUS=0 bash scripts/run_gpu_criugpu.sh
GPUS=0,1 bash scripts/run_gpu_criugpu.sh
GPUS=0,1,2 bash scripts/run_gpu_criugpu.sh
python3 scripts/summarize.py
```

The main files to inspect after each run are `results/*_metrics.csv`, `results/*_app.csv`, and each `checkpoints/<run_id>/dump.log` / `restore.log`.