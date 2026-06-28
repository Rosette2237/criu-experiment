# CRIU CPU-GPU Checkpointing Experiment

This repository contains a minimal experiment for comparing:

1. CPU-only execution with no checkpointing
2. CUDA GPU execution with no checkpointing
3. CPU-only CRIU checkpoint/restore
4. GPU CRIUgpu checkpoint/restore

The CPU workload is compiled with `g++` and never initializes CUDA. The GPU workload is compiled with `nvcc`, uses `cudaMalloc`, and avoids CUDA Unified Virtual Memory, CUDA IPC, and NCCL.

---

## System Requirements

| Component | Minimum | Tested with |
|---|---|---|
| OS | RHEL / CentOS 10 or equivalent | RHEL 10.2 |
| Kernel | 6.1+ | 6.12.0 |
| C++ compiler | g++ with C++17 support | g++ 14 |
| CUDA compiler | nvcc (CUDA 12.3+) | CUDA 13.2 |
| NVIDIA driver | 550+ (required for cuda-checkpoint) | 595.71.05 |
| CRIU | 4.0+ | 4.2 (built from source) |
| cuda-checkpoint | any | NVIDIA/cuda-checkpoint (GitHub) |
| GPUs | 1+ NVIDIA GPU | 2x NVIDIA L40 |

CRIU must be built from source to ensure all plugins (including the CUDA plugin) are compiled in. The DNF/APT package may omit optional features depending on distro packaging.

---

## Directory Layout

This project uses a split layout to keep large generated files off of Git:

```
~/criu-experiment/          # this repo
  src/                      # C++ and CUDA source
  scripts/                  # experiment and preflight scripts
  build/                    # compiled binaries (gitignored, created by make)
  results/                  # CSVs and small logs

~/criu-tools-src/           # installation downloads (not in repo)
  criu-v4.2.tar.gz
  cuda-checkpoint

~/criu-tools-build/         # CRIU source tree and build output (not in repo)
  criu-4.2/

~/criu-checkpoints/         # large CRIU checkpoint image dumps (not in repo)
  cpu_<run_id>/
  gpu_<run_id>/

~/criu-run-logs/            # build and install logs (not in repo)
  criu-4.2-build.log
  criu-4.2-install.log
```

Checkpoint images are written to `~/criu-checkpoints/` and can reach tens of GB for multi-GPU runs. To clean them: `rm -rf ~/criu-checkpoints/`.

---

## Build

```bash
make
```

## Preflight

Verifies that all required tools are installed and the kernel supports CRIU:

```bash
bash scripts/preflight.sh
```

## Baselines

```bash
bash scripts/run_cpu_baseline.sh
GPUS=0,1 bash scripts/run_gpu_baseline.sh
```

## CPU-only CRIU test

```bash
bash scripts/run_cpu_criu.sh
```

## GPU CRIUgpu tests

```bash
GPUS=0 bash scripts/run_gpu_criugpu.sh
GPUS=0,1 bash scripts/run_gpu_criugpu.sh
```

## Run all

```bash
bash scripts/run_all.sh
```

## Useful knobs

```bash
N=1024           # matrix dimension (N x N)
GPU_EXTRA_MB=2048  # extra GPU memory per GPU allocated to inflate checkpoint size
CPU_EXTRA_MB=512   # extra CPU memory allocated to inflate checkpoint size
PRE_SECONDS=10     # seconds the workload runs before checkpoint
POST_SECONDS=10    # seconds the workload runs after restore
GPUS=0,1           # comma-separated list of GPU IDs to use
```

Keep per-GPU memory below the available device memory. For two L40 GPUs (48 GiB each), `GPU_EXTRA_MB=2048` is a safe default.

---

## Quick start

```bash
git clone <repo-url>
cd criu-experiment
chmod +x scripts/*.sh
bash scripts/preflight.sh
make
bash scripts/run_cpu_baseline.sh
GPUS=0,1 bash scripts/run_gpu_baseline.sh
bash scripts/run_cpu_criu.sh
GPUS=0 bash scripts/run_gpu_criugpu.sh
GPUS=0,1 bash scripts/run_gpu_criugpu.sh
python3 scripts/summarize.py
```

---

## Output files

After each run, inspect:

- `results/*_metrics.csv` — checkpoint timing and size per run
- `results/*_app.csv` — per-iteration application metrics
- `~/criu-checkpoints/<run_id>/dump.log` — verbose CRIU dump log
- `~/criu-checkpoints/<run_id>/restore.log` — verbose CRIU restore log