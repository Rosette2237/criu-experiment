#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source scripts/criu_env.sh
echo "CRIU_LIBDIR=$CRIU_LIBDIR"

bash scripts/preflight.sh

echo
echo "== CPU baseline =="
bash scripts/run_cpu_baseline.sh

echo
echo "== GPU baseline: all 3 GPUs =="
GPUS=0,1,2 bash scripts/run_gpu_baseline.sh

echo
echo "== CPU-only CRIU checkpoint/restore =="
bash scripts/run_cpu_criu.sh

echo
echo "== GPU CRIUgpu checkpoint/restore: 1 GPU =="
GPUS=0 bash scripts/run_gpu_criugpu.sh

echo
echo "== GPU CRIUgpu checkpoint/restore: 2 GPUs =="
GPUS=0,1 bash scripts/run_gpu_criugpu.sh

echo
echo "== GPU CRIUgpu checkpoint/restore: 3 GPUs =="
GPUS=0,1,2 bash scripts/run_gpu_criugpu.sh

python3 scripts/summarize.py