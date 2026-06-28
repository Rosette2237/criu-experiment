#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p results
make

N="${N:-1024}"
ITERS="${ITERS:-30}"
GPU_EXTRA_MB="${GPU_EXTRA_MB:-2048}"
GPUS="${GPUS:-0,1,2}"

CSV="results/gpu_baseline_app.csv"
rm -f "$CSV"

./build/gpu_matmul \
  --n "$N" \
  --iters "$ITERS" \
  --extra-mb "$GPU_EXTRA_MB" \
  --gpus "$GPUS" \
  --csv "$CSV"

echo "GPU baseline complete: $CSV"