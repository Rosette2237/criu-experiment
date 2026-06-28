#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p results
make

N="${N:-512}"
ITERS="${ITERS:-20}"
CPU_EXTRA_MB="${CPU_EXTRA_MB:-512}"

CSV="results/cpu_baseline_app.csv"
rm -f "$CSV"

./build/cpu_matmul \
  --n "$N" \
  --iters "$ITERS" \
  --extra-mb "$CPU_EXTRA_MB" \
  --csv "$CSV"

echo "CPU baseline complete: $CSV"