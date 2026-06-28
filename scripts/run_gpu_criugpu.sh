#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUDO="${SUDO:-sudo}"
N="${N:-1024}"
GPU_EXTRA_MB="${GPU_EXTRA_MB:-2048}"
GPUS="${GPUS:-0,1,2}"
PRE_SECONDS="${PRE_SECONDS:-10}"
POST_SECONDS="${POST_SECONDS:-10}"

RUN_ID="gpu_$(echo "$GPUS" | tr ',' '-')_$(date +%Y%m%d_%H%M%S)"
CKPT_DIR="$HOME/criu-checkpoints/$RUN_ID"
APP_CSV="results/${RUN_ID}_app.csv"
METRICS_CSV="results/gpu_criugpu_metrics.csv"

mkdir -p results "$CKPT_DIR"
make

./build/gpu_matmul \
  --n "$N" \
  --iters 0 \
  --extra-mb "$GPU_EXTRA_MB" \
  --gpus "$GPUS" \
  --csv "$APP_CSV" \
  </dev/null \
  > "results/${RUN_ID}.stdout" \
  2> "results/${RUN_ID}.stderr" &

PID=$!
echo "Started GPU workload PID=$PID GPUS=$GPUS"

sleep "$PRE_SECONDS"

echo "nvidia-smi before dump:"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv || true

DUMP_START_NS="$(date +%s%N)"
$SUDO criu dump \
  --shell-job \
  --tree "$PID" \
  --images-dir "$CKPT_DIR" \
  -v4 \
  -o dump.log
DUMP_END_NS="$(date +%s%N)"

RESTORE_START_NS="$(date +%s%N)"
$SUDO criu restore \
  --restore-detached \
  --images-dir "$CKPT_DIR" \
  -v4 \
  -o restore.log
RESTORE_END_NS="$(date +%s%N)"

sleep "$POST_SECONDS"

echo "nvidia-smi after restore:"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv || true

pkill -TERM -f "$ROOT/build/gpu_matmul" || true
sleep 1

CKPT_BYTES="$(du -sb "$CKPT_DIR" | awk '{print $1}')"
DUMP_S="$(awk "BEGIN {print ($DUMP_END_NS - $DUMP_START_NS) / 1000000000}")"
RESTORE_S="$(awk "BEGIN {print ($RESTORE_END_NS - $RESTORE_START_NS) / 1000000000}")"

if [[ ! -f "$METRICS_CSV" ]]; then
  echo "run_id,mode,gpus,n,extra_mb,dump_wall_s,restore_wall_s,checkpoint_bytes,app_csv,ckpt_dir" > "$METRICS_CSV"
fi

echo "$RUN_ID,gpu_criugpu,$GPUS,$N,$GPU_EXTRA_MB,$DUMP_S,$RESTORE_S,$CKPT_BYTES,$APP_CSV,$CKPT_DIR" >> "$METRICS_CSV"

echo "GPU CRIUgpu test complete"
echo "metrics: $METRICS_CSV"
echo "app log: $APP_CSV"
echo "checkpoint dir: $CKPT_DIR"