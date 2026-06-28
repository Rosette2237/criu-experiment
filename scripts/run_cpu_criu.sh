#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUDO="${SUDO:-sudo}"
N="${N:-512}"
CPU_EXTRA_MB="${CPU_EXTRA_MB:-512}"
PRE_SECONDS="${PRE_SECONDS:-8}"
POST_SECONDS="${POST_SECONDS:-8}"

RUN_ID="cpu_$(date +%Y%m%d_%H%M%S)"
CKPT_DIR="$HOME/criu-checkpoints/$RUN_ID"
APP_CSV="results/${RUN_ID}_app.csv"
METRICS_CSV="results/cpu_criu_metrics.csv"

mkdir -p results "$CKPT_DIR"
make

./build/cpu_matmul \
  --n "$N" \
  --iters 0 \
  --extra-mb "$CPU_EXTRA_MB" \
  --csv "$APP_CSV" \
  > "results/${RUN_ID}.stdout" \
  2> "results/${RUN_ID}.stderr" &

PID=$!
echo "Started CPU workload PID=$PID"
sleep "$PRE_SECONDS"

DUMP_START_NS="$(date +%s%N)"
$SUDO criu dump \
  --shell-job \
  --tree "$PID" \
  --images-dir "$CKPT_DIR" \
  --manage-cgroups ignore \
  -v4 \
  -o dump.log
DUMP_END_NS="$(date +%s%N)"

RESTORE_START_NS="$(date +%s%N)"
$SUDO criu restore \
  --shell-job \
  --restore-detached \
  --images-dir "$CKPT_DIR" \
  --manage-cgroups ignore \
  -v4 \
  -o restore.log
RESTORE_END_NS="$(date +%s%N)"

sleep "$POST_SECONDS"

pkill -TERM -f "$ROOT/build/cpu_matmul" || true
sleep 1

CKPT_BYTES="$(du -sb "$CKPT_DIR" | awk '{print $1}')"
DUMP_S="$(awk "BEGIN {print ($DUMP_END_NS - $DUMP_START_NS) / 1000000000}")"
RESTORE_S="$(awk "BEGIN {print ($RESTORE_END_NS - $RESTORE_START_NS) / 1000000000}")"

if [[ ! -f "$METRICS_CSV" ]]; then
  echo "run_id,mode,gpus,n,extra_mb,dump_wall_s,restore_wall_s,checkpoint_bytes,app_csv,ckpt_dir" > "$METRICS_CSV"
fi

echo "$RUN_ID,cpu_criu,none,$N,$CPU_EXTRA_MB,$DUMP_S,$RESTORE_S,$CKPT_BYTES,$APP_CSV,$CKPT_DIR" >> "$METRICS_CSV"

echo "CPU CRIU test complete"
echo "metrics: $METRICS_CSV"
echo "app log: $APP_CSV"
echo "checkpoint dir: $CKPT_DIR"