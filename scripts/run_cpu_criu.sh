#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$(dirname "${BASH_SOURCE[0]}")/criu_env.sh"

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

# Launch the workload fully detached from the terminal so CRIU sees no TTY:
#   - setsid:        new session with no controlling terminal
#   - fd-closing:    drop any fds inherited from the launching terminal
#                    (e.g. a leaked /dev/ptmx master that breaks restore)
#   - stdin/out/err: /dev/null and plain files (CRIU-safe)
# setsid forks, so $! is the wrapper, not the workload. The exec'd process keeps
# the inner shell's PID, so it records its own PID to PIDFILE for CRIU --tree.
PIDFILE="$CKPT_DIR/workload.pid"
setsid bash -c '
  echo $$ > "$1"
  for fd in /proc/self/fd/*; do
    n=${fd##*/}
    [ "$n" -gt 2 ] && eval "exec $n>&-"
  done 2>/dev/null
  exec ./build/cpu_matmul --n "$2" --iters 0 --extra-mb "$3" --csv "$4"
' bash "$PIDFILE" "$N" "$CPU_EXTRA_MB" "$APP_CSV" \
  </dev/null \
  > "results/${RUN_ID}.stdout" \
  2> "results/${RUN_ID}.stderr" &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
PID="$(cat "$PIDFILE")"
echo "Started CPU workload PID=$PID"
sleep "$PRE_SECONDS"

DUMP_START_NS="$(date +%s%N)"
$SUDO criu dump \
  --tree "$PID" \
  --images-dir "$CKPT_DIR" \
  --libdir "$CRIU_LIBDIR" \
  -v4 \
  -o dump.log
DUMP_END_NS="$(date +%s%N)"

RESTORE_START_NS="$(date +%s%N)"
$SUDO criu restore \
  --restore-detached \
  --images-dir "$CKPT_DIR" \
  --libdir "$CRIU_LIBDIR" \
  -v4 \
  -o restore.log
RESTORE_END_NS="$(date +%s%N)"

sleep "$POST_SECONDS"

pkill -TERM -f "./build/cpu_matmul" || true
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