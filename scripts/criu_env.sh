#!/usr/bin/env bash
# Shared CRIU environment for all experiment scripts. Source this, then pass
# --libdir "$CRIU_LIBDIR" to every `criu dump` / `criu restore`.
#
# CRIU on this host is built from source under /usr/local, so its plugins
# (including cuda_plugin.so, needed for GPU checkpoint/restore) install to
# /usr/local/lib/criu -- NOT criu's compiled-in default of /usr/lib/criu, which
# does not exist here. Without --libdir, cuda_plugin.so never loads and a GPU
# dump aborts on the first /dev/nvidia* mapping:
#   Error (criu/proc_parse.c:118): handle_device_vma plugin failed: ...
#
# Auto-detect the dir that actually contains the plugins; allow override via the
# CRIU_LIBDIR env var. Already-exported values (e.g. from run_all.sh) are kept.
if [[ -z "${CRIU_LIBDIR:-}" ]]; then
  for d in /usr/local/lib/criu /usr/lib/criu /usr/lib64/criu; do
    [[ -e "$d/cuda_plugin.so" ]] && CRIU_LIBDIR="$d" && break
  done
fi
export CRIU_LIBDIR="${CRIU_LIBDIR:-/usr/local/lib/criu}"