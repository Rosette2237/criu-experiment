#!/usr/bin/env bash
set -euo pipefail

SUDO="${SUDO:-sudo}"

echo "== tool checks =="
command -v nvidia-smi
command -v nvcc
command -v criu
command -v cuda-checkpoint

echo
echo "== versions =="
nvidia-smi
nvcc --version
criu --version
cuda-checkpoint --help | head -80

echo
echo "== CRIU kernel feature check =="
$SUDO criu check || true

echo
echo "Preflight finished."