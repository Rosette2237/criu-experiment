#!/usr/bin/env bash
#
# Build CUDA and CPU benchmark binaries.
#
# Run from repo root:
#
#   ./scripts/01_build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/env.sh"

criu_exp_load_env
criu_exp_make_dirs

criu_exp_info "Starting build"
criu_exp_info "REPO_ROOT=${REPO_ROOT}"
criu_exp_info "BUILD_DIR=${BUILD_DIR}"
criu_exp_info "BIN_DIR=${BIN_DIR}"
criu_exp_info "CUDA_ARCH=${CUDA_ARCH}"

if ! command -v "${MAKE_BIN}" >/dev/null 2>&1; then
    criu_exp_die "make command not found: ${MAKE_BIN}"
fi

if ! command -v "${CC_BIN}" >/dev/null 2>&1; then
    criu_exp_die "C compiler not found: ${CC_BIN}"
fi

if ! command -v "${NVCC_BIN}" >/dev/null 2>&1; then
    criu_exp_die "CUDA compiler nvcc not found: ${NVCC_BIN}"
fi

criu_exp_info "Compiler versions"

{
    echo "===== ${CC_BIN} --version ====="
    "${CC_BIN}" --version || true

    echo
    echo "===== ${NVCC_BIN} --version ====="
    "${NVCC_BIN}" --version || true

    echo
    echo "===== ${MAKE_BIN} --version ====="
    "${MAKE_BIN}" --version || true
} >&2

criu_exp_info "Cleaning previous build artifacts"

"${MAKE_BIN}" -C "${REPO_ROOT}" clean

criu_exp_info "Building benchmark binaries"

"${MAKE_BIN}" -C "${REPO_ROOT}" all \
    NVCC="${NVCC_BIN}" \
    CC="${CC_BIN}" \
    CUDA_ARCH="${CUDA_ARCH}"

criu_exp_info "Checking build outputs"

if [[ ! -x "${CUDA_BENCH_BIN}" ]]; then
    criu_exp_die "CUDA benchmark binary was not produced or is not executable: ${CUDA_BENCH_BIN}"
fi

if [[ ! -x "${CPU_BASELINE_BIN}" ]]; then
    criu_exp_die "CPU baseline binary was not produced or is not executable: ${CPU_BASELINE_BIN}"
fi

criu_exp_info "Build completed successfully"
criu_exp_info "CUDA benchmark binary: ${CUDA_BENCH_BIN}"
criu_exp_info "CPU baseline binary: ${CPU_BASELINE_BIN}"

criu_exp_info "Next step: ./scripts/02_run_cuda_baseline.sh"