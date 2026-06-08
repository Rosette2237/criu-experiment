#!/usr/bin/env bash
#
# Run the CUDA matrix multiplication benchmark without CRIU.
#
# Run from repo root:
#
#   ./scripts/02_run_cuda_baseline.sh
#
# Optional overrides:
#
#   MATRIX_SIZE=2048 ITERATIONS=80 ./scripts/02_run_cuda_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/env.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/metrics_helpers.sh"

criu_exp_load_env

# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/single_gpu.env"

criu_exp_reset_run_paths "$(criu_exp_new_run_id cuda_baseline)"

criu_exp_info "Starting CUDA baseline experiment"
criu_exp_print_run_summary

criu_exp_make_dirs
criu_exp_clear_cuda_markers

if [[ ! -x "${CUDA_BENCH_BIN}" ]]; then
    criu_exp_die "CUDA benchmark binary not found or not executable: ${CUDA_BENCH_BIN}. Run ./scripts/01_build.sh first."
fi

criu_exp_write_environment_json "${ENV_JSON}"

if [[ "${COLLECT_NVIDIA_SMI:-1}" == "1" ]]; then
    criu_exp_capture_nvidia_smi "${NVIDIA_SMI_BEFORE}"
fi

criu_exp_info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
criu_exp_info "CUDA_DEVICE=${CUDA_DEVICE}"
criu_exp_info "MATRIX_SIZE=${MATRIX_SIZE}"
criu_exp_info "ITERATIONS=${ITERATIONS}"
criu_exp_info "BLOCK_SIZE=${BLOCK_SIZE}"
criu_exp_info "SLEEP_MS_BETWEEN_ITERATIONS=${SLEEP_MS_BETWEEN_ITERATIONS}"
criu_exp_info "VERIFY=${VERIFY}"
criu_exp_info "VERIFY_SAMPLES=${VERIFY_SAMPLES}"

BASELINE_START_NS="$(criu_exp_now_ns)"

set +e
"${CUDA_BENCH_BIN}" \
    --matrix-size "${MATRIX_SIZE}" \
    --iterations "${ITERATIONS}" \
    --block-size "${BLOCK_SIZE}" \
    --device "${CUDA_DEVICE}" \
    --sleep-ms-between-iterations "${SLEEP_MS_BETWEEN_ITERATIONS}" \
    --verify "${VERIFY}" \
    --verify-samples "${VERIFY_SAMPLES}" \
    --output-json "${CUDA_BASELINE_JSON}" \
    --ready-file "${CUDA_READY_FILE}" \
    --progress-file "${CUDA_PROGRESS_FILE}" \
    --done-file "${CUDA_DONE_FILE}" \
    --clear-markers 1 \
    > "${CUDA_STDOUT_LOG}" \
    2> "${CUDA_STDERR_LOG}"
CUDA_EXIT_CODE=$?
set -e

BASELINE_END_NS="$(criu_exp_now_ns)"
BASELINE_WALL_MS="$(criu_exp_elapsed_ms "${BASELINE_START_NS}" "${BASELINE_END_NS}")"

{
    printf "CUDA_BASELINE_START_NS=%s\n" "${BASELINE_START_NS}"
    printf "CUDA_BASELINE_END_NS=%s\n" "${BASELINE_END_NS}"
    printf "CUDA_BASELINE_WALL_MS=%s\n" "${BASELINE_WALL_MS}"
    printf "CUDA_BASELINE_EXIT_CODE=%s\n" "${CUDA_EXIT_CODE}"
} > "${TIMING_FILE}"

if [[ "${COLLECT_NVIDIA_SMI:-1}" == "1" ]]; then
    criu_exp_capture_nvidia_smi "${NVIDIA_SMI_AFTER}"
fi

criu_exp_info "CUDA baseline process exit code: ${CUDA_EXIT_CODE}"
criu_exp_info "CUDA baseline wall time ms: ${BASELINE_WALL_MS}"

if [[ "${CUDA_EXIT_CODE}" -ne 0 ]]; then
    criu_exp_error "CUDA baseline failed"
    criu_exp_error "stdout log: ${CUDA_STDOUT_LOG}"
    criu_exp_error "stderr log: ${CUDA_STDERR_LOG}"
    criu_exp_print_result_locations
    exit "${CUDA_EXIT_CODE}"
fi

if [[ ! -f "${CUDA_BASELINE_JSON}" ]]; then
    criu_exp_error "CUDA baseline JSON was not produced: ${CUDA_BASELINE_JSON}"
    criu_exp_print_result_locations
    exit 1
fi

if ! criu_exp_json_passed "${CUDA_BASELINE_JSON}"; then
    criu_exp_error "CUDA baseline JSON reports passed=false"
    criu_exp_print_result_locations
    exit 1
fi

criu_exp_info "CUDA baseline experiment completed successfully"
criu_exp_info "Result JSON: ${CUDA_BASELINE_JSON}"
criu_exp_info "Timing file: ${TIMING_FILE}"
criu_exp_print_result_locations

criu_exp_info "Next step: ./scripts/03_run_cpu_criu_baseline.sh"