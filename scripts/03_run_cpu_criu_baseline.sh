#!/usr/bin/env bash
#
# Run a CPU-only CRIU checkpoint/restore baseline.
#
# This test verifies that CRIU works on the VM before testing CUDA-aware
# checkpoint/restore.
#
# Run from repo root:
#
#   ./scripts/03_run_cpu_criu_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/env.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/metrics_helpers.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/criu_helpers.sh"

criu_exp_load_env
criu_exp_reset_run_paths "$(criu_exp_new_run_id cpu_criu_baseline)"

criu_exp_info "Starting CPU-only CRIU baseline experiment"
criu_exp_print_run_summary

criu_exp_make_dirs
criu_exp_clear_cpu_markers

if [[ ! -x "${CPU_BASELINE_BIN}" ]]; then
    criu_exp_die "CPU baseline binary not found or not executable: ${CPU_BASELINE_BIN}. Run ./scripts/01_build.sh first."
fi

criu_exp_require_criu

criu_exp_write_environment_json "${ENV_JSON}"

if [[ "${COLLECT_NVIDIA_SMI:-1}" == "1" ]]; then
    criu_exp_capture_nvidia_smi "${NVIDIA_SMI_BEFORE}"
fi

criu_exp_info "Running CRIU check before CPU baseline"
criu_exp_check_criu_basic

criu_exp_info "CPU_ITERATIONS=${CPU_ITERATIONS}"
criu_exp_info "CPU_SLEEP_MS=${CPU_SLEEP_MS}"
criu_exp_info "CHECKPOINT_AFTER_ITERATION=${CHECKPOINT_AFTER_ITERATION}"
criu_exp_info "WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS}"
criu_exp_info "RESTORE_WAIT_TIMEOUT_SECONDS=${RESTORE_WAIT_TIMEOUT_SECONDS}"

if (( CHECKPOINT_AFTER_ITERATION <= 0 )); then
    criu_exp_die "CHECKPOINT_AFTER_ITERATION must be > 0"
fi

if (( CHECKPOINT_AFTER_ITERATION >= CPU_ITERATIONS )); then
    criu_exp_die "CHECKPOINT_AFTER_ITERATION must be less than CPU_ITERATIONS"
fi

CPU_PID_FILE="${RUN_DIR}/cpu_program.pid"

criu_exp_info "Launching CPU baseline process"

"${CPU_BASELINE_BIN}" \
    --iterations "${CPU_ITERATIONS}" \
    --sleep-ms "${CPU_SLEEP_MS}" \
    --output-json "${CPU_BASELINE_JSON}" \
    --ready-file "${CPU_READY_FILE}" \
    --progress-file "${CPU_PROGRESS_FILE}" \
    --done-file "${CPU_DONE_FILE}" \
    --clear-markers 1 \
    > "${CPU_STDOUT_LOG}" \
    2> "${CPU_STDERR_LOG}" &

CPU_PID=$!
echo "${CPU_PID}" > "${CPU_PID_FILE}"

criu_exp_info "CPU baseline process started: pid=${CPU_PID}"
criu_exp_info "PID file: ${CPU_PID_FILE}"

cleanup_on_failure() {
    local exit_code=$?

    if [[ "${exit_code}" -ne 0 ]]; then
        criu_exp_warn "CPU CRIU baseline failed; attempting cleanup"

        if [[ -n "${CPU_PID:-}" ]] && kill -0 "${CPU_PID}" >/dev/null 2>&1; then
            criu_exp_warn "Killing original CPU process pid=${CPU_PID}"
            kill "${CPU_PID}" >/dev/null 2>&1 || true
        fi
    fi

    exit "${exit_code}"
}

trap cleanup_on_failure EXIT

criu_exp_wait_for_file "${CPU_READY_FILE}" "${WAIT_TIMEOUT_SECONDS}" "CPU ready marker"

criu_exp_info "Waiting until CPU progress reaches checkpoint point"
criu_exp_wait_for_progress_at_least \
    "${CPU_PROGRESS_FILE}" \
    "${CHECKPOINT_AFTER_ITERATION}" \
    "${WAIT_TIMEOUT_SECONDS}"

if [[ "${COLLECT_NVIDIA_SMI:-1}" == "1" ]]; then
    criu_exp_capture_nvidia_smi "${NVIDIA_SMI_DURING}"
fi

criu_exp_info "Checkpointing CPU process with CRIU"

criu_exp_dump_process \
    "${CPU_PID}" \
    "${CHECKPOINT_DIR}" \
    "${CRIU_DUMP_LOG}" \
    "${TIMING_FILE}"

if [[ "${COLLECT_CHECKPOINT_SIZE:-1}" == "1" ]]; then
    criu_exp_write_checkpoint_size \
        "${CHECKPOINT_DIR}" \
        "${CHECKPOINT_SIZE_FILE}"
fi

criu_exp_info "Restoring CPU process with CRIU"

criu_exp_restore_process \
    "${CHECKPOINT_DIR}" \
    "${CRIU_RESTORE_LOG}" \
    "${TIMING_FILE}"

criu_exp_info "Waiting for restored CPU process to complete"

criu_exp_wait_for_file \
    "${CPU_DONE_FILE}" \
    "${RESTORE_WAIT_TIMEOUT_SECONDS}" \
    "CPU done marker"

if [[ "${COLLECT_NVIDIA_SMI:-1}" == "1" ]]; then
    criu_exp_capture_nvidia_smi "${NVIDIA_SMI_AFTER}"
fi

if [[ ! -f "${CPU_BASELINE_JSON}" ]]; then
    criu_exp_error "CPU baseline JSON was not produced: ${CPU_BASELINE_JSON}"
    criu_exp_print_result_locations
    exit 1
fi

if ! criu_exp_json_passed "${CPU_BASELINE_JSON}"; then
    criu_exp_error "CPU baseline JSON reports passed=false"
    criu_exp_error "stdout log: ${CPU_STDOUT_LOG}"
    criu_exp_error "stderr log: ${CPU_STDERR_LOG}"
    criu_exp_error "CRIU dump log: ${CRIU_DUMP_LOG}"
    criu_exp_error "CRIU restore log: ${CRIU_RESTORE_LOG}"
    criu_exp_print_result_locations
    exit 1
fi

criu_exp_info "CPU-only CRIU baseline completed successfully"
criu_exp_info "Result JSON: ${CPU_BASELINE_JSON}"
criu_exp_info "Timing file: ${TIMING_FILE}"
criu_exp_info "CRIU dump log: ${CRIU_DUMP_LOG}"
criu_exp_info "CRIU restore log: ${CRIU_RESTORE_LOG}"

criu_exp_print_result_locations

trap - EXIT

criu_exp_info "Next step: ./scripts/04_run_cuda_checkpoint_restore.sh"