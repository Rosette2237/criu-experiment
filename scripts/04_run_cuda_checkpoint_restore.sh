#!/usr/bin/env bash
#
# Run the CUDA matrix multiplication benchmark, checkpoint it mid-execution,
# restore it with CRIU, resume CUDA state, and verify final correctness.
#
# Run from repo root:
#
#   ./scripts/04_run_cuda_checkpoint_restore.sh
#
# Optional overrides:
#
#   MATRIX_SIZE=512 ITERATIONS=30 CHECKPOINT_AFTER_ITERATION=10 ./scripts/04_run_cuda_checkpoint_restore.sh
#
# Recommended first CUDA checkpoint/restore test:
#
#   MATRIX_SIZE=512 ITERATIONS=30 CHECKPOINT_AFTER_ITERATION=10 ./scripts/04_run_cuda_checkpoint_restore.sh

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

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/cuda_checkpoint_helpers.sh"

criu_exp_load_env

# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/single_gpu.env"

criu_exp_reset_run_paths "$(criu_exp_new_run_id cuda_criu_single_gpu)"

criu_exp_info "Starting CUDA CRIU checkpoint/restore experiment"
criu_exp_print_run_summary

criu_exp_make_dirs
criu_exp_clear_cuda_markers

if [[ ! -x "${CUDA_BENCH_BIN}" ]]; then
    criu_exp_die "CUDA benchmark binary not found or not executable: ${CUDA_BENCH_BIN}. Run ./scripts/01_build.sh first."
fi

criu_exp_require_criu

if [[ "${CUDA_CHECKPOINT_MODE}" != "criu-plugin" ]]; then
    if ! command -v "${CUDA_CHECKPOINT_BIN}" >/dev/null 2>&1; then
        if [[ "${CUDA_CHECKPOINT_STRICT:-1}" == "1" ]]; then
            criu_exp_die "cuda-checkpoint not found, but CUDA_CHECKPOINT_MODE=${CUDA_CHECKPOINT_MODE}. Set CUDA_CHECKPOINT_MODE=criu-plugin to rely on CRIU plugin only."
        else
            criu_exp_warn "cuda-checkpoint not found; continuing because CUDA_CHECKPOINT_STRICT=0"
        fi
    fi
fi

if (( CHECKPOINT_AFTER_ITERATION <= 0 )); then
    criu_exp_die "CHECKPOINT_AFTER_ITERATION must be > 0"
fi

if (( CHECKPOINT_AFTER_ITERATION >= ITERATIONS )); then
    criu_exp_die "CHECKPOINT_AFTER_ITERATION must be less than ITERATIONS"
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
criu_exp_info "CHECKPOINT_AFTER_ITERATION=${CHECKPOINT_AFTER_ITERATION}"
criu_exp_info "CUDA_CHECKPOINT_MODE=${CUDA_CHECKPOINT_MODE}"
criu_exp_info "CRIU_LOG_LEVEL=${CRIU_LOG_LEVEL}"

CUDA_PID_FILE="${RUN_DIR}/cuda_program.pid"

cleanup_on_failure() {
    local exit_code=$?

    if [[ "${exit_code}" -ne 0 ]]; then
        criu_exp_warn "CUDA checkpoint/restore experiment failed; attempting cleanup"

        if [[ -n "${CUDA_PID:-}" ]] && kill -0 "${CUDA_PID}" >/dev/null 2>&1; then
            criu_exp_warn "Killing CUDA benchmark process pid=${CUDA_PID}"
            kill "${CUDA_PID}" >/dev/null 2>&1 || true
            sleep 1
            kill -9 "${CUDA_PID}" >/dev/null 2>&1 || true
        fi
    fi

    exit "${exit_code}"
}

trap cleanup_on_failure EXIT

criu_exp_info "Launching CUDA benchmark process"

CUDA_PROGRAM_START_NS="$(criu_exp_now_ns)"

"${CUDA_BENCH_BIN}" \
    --matrix-size "${MATRIX_SIZE}" \
    --iterations "${ITERATIONS}" \
    --block-size "${BLOCK_SIZE}" \
    --device "${CUDA_DEVICE}" \
    --sleep-ms-between-iterations "${SLEEP_MS_BETWEEN_ITERATIONS}" \
    --verify "${VERIFY}" \
    --verify-samples "${VERIFY_SAMPLES}" \
    --output-json "${CHECKPOINT_RESTORE_JSON}" \
    --ready-file "${CUDA_READY_FILE}" \
    --progress-file "${CUDA_PROGRESS_FILE}" \
    --done-file "${CUDA_DONE_FILE}" \
    --clear-markers 1 \
    > "${CUDA_STDOUT_LOG}" \
    2> "${CUDA_STDERR_LOG}" &

CUDA_PID=$!
echo "${CUDA_PID}" > "${CUDA_PID_FILE}"

{
    printf "CUDA_PROGRAM_START_NS=%s\n" "${CUDA_PROGRAM_START_NS}"
    printf "CUDA_PROGRAM_PID=%s\n" "${CUDA_PID}"
} > "${TIMING_FILE}"

criu_exp_info "CUDA benchmark process started: pid=${CUDA_PID}"
criu_exp_info "PID file: ${CUDA_PID_FILE}"

criu_exp_wait_for_file \
    "${CUDA_READY_FILE}" \
    "${WAIT_TIMEOUT_SECONDS}" \
    "CUDA ready marker"

criu_exp_info "Waiting until CUDA progress reaches checkpoint point"

criu_exp_wait_for_progress_at_least \
    "${CUDA_PROGRESS_FILE}" \
    "${CHECKPOINT_AFTER_ITERATION}" \
    "${WAIT_TIMEOUT_SECONDS}"

if [[ "${COLLECT_NVIDIA_SMI:-1}" == "1" ]]; then
    criu_exp_capture_nvidia_smi "${NVIDIA_SMI_DURING}"
fi

criu_exp_info "Preparing CUDA process for checkpoint"

criu_exp_cuda_prepare_for_dump_if_needed \
    "${CUDA_PID}" \
    "${TIMING_FILE}"

criu_exp_info "Checkpointing CUDA process with CRIU"

criu_exp_dump_process \
    "${CUDA_PID}" \
    "${CHECKPOINT_DIR}" \
    "${CRIU_DUMP_LOG}" \
    "${TIMING_FILE}"

if [[ "${COLLECT_CHECKPOINT_SIZE:-1}" == "1" ]]; then
    criu_exp_write_checkpoint_size \
        "${CHECKPOINT_DIR}" \
        "${CHECKPOINT_SIZE_FILE}"
fi

criu_exp_info "Confirming original CUDA process was checkpointed"

if kill -0 "${CUDA_PID}" >/dev/null 2>&1; then
    if [[ "${CRIU_LEAVE_RUNNING:-0}" == "1" ]]; then
        criu_exp_warn "Original CUDA process is still alive because CRIU_LEAVE_RUNNING=1"
    else
        criu_exp_warn "Original CUDA process still appears alive after dump: pid=${CUDA_PID}"
    fi
else
    criu_exp_info "Original CUDA process is no longer running after dump, as expected"
fi

criu_exp_info "Restoring CUDA process with CRIU"

criu_exp_restore_process \
    "${CHECKPOINT_DIR}" \
    "${CRIU_RESTORE_LOG}" \
    "${TIMING_FILE}"

criu_exp_info "Waiting for restored process PID to appear"

RESTORE_PID_WAIT_START="$(date +%s)"

while true; do
    if kill -0 "${CUDA_PID}" >/dev/null 2>&1; then
        criu_exp_info "Restored CUDA process is visible: pid=${CUDA_PID}"
        break
    fi

    NOW="$(date +%s)"
    ELAPSED=$((NOW - RESTORE_PID_WAIT_START))

    if (( ELAPSED >= RESTORE_WAIT_TIMEOUT_SECONDS )); then
        criu_exp_error "Timed out waiting for restored CUDA process pid=${CUDA_PID}"
        criu_exp_error "CRIU restore log: ${CRIU_RESTORE_LOG}"
        exit 1
    fi

    sleep 1
done

criu_exp_info "Resuming CUDA state after restore"

criu_exp_cuda_resume_after_restore_if_needed \
    "${CUDA_PID}" \
    "${TIMING_FILE}"

criu_exp_info "Waiting for restored CUDA benchmark to complete"

criu_exp_wait_for_file \
    "${CUDA_DONE_FILE}" \
    "${RESTORE_WAIT_TIMEOUT_SECONDS}" \
    "CUDA done marker"

CUDA_PROGRAM_END_NS="$(criu_exp_now_ns)"
CUDA_PROGRAM_TOTAL_WALL_MS="$(criu_exp_elapsed_ms "${CUDA_PROGRAM_START_NS}" "${CUDA_PROGRAM_END_NS}")"

{
    printf "CUDA_PROGRAM_END_NS=%s\n" "${CUDA_PROGRAM_END_NS}"
    printf "CUDA_PROGRAM_TOTAL_WALL_MS=%s\n" "${CUDA_PROGRAM_TOTAL_WALL_MS}"
} >> "${TIMING_FILE}"

if [[ "${COLLECT_NVIDIA_SMI:-1}" == "1" ]]; then
    criu_exp_capture_nvidia_smi "${NVIDIA_SMI_AFTER}"
fi

if [[ ! -f "${CHECKPOINT_RESTORE_JSON}" ]]; then
    criu_exp_error "Checkpoint/restore JSON was not produced: ${CHECKPOINT_RESTORE_JSON}"
    criu_exp_error "stdout log: ${CUDA_STDOUT_LOG}"
    criu_exp_error "stderr log: ${CUDA_STDERR_LOG}"
    criu_exp_error "CRIU dump log: ${CRIU_DUMP_LOG}"
    criu_exp_error "CRIU restore log: ${CRIU_RESTORE_LOG}"
    criu_exp_print_result_locations
    exit 1
fi

if ! criu_exp_json_passed "${CHECKPOINT_RESTORE_JSON}"; then
    criu_exp_error "Checkpoint/restore JSON reports passed=false"
    criu_exp_error "stdout log: ${CUDA_STDOUT_LOG}"
    criu_exp_error "stderr log: ${CUDA_STDERR_LOG}"
    criu_exp_error "CRIU dump log: ${CRIU_DUMP_LOG}"
    criu_exp_error "CRIU restore log: ${CRIU_RESTORE_LOG}"
    criu_exp_print_result_locations
    exit 1
fi

FINAL_COMPLETED_ITERATIONS="$(
    criu_exp_extract_json_number \
        "${CHECKPOINT_RESTORE_JSON}" \
        "completed_iterations"
)"

if [[ "${FINAL_COMPLETED_ITERATIONS}" != "${ITERATIONS}" ]]; then
    criu_exp_error "Restored CUDA process did not complete all iterations: completed=${FINAL_COMPLETED_ITERATIONS}, expected=${ITERATIONS}"
    criu_exp_print_result_locations
    exit 1
fi

criu_exp_info "CUDA CRIU checkpoint/restore experiment completed successfully"
criu_exp_info "Completed iterations: ${FINAL_COMPLETED_ITERATIONS}/${ITERATIONS}"
criu_exp_info "Result JSON: ${CHECKPOINT_RESTORE_JSON}"
criu_exp_info "Timing file: ${TIMING_FILE}"
criu_exp_info "Checkpoint size file: ${CHECKPOINT_SIZE_FILE}"
criu_exp_info "CRIU dump log: ${CRIU_DUMP_LOG}"
criu_exp_info "CRIU restore log: ${CRIU_RESTORE_LOG}"

criu_exp_print_result_locations

trap - EXIT

criu_exp_info "Next step: ./scripts/05_run_matrix_sweep.sh"