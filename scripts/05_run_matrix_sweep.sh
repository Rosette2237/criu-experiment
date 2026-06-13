#!/usr/bin/env bash
#
# Run a small CUDA checkpoint/restore matrix-size sweep.
#
# This script repeatedly invokes:
#
#   ./scripts/04_run_cuda_checkpoint_restore.sh
#
# with different MATRIX_SIZE values.
#
# Run from repo root:
#
#   ./scripts/05_run_matrix_sweep.sh
#
# Optional overrides:
#
#   MATRIX_SIZES="512 1024 2048" ./scripts/05_run_matrix_sweep.sh
#
#   MATRIX_SIZES="512 1024 2048 4096" \
#   SWEEP_ITERATIONS=80 \
#   SWEEP_CHECKPOINT_AFTER_ITERATION=30 \
#   ./scripts/05_run_matrix_sweep.sh

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
source "${REPO_ROOT}/configs/matrix_sweep_small.env"

SWEEP_RUN_ID="$(criu_exp_new_run_id matrix_sweep)"
SWEEP_RUN_DIR="${RAW_RESULTS_DIR}/${SWEEP_RUN_ID}"
SWEEP_SUMMARY_CSV="${SWEEP_RUN_DIR}/sweep_summary.csv"
SWEEP_LOG="${SWEEP_RUN_DIR}/sweep.log"

mkdir -p "${SWEEP_RUN_DIR}"

criu_exp_info "Starting CUDA CRIU matrix-size sweep" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_RUN_ID=${SWEEP_RUN_ID}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_RUN_DIR=${SWEEP_RUN_DIR}" | tee -a "${SWEEP_LOG}"
criu_exp_info "MATRIX_SIZES=${MATRIX_SIZES}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_ITERATIONS=${SWEEP_ITERATIONS}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_BLOCK_SIZE=${SWEEP_BLOCK_SIZE}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_SLEEP_MS_BETWEEN_ITERATIONS=${SWEEP_SLEEP_MS_BETWEEN_ITERATIONS}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_CHECKPOINT_AFTER_ITERATION=${SWEEP_CHECKPOINT_AFTER_ITERATION}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_VERIFY=${SWEEP_VERIFY}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_VERIFY_SAMPLES=${SWEEP_VERIFY_SAMPLES}" | tee -a "${SWEEP_LOG}"
criu_exp_info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "${SWEEP_LOG}"
criu_exp_info "SWEEP_CUDA_DEVICE=${SWEEP_CUDA_DEVICE}" | tee -a "${SWEEP_LOG}"

if [[ ! -x "${REPO_ROOT}/scripts/04_run_cuda_checkpoint_restore.sh" ]]; then
    criu_exp_die "Required script is not executable: ${REPO_ROOT}/scripts/04_run_cuda_checkpoint_restore.sh"
fi

if [[ ! -x "${CUDA_BENCH_BIN}" ]]; then
    criu_exp_die "CUDA benchmark binary not found or not executable: ${CUDA_BENCH_BIN}. Run ./scripts/01_build.sh first."
fi

{
    printf "sweep_run_id,matrix_size,iterations,checkpoint_after_iteration,status,child_run_id,child_run_dir,checkpoint_restore_json,timing_file,checkpoint_size_file,dump_wall_ms,restore_wall_ms,cuda_suspend_wall_ms,cuda_resume_wall_ms,cuda_program_total_wall_ms,checkpoint_image_bytes,completed_iterations,passed\n"
} > "${SWEEP_SUMMARY_CSV}"

extract_timing_value() {
    local file="$1"
    local key="$2"

    if [[ ! -f "${file}" ]]; then
        printf "\n"
        return 0
    fi

    grep "^${key}=" "${file}" | tail -n 1 | cut -d= -f2- || true
}

extract_checkpoint_size_value() {
    local file="$1"
    local key="$2"

    if [[ ! -f "${file}" ]]; then
        printf "\n"
        return 0
    fi

    grep "^${key}=" "${file}" | tail -n 1 | cut -d= -f2- || true
}

latest_child_run_dir_before() {
    find "${RAW_RESULTS_DIR}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name '*cuda_criu_single_gpu' \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2- || true
}

LAST_SUCCESSFUL_SIZE=""
FAILED_COUNT=0
SUCCESS_COUNT=0
TOTAL_COUNT=0

for SIZE in ${MATRIX_SIZES}; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    criu_exp_info "Starting sweep case: MATRIX_SIZE=${SIZE}" | tee -a "${SWEEP_LOG}"

    CASE_START_NS="$(criu_exp_now_ns)"

    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    CUDA_DEVICE="${SWEEP_CUDA_DEVICE}" \
    MATRIX_SIZE="${SIZE}" \
    ITERATIONS="${SWEEP_ITERATIONS}" \
    BLOCK_SIZE="${SWEEP_BLOCK_SIZE}" \
    SLEEP_MS_BETWEEN_ITERATIONS="${SWEEP_SLEEP_MS_BETWEEN_ITERATIONS}" \
    CHECKPOINT_AFTER_ITERATION="${SWEEP_CHECKPOINT_AFTER_ITERATION}" \
    VERIFY="${SWEEP_VERIFY}" \
    VERIFY_SAMPLES="${SWEEP_VERIFY_SAMPLES}" \
    WAIT_TIMEOUT_SECONDS="${SWEEP_WAIT_TIMEOUT_SECONDS}" \
    RESTORE_WAIT_TIMEOUT_SECONDS="${SWEEP_RESTORE_WAIT_TIMEOUT_SECONDS}" \
        "${REPO_ROOT}/scripts/04_run_cuda_checkpoint_restore.sh" \
        > "${SWEEP_RUN_DIR}/matrix_${SIZE}_stdout.log" \
        2> "${SWEEP_RUN_DIR}/matrix_${SIZE}_stderr.log"
    CASE_EXIT_CODE=$?
    set -e

    CASE_END_NS="$(criu_exp_now_ns)"
    CASE_WALL_MS="$(criu_exp_elapsed_ms "${CASE_START_NS}" "${CASE_END_NS}")"

    CHILD_RUN_DIR="$(latest_child_run_dir_before)"
    CHILD_RUN_ID="$(basename "${CHILD_RUN_DIR:-unknown}")"

    CHILD_JSON="${CHILD_RUN_DIR}/checkpoint_restore.json"
    CHILD_TIMING="${CHILD_RUN_DIR}/checkpoint_timing.env"
    CHILD_SIZE_FILE="${CHILD_RUN_DIR}/checkpoint_size.txt"

    DUMP_WALL_MS="$(extract_timing_value "${CHILD_TIMING}" "DUMP_WALL_MS")"
    RESTORE_WALL_MS="$(extract_timing_value "${CHILD_TIMING}" "RESTORE_WALL_MS")"
    CUDA_SUSPEND_WALL_MS="$(extract_timing_value "${CHILD_TIMING}" "CUDA_SUSPEND_WALL_MS")"
    CUDA_RESUME_WALL_MS="$(extract_timing_value "${CHILD_TIMING}" "CUDA_RESUME_WALL_MS")"
    CUDA_PROGRAM_TOTAL_WALL_MS="$(extract_timing_value "${CHILD_TIMING}" "CUDA_PROGRAM_TOTAL_WALL_MS")"
    CHECKPOINT_IMAGE_BYTES="$(extract_checkpoint_size_value "${CHILD_SIZE_FILE}" "CHECKPOINT_IMAGE_BYTES")"

    COMPLETED_ITERATIONS=""
    PASSED="false"

    if [[ -f "${CHILD_JSON}" ]]; then
        COMPLETED_ITERATIONS="$(criu_exp_extract_json_number "${CHILD_JSON}" "completed_iterations")"

        if criu_exp_json_passed "${CHILD_JSON}"; then
            PASSED="true"
        else
            PASSED="false"
        fi
    fi

    if [[ "${CASE_EXIT_CODE}" -eq 0 && "${PASSED}" == "true" ]]; then
        STATUS="success"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        LAST_SUCCESSFUL_SIZE="${SIZE}"
        criu_exp_info "Sweep case succeeded: MATRIX_SIZE=${SIZE}, wall_ms=${CASE_WALL_MS}" | tee -a "${SWEEP_LOG}"
    else
        STATUS="failed"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        criu_exp_error "Sweep case failed: MATRIX_SIZE=${SIZE}, exit_code=${CASE_EXIT_CODE}, wall_ms=${CASE_WALL_MS}" | tee -a "${SWEEP_LOG}"
        criu_exp_error "stdout log: ${SWEEP_RUN_DIR}/matrix_${SIZE}_stdout.log" | tee -a "${SWEEP_LOG}"
        criu_exp_error "stderr log: ${SWEEP_RUN_DIR}/matrix_${SIZE}_stderr.log" | tee -a "${SWEEP_LOG}"
    fi

    {
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${SWEEP_RUN_ID}" \
            "${SIZE}" \
            "${SWEEP_ITERATIONS}" \
            "${SWEEP_CHECKPOINT_AFTER_ITERATION}" \
            "${STATUS}" \
            "${CHILD_RUN_ID}" \
            "${CHILD_RUN_DIR}" \
            "${CHILD_JSON}" \
            "${CHILD_TIMING}" \
            "${CHILD_SIZE_FILE}" \
            "${DUMP_WALL_MS}" \
            "${RESTORE_WALL_MS}" \
            "${CUDA_SUSPEND_WALL_MS}" \
            "${CUDA_RESUME_WALL_MS}" \
            "${CUDA_PROGRAM_TOTAL_WALL_MS}" \
            "${CHECKPOINT_IMAGE_BYTES}" \
            "${COMPLETED_ITERATIONS}" \
            "${PASSED}"
    } >> "${SWEEP_SUMMARY_CSV}"

    criu_exp_info "Sweep case result appended to: ${SWEEP_SUMMARY_CSV}" | tee -a "${SWEEP_LOG}"
done

{
    printf "\n"
    printf "Sweep completed\n"
    printf "  total cases:      %s\n" "${TOTAL_COUNT}"
    printf "  successful cases: %s\n" "${SUCCESS_COUNT}"
    printf "  failed cases:     %s\n" "${FAILED_COUNT}"
    printf "  last success:     %s\n" "${LAST_SUCCESSFUL_SIZE:-none}"
    printf "  summary csv:      %s\n" "${SWEEP_SUMMARY_CSV}"
    printf "  sweep log:        %s\n" "${SWEEP_LOG}"
    printf "\n"
} | tee -a "${SWEEP_LOG}"

if [[ "${FAILED_COUNT}" -ne 0 ]]; then
    criu_exp_error "Matrix sweep completed with failures"
    criu_exp_error "Inspect sweep directory: ${SWEEP_RUN_DIR}"
    exit 1
fi

criu_exp_info "Matrix sweep completed successfully"
criu_exp_info "Next step: ./scripts/06_collect_results.sh"