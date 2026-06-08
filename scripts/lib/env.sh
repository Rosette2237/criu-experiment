#!/usr/bin/env bash
#
# Shared environment and utility functions for CRIU CUDA checkpoint experiments.
#
# This file is meant to be sourced by other scripts:
#
#   source scripts/lib/env.sh

set -o pipefail

criu_exp_find_repo_root() {
    local current_dir

    current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    printf "%s\n" "${current_dir}"
}

criu_exp_load_env() {
    if [[ -z "${REPO_ROOT:-}" ]]; then
        REPO_ROOT="$(criu_exp_find_repo_root)"
        export REPO_ROOT
    fi

    if [[ -f "${REPO_ROOT}/configs/default.env" ]]; then
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/configs/default.env"
    else
        echo "ERROR: Missing ${REPO_ROOT}/configs/default.env" >&2
        return 1
    fi

    if [[ -f "${REPO_ROOT}/configs/criu_options.env" ]]; then
        # shellcheck source=/dev/null
        source "${REPO_ROOT}/configs/criu_options.env"
    fi

    return 0
}

criu_exp_load_config_if_exists() {
    local config_path="$1"

    if [[ -f "${config_path}" ]]; then
        # shellcheck source=/dev/null
        source "${config_path}"
    else
        echo "ERROR: Config file not found: ${config_path}" >&2
        return 1
    fi
}

criu_exp_log() {
    local level="$1"
    shift

    local timestamp
    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    printf '[%s] [%s] %s\n' "${timestamp}" "${level}" "$*" >&2
}

criu_exp_info() {
    criu_exp_log "INFO" "$@"
}

criu_exp_warn() {
    criu_exp_log "WARN" "$@"
}

criu_exp_error() {
    criu_exp_log "ERROR" "$@"
}

criu_exp_die() {
    criu_exp_error "$@"
    exit 1
}

criu_exp_require_command() {
    local command_name="$1"
    local display_name="${2:-${command_name}}"

    if ! command -v "${command_name}" >/dev/null 2>&1; then
        criu_exp_error "Required command not found: ${display_name}"
        return 1
    fi

    return 0
}

criu_exp_command_exists() {
    local command_name="$1"

    command -v "${command_name}" >/dev/null 2>&1
}

criu_exp_make_dirs() {
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${BIN_DIR}"
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${RAW_RESULTS_DIR}"
    mkdir -p "${PARSED_RESULTS_DIR}"
    mkdir -p "${LOGS_DIR}"
    mkdir -p "${FIGURES_DIR}"
    mkdir -p "${CHECKPOINT_ROOT}"
    mkdir -p "${TMP_DIR}"
    mkdir -p "${RUN_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
}

criu_exp_new_run_id() {
    local prefix="${1:-run}"
    local timestamp
    timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"

    printf "%s_%s\n" "${timestamp}" "${prefix}"
}

criu_exp_reset_run_paths() {
    local run_id="$1"

    if [[ -z "${run_id}" ]]; then
        criu_exp_die "criu_exp_reset_run_paths requires a run ID"
    fi

    RUN_ID="${run_id}"
    RUN_DIR="${RAW_RESULTS_DIR}/${RUN_ID}"
    CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"

    ENV_JSON="${RUN_DIR}/env.json"

    CUDA_BASELINE_JSON="${RUN_DIR}/cuda_baseline.json"
    CHECKPOINT_RESTORE_JSON="${RUN_DIR}/checkpoint_restore.json"
    CPU_BASELINE_JSON="${RUN_DIR}/cpu_criu_baseline.json"

    CUDA_STDOUT_LOG="${RUN_DIR}/cuda_program_stdout.log"
    CUDA_STDERR_LOG="${RUN_DIR}/cuda_program_stderr.log"
    CPU_STDOUT_LOG="${RUN_DIR}/cpu_program_stdout.log"
    CPU_STDERR_LOG="${RUN_DIR}/cpu_program_stderr.log"

    CRIU_DUMP_LOG="${RUN_DIR}/criu_dump.log"
    CRIU_RESTORE_LOG="${RUN_DIR}/criu_restore.log"

    NVIDIA_SMI_BEFORE="${RUN_DIR}/nvidia_smi_before.txt"
    NVIDIA_SMI_DURING="${RUN_DIR}/nvidia_smi_during.txt"
    NVIDIA_SMI_AFTER="${RUN_DIR}/nvidia_smi_after.txt"

    CHECKPOINT_SIZE_FILE="${RUN_DIR}/checkpoint_size.txt"
    TIMING_FILE="${RUN_DIR}/checkpoint_timing.env"

    export RUN_ID
    export RUN_DIR
    export CHECKPOINT_DIR
    export ENV_JSON
    export CUDA_BASELINE_JSON
    export CHECKPOINT_RESTORE_JSON
    export CPU_BASELINE_JSON
    export CUDA_STDOUT_LOG
    export CUDA_STDERR_LOG
    export CPU_STDOUT_LOG
    export CPU_STDERR_LOG
    export CRIU_DUMP_LOG
    export CRIU_RESTORE_LOG
    export NVIDIA_SMI_BEFORE
    export NVIDIA_SMI_DURING
    export NVIDIA_SMI_AFTER
    export CHECKPOINT_SIZE_FILE
    export TIMING_FILE

    criu_exp_make_dirs
}

criu_exp_clear_cuda_markers() {
    rm -f "${CUDA_READY_FILE}"
    rm -f "${CUDA_PROGRESS_FILE}"
    rm -f "${CUDA_DONE_FILE}"
}

criu_exp_clear_cpu_markers() {
    rm -f "${CPU_READY_FILE}"
    rm -f "${CPU_PROGRESS_FILE}"
    rm -f "${CPU_DONE_FILE}"
}

criu_exp_wait_for_file() {
    local path="$1"
    local timeout_seconds="$2"
    local description="${3:-file}"

    local start_time
    local now
    local elapsed

    start_time="$(date +%s)"

    while true; do
        if [[ -f "${path}" ]]; then
            return 0
        fi

        now="$(date +%s)"
        elapsed=$((now - start_time))

        if (( elapsed >= timeout_seconds )); then
            criu_exp_error "Timed out waiting for ${description}: ${path}"
            return 1
        fi

        sleep 1
    done
}

criu_exp_read_progress_file() {
    local path="$1"

    if [[ ! -f "${path}" ]]; then
        printf "0\n"
        return 0
    fi

    local value
    value="$(cat "${path}" 2>/dev/null | tr -dc '0-9' || true)"

    if [[ -z "${value}" ]]; then
        printf "0\n"
    else
        printf "%s\n" "${value}"
    fi
}

criu_exp_wait_for_progress_at_least() {
    local path="$1"
    local target_iteration="$2"
    local timeout_seconds="$3"

    local start_time
    local now
    local elapsed
    local current_iteration

    start_time="$(date +%s)"

    while true; do
        current_iteration="$(criu_exp_read_progress_file "${path}")"

        if (( current_iteration >= target_iteration )); then
            criu_exp_info "Progress reached ${current_iteration}; target was ${target_iteration}"
            return 0
        fi

        now="$(date +%s)"
        elapsed=$((now - start_time))

        if (( elapsed >= timeout_seconds )); then
            criu_exp_error "Timed out waiting for progress ${target_iteration}; latest progress=${current_iteration}"
            return 1
        fi

        sleep 1
    done
}

criu_exp_wait_for_process_exit() {
    local pid="$1"
    local timeout_seconds="$2"

    local start_time
    local now
    local elapsed

    start_time="$(date +%s)"

    while true; do
        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            return 0
        fi

        now="$(date +%s)"
        elapsed=$((now - start_time))

        if (( elapsed >= timeout_seconds )); then
            criu_exp_error "Timed out waiting for process to exit: pid=${pid}"
            return 1
        fi

        sleep 1
    done
}

criu_exp_elapsed_ms() {
    local start_ns="$1"
    local end_ns="$2"

    awk -v s="${start_ns}" -v e="${end_ns}" 'BEGIN { printf "%.6f", (e - s) / 1000000.0 }'
}

criu_exp_now_ns() {
    date +%s%N
}

criu_exp_write_basic_env_json() {
    local output_path="$1"

    local hostname_value
    local kernel_value
    local os_pretty_name
    local nvidia_smi_path
    local nvcc_path
    local criu_path
    local cuda_checkpoint_path

    hostname_value="$(hostname 2>/dev/null || printf "unknown")"
    kernel_value="$(uname -a 2>/dev/null || printf "unknown")"

    os_pretty_name="unknown"
    if [[ -f /etc/os-release ]]; then
        os_pretty_name="$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' || printf "unknown")"
    fi

    nvidia_smi_path="$(command -v "${NVIDIA_SMI_BIN}" 2>/dev/null || true)"
    nvcc_path="$(command -v "${NVCC_BIN}" 2>/dev/null || true)"
    criu_path="$(command -v "${CRIU_BIN}" 2>/dev/null || true)"
    cuda_checkpoint_path="$(command -v "${CUDA_CHECKPOINT_BIN}" 2>/dev/null || true)"

    {
        printf "{\n"
        printf "  \"run_id\": \"%s\",\n" "${RUN_ID}"
        printf "  \"timestamp_utc\": \"%s\",\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        printf "  \"hostname\": \"%s\",\n" "${hostname_value}"
        printf "  \"kernel\": \"%s\",\n" "${kernel_value}"
        printf "  \"os_pretty_name\": \"%s\",\n" "${os_pretty_name}"
        printf "  \"repo_root\": \"%s\",\n" "${REPO_ROOT}"
        printf "  \"cuda_visible_devices\": \"%s\",\n" "${CUDA_VISIBLE_DEVICES:-}"
        printf "  \"cuda_device\": \"%s\",\n" "${CUDA_DEVICE:-}"
        printf "  \"matrix_size\": \"%s\",\n" "${MATRIX_SIZE:-}"
        printf "  \"iterations\": \"%s\",\n" "${ITERATIONS:-}"
        printf "  \"checkpoint_after_iteration\": \"%s\",\n" "${CHECKPOINT_AFTER_ITERATION:-}"
        printf "  \"paths\": {\n"
        printf "    \"nvidia_smi\": \"%s\",\n" "${nvidia_smi_path}"
        printf "    \"nvcc\": \"%s\",\n" "${nvcc_path}"
        printf "    \"criu\": \"%s\",\n" "${criu_path}"
        printf "    \"cuda_checkpoint\": \"%s\"\n" "${cuda_checkpoint_path}"
        printf "  }\n"
        printf "}\n"
    } > "${output_path}"
}

criu_exp_capture_nvidia_smi() {
    local output_path="$1"

    if criu_exp_command_exists "${NVIDIA_SMI_BIN}"; then
        "${NVIDIA_SMI_BIN}" > "${output_path}" 2>&1 || true
    else
        printf "nvidia-smi not found\n" > "${output_path}"
    fi
}

criu_exp_print_run_summary() {
    criu_exp_info "RUN_ID=${RUN_ID}"
    criu_exp_info "RUN_DIR=${RUN_DIR}"
    criu_exp_info "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
}