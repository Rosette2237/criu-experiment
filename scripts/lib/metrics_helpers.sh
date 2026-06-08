#!/usr/bin/env bash
#
# Shared metric collection helpers for CRIU CUDA checkpoint experiments.
#
# This file is meant to be sourced after scripts/lib/env.sh:
#
#   source scripts/lib/env.sh
#   source scripts/lib/metrics_helpers.sh

set -o pipefail

criu_exp_json_escape() {
    local input="$1"

    input="${input//\\/\\\\}"
    input="${input//\"/\\\"}"
    input="${input//$'\n'/\\n}"
    input="${input//$'\r'/}"

    printf "%s" "${input}"
}

criu_exp_command_output_or_empty() {
    local command_name="$1"
    shift

    if command -v "${command_name}" >/dev/null 2>&1; then
        "${command_name}" "$@" 2>/dev/null || true
    fi
}

criu_exp_command_first_line_or_empty() {
    local command_name="$1"
    shift

    criu_exp_command_output_or_empty "${command_name}" "$@" | head -n 1
}

criu_exp_get_os_pretty_name() {
    if [[ -f /etc/os-release ]]; then
        grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"' || true
    fi
}

criu_exp_get_nvidia_driver_version() {
    if command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1; then
        "${NVIDIA_SMI_BIN}" \
            --query-gpu=driver_version \
            --format=csv,noheader \
            2>/dev/null | head -n 1 || true
    fi
}

criu_exp_get_gpu_names_csv() {
    if command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1; then
        "${NVIDIA_SMI_BIN}" \
            --query-gpu=index,name,memory.total \
            --format=csv,noheader \
            2>/dev/null | sed 's/[[:space:]]*$//' || true
    fi
}

criu_exp_get_cuda_toolkit_version() {
    if command -v "${NVCC_BIN}" >/dev/null 2>&1; then
        "${NVCC_BIN}" --version 2>/dev/null | tail -n 1 || true
    fi
}

criu_exp_get_criu_version() {
    if command -v "${CRIU_BIN}" >/dev/null 2>&1; then
        "${CRIU_BIN}" --version 2>/dev/null | head -n 1 || true
    fi
}

criu_exp_get_cuda_checkpoint_version() {
    if command -v "${CUDA_CHECKPOINT_BIN}" >/dev/null 2>&1; then
        "${CUDA_CHECKPOINT_BIN}" --help 2>&1 | head -n 1 || true
    fi
}

criu_exp_write_environment_json() {
    local output_path="$1"

    mkdir -p "$(dirname "${output_path}")"

    local timestamp_utc
    local hostname_value
    local user_value
    local kernel_value
    local os_pretty_name
    local nvidia_driver_version
    local gpu_info
    local cuda_toolkit_version
    local criu_version
    local cuda_checkpoint_version
    local nvidia_smi_path
    local nvcc_path
    local criu_path
    local cuda_checkpoint_path
    local gcc_path
    local python_path
    local make_path

    timestamp_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    hostname_value="$(hostname 2>/dev/null || printf "unknown")"
    user_value="$(whoami 2>/dev/null || printf "unknown")"
    kernel_value="$(uname -a 2>/dev/null || printf "unknown")"
    os_pretty_name="$(criu_exp_get_os_pretty_name)"
    nvidia_driver_version="$(criu_exp_get_nvidia_driver_version)"
    gpu_info="$(criu_exp_get_gpu_names_csv)"
    cuda_toolkit_version="$(criu_exp_get_cuda_toolkit_version)"
    criu_version="$(criu_exp_get_criu_version)"
    cuda_checkpoint_version="$(criu_exp_get_cuda_checkpoint_version)"

    nvidia_smi_path="$(command -v "${NVIDIA_SMI_BIN}" 2>/dev/null || true)"
    nvcc_path="$(command -v "${NVCC_BIN}" 2>/dev/null || true)"
    criu_path="$(command -v "${CRIU_BIN}" 2>/dev/null || true)"
    cuda_checkpoint_path="$(command -v "${CUDA_CHECKPOINT_BIN}" 2>/dev/null || true)"
    gcc_path="$(command -v "${CC_BIN}" 2>/dev/null || true)"
    python_path="$(command -v "${PYTHON_BIN}" 2>/dev/null || true)"
    make_path="$(command -v "${MAKE_BIN}" 2>/dev/null || true)"

    {
        printf "{\n"
        printf "  \"run_id\": \"%s\",\n" "$(criu_exp_json_escape "${RUN_ID}")"
        printf "  \"timestamp_utc\": \"%s\",\n" "$(criu_exp_json_escape "${timestamp_utc}")"
        printf "  \"hostname\": \"%s\",\n" "$(criu_exp_json_escape "${hostname_value}")"
        printf "  \"user\": \"%s\",\n" "$(criu_exp_json_escape "${user_value}")"
        printf "  \"kernel\": \"%s\",\n" "$(criu_exp_json_escape "${kernel_value}")"
        printf "  \"os_pretty_name\": \"%s\",\n" "$(criu_exp_json_escape "${os_pretty_name}")"
        printf "  \"repo_root\": \"%s\",\n" "$(criu_exp_json_escape "${REPO_ROOT}")"
        printf "  \"cuda_visible_devices\": \"%s\",\n" "$(criu_exp_json_escape "${CUDA_VISIBLE_DEVICES:-}")"
        printf "  \"selected_cuda_device\": \"%s\",\n" "$(criu_exp_json_escape "${CUDA_DEVICE:-}")"
        printf "  \"nvidia_driver_version\": \"%s\",\n" "$(criu_exp_json_escape "${nvidia_driver_version}")"
        printf "  \"cuda_toolkit_version\": \"%s\",\n" "$(criu_exp_json_escape "${cuda_toolkit_version}")"
        printf "  \"criu_version\": \"%s\",\n" "$(criu_exp_json_escape "${criu_version}")"
        printf "  \"cuda_checkpoint_version\": \"%s\",\n" "$(criu_exp_json_escape "${cuda_checkpoint_version}")"
        printf "  \"gpu_info_csv\": \"%s\",\n" "$(criu_exp_json_escape "${gpu_info}")"
        printf "  \"experiment_config\": {\n"
        printf "    \"matrix_size\": \"%s\",\n" "$(criu_exp_json_escape "${MATRIX_SIZE:-}")"
        printf "    \"iterations\": \"%s\",\n" "$(criu_exp_json_escape "${ITERATIONS:-}")"
        printf "    \"block_size\": \"%s\",\n" "$(criu_exp_json_escape "${BLOCK_SIZE:-}")"
        printf "    \"sleep_ms_between_iterations\": \"%s\",\n" "$(criu_exp_json_escape "${SLEEP_MS_BETWEEN_ITERATIONS:-}")"
        printf "    \"verify\": \"%s\",\n" "$(criu_exp_json_escape "${VERIFY:-}")"
        printf "    \"verify_samples\": \"%s\",\n" "$(criu_exp_json_escape "${VERIFY_SAMPLES:-}")"
        printf "    \"checkpoint_after_iteration\": \"%s\",\n" "$(criu_exp_json_escape "${CHECKPOINT_AFTER_ITERATION:-}")"
        printf "    \"cuda_checkpoint_mode\": \"%s\",\n" "$(criu_exp_json_escape "${CUDA_CHECKPOINT_MODE:-}")"
        printf "    \"criu_log_level\": \"%s\"\n" "$(criu_exp_json_escape "${CRIU_LOG_LEVEL:-}")"
        printf "  },\n"
        printf "  \"paths\": {\n"
        printf "    \"nvidia_smi\": \"%s\",\n" "$(criu_exp_json_escape "${nvidia_smi_path}")"
        printf "    \"nvcc\": \"%s\",\n" "$(criu_exp_json_escape "${nvcc_path}")"
        printf "    \"criu\": \"%s\",\n" "$(criu_exp_json_escape "${criu_path}")"
        printf "    \"cuda_checkpoint\": \"%s\",\n" "$(criu_exp_json_escape "${cuda_checkpoint_path}")"
        printf "    \"gcc\": \"%s\",\n" "$(criu_exp_json_escape "${gcc_path}")"
        printf "    \"python3\": \"%s\",\n" "$(criu_exp_json_escape "${python_path}")"
        printf "    \"make\": \"%s\"\n" "$(criu_exp_json_escape "${make_path}")"
        printf "  }\n"
        printf "}\n"
    } > "${output_path}"

    criu_exp_info "Wrote environment metadata: ${output_path}"
}

criu_exp_append_timing_kv() {
    local output_path="$1"
    local key="$2"
    local value="$3"

    mkdir -p "$(dirname "${output_path}")"

    printf "%s=%s\n" "${key}" "${value}" >> "${output_path}"
}

criu_exp_write_exit_code_file() {
    local output_path="$1"
    local label="$2"
    local exit_code="$3"

    mkdir -p "$(dirname "${output_path}")"

    {
        printf "%s_EXIT_CODE=%s\n" "${label}" "${exit_code}"
        printf "%s_TIMESTAMP_UTC=%s\n" "${label}" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    } >> "${output_path}"
}

criu_exp_file_size_bytes() {
    local path="$1"

    if [[ ! -e "${path}" ]]; then
        printf "0\n"
        return 0
    fi

    if stat -c '%s' "${path}" >/dev/null 2>&1; then
        stat -c '%s' "${path}"
    else
        wc -c < "${path}" | tr -d ' '
    fi
}

criu_exp_directory_size_bytes() {
    local path="$1"

    if [[ ! -d "${path}" ]]; then
        printf "0\n"
        return 0
    fi

    du -sb "${path}" 2>/dev/null | awk '{print $1}'
}

criu_exp_directory_size_human() {
    local path="$1"

    if [[ ! -d "${path}" ]]; then
        printf "missing\n"
        return 0
    fi

    du -sh "${path}" 2>/dev/null | awk '{print $1}'
}

criu_exp_write_checkpoint_size_metrics() {
    local image_dir="$1"
    local output_path="$2"

    mkdir -p "$(dirname "${output_path}")"

    local exists
    local bytes
    local human

    if [[ -d "${image_dir}" ]]; then
        exists="1"
    else
        exists="0"
    fi

    bytes="$(criu_exp_directory_size_bytes "${image_dir}")"
    human="$(criu_exp_directory_size_human "${image_dir}")"

    {
        printf "CHECKPOINT_IMAGE_DIR=%s\n" "${image_dir}"
        printf "CHECKPOINT_IMAGE_EXISTS=%s\n" "${exists}"
        printf "CHECKPOINT_IMAGE_BYTES=%s\n" "${bytes}"
        printf "CHECKPOINT_IMAGE_HUMAN=%s\n" "${human}"
    } > "${output_path}"

    criu_exp_info "Checkpoint image size metrics written: ${output_path}"
}

criu_exp_extract_json_number() {
    local json_file="$1"
    local key="$2"

    if [[ ! -f "${json_file}" ]]; then
        printf "\n"
        return 0
    fi

    "${PYTHON_BIN}" - "${json_file}" "${key}" <<'PY'
import json
import sys

path = sys.argv[1]
key = sys.argv[2]

try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    value = data
    for part in key.split("."):
        value = value[part]

    print(value)
except Exception:
    print("")
PY
}

criu_exp_json_passed() {
    local json_file="$1"

    local value
    value="$(criu_exp_extract_json_number "${json_file}" "passed")"

    case "${value}" in
        True|true|1)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

criu_exp_write_run_status() {
    local output_path="$1"
    local status="$2"
    local message="$3"

    mkdir -p "$(dirname "${output_path}")"

    {
        printf "RUN_ID=%s\n" "${RUN_ID}"
        printf "STATUS=%s\n" "${status}"
        printf "MESSAGE=%s\n" "${message}"
        printf "TIMESTAMP_UTC=%s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    } > "${output_path}"
}

criu_exp_print_result_locations() {
    printf "\n" >&2
    criu_exp_info "Result files:"
    printf "  RUN_DIR:                  %s\n" "${RUN_DIR}" >&2
    printf "  ENV_JSON:                 %s\n" "${ENV_JSON}" >&2
    printf "  CUDA_BASELINE_JSON:       %s\n" "${CUDA_BASELINE_JSON}" >&2
    printf "  CHECKPOINT_RESTORE_JSON:  %s\n" "${CHECKPOINT_RESTORE_JSON}" >&2
    printf "  CPU_BASELINE_JSON:        %s\n" "${CPU_BASELINE_JSON}" >&2
    printf "  TIMING_FILE:              %s\n" "${TIMING_FILE}" >&2
    printf "  CHECKPOINT_SIZE_FILE:     %s\n" "${CHECKPOINT_SIZE_FILE}" >&2
    printf "  CHECKPOINT_DIR:           %s\n" "${CHECKPOINT_DIR}" >&2
    printf "\n" >&2
}