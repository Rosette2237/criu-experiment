#!/usr/bin/env bash
#
# Environment check for the CRIU CUDA checkpoint/restore experiment.
#
# Run from repo root:
#
#   ./scripts/00_env_check.sh

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
criu_exp_reset_run_paths "$(criu_exp_new_run_id env_check)"

criu_exp_info "Starting environment check"
criu_exp_print_run_summary

criu_exp_make_dirs
criu_exp_write_environment_json "${ENV_JSON}"

CHECK_FAILED=0

check_command_required() {
    local command_name="$1"
    local display_name="$2"

    if command -v "${command_name}" >/dev/null 2>&1; then
        criu_exp_info "Found ${display_name}: $(command -v "${command_name}")"
    else
        criu_exp_error "Missing required command: ${display_name} (${command_name})"
        CHECK_FAILED=1
    fi
}

check_command_optional() {
    local command_name="$1"
    local display_name="$2"

    if command -v "${command_name}" >/dev/null 2>&1; then
        criu_exp_info "Found optional command ${display_name}: $(command -v "${command_name}")"
    else
        criu_exp_warn "Optional command not found: ${display_name} (${command_name})"
    fi
}

check_file_exists() {
    local path="$1"
    local description="$2"

    if [[ -e "${path}" ]]; then
        criu_exp_info "Found ${description}: ${path}"
    else
        criu_exp_error "Missing ${description}: ${path}"
        CHECK_FAILED=1
    fi
}

check_directory_exists() {
    local path="$1"
    local description="$2"

    if [[ -d "${path}" ]]; then
        criu_exp_info "Found ${description}: ${path}"
    else
        criu_exp_warn "Creating missing ${description}: ${path}"
        mkdir -p "${path}"
    fi
}

criu_exp_info "Checking repository structure"

check_file_exists "${REPO_ROOT}/Makefile" "Makefile"
check_file_exists "${REPO_ROOT}/configs/default.env" "default config"
check_file_exists "${REPO_ROOT}/configs/criu_options.env" "CRIU config"
check_file_exists "${REPO_ROOT}/configs/single_gpu.env" "single-GPU config"
check_file_exists "${REPO_ROOT}/src/cuda_matmul/matmul_bench.cu" "CUDA benchmark source"
check_file_exists "${REPO_ROOT}/src/cuda_matmul/matmul_config.h" "CUDA benchmark config header"
check_file_exists "${REPO_ROOT}/src/cpu_baseline/cpu_sleep_loop.c" "CPU baseline source"
check_file_exists "${REPO_ROOT}/src/common/timing.h" "timing helper header"
check_file_exists "${REPO_ROOT}/src/common/logging.h" "logging helper header"
check_file_exists "${REPO_ROOT}/src/common/signal_markers.h" "marker helper header"

check_directory_exists "${BUILD_DIR}" "build directory"
check_directory_exists "${BIN_DIR}" "binary directory"
check_directory_exists "${RAW_RESULTS_DIR}" "raw results directory"
check_directory_exists "${PARSED_RESULTS_DIR}" "parsed results directory"
check_directory_exists "${LOGS_DIR}" "logs directory"
check_directory_exists "${FIGURES_DIR}" "figures directory"
check_directory_exists "${CHECKPOINT_ROOT}" "checkpoint image root"
check_directory_exists "${TMP_DIR}" "temporary directory"

criu_exp_info "Checking required commands"

check_command_required "bash" "Bash"
check_command_required "${MAKE_BIN}" "make"
check_command_required "${CC_BIN}" "C compiler"
check_command_required "${PYTHON_BIN}" "Python 3"
check_command_required "${NVIDIA_SMI_BIN}" "nvidia-smi"
check_command_required "${NVCC_BIN}" "nvcc"
check_command_required "${CRIU_BIN}" "CRIU"

criu_exp_info "Checking optional commands"

check_command_optional "${CUDA_CHECKPOINT_BIN}" "cuda-checkpoint"
check_command_optional "jq" "jq"
check_command_optional "bc" "bc"
check_command_optional "timeout" "timeout"
check_command_optional "du" "du"
check_command_optional "awk" "awk"
check_command_optional "sed" "sed"
check_command_optional "grep" "grep"

criu_exp_info "Printing basic system information"

{
    echo "===== uname -a ====="
    uname -a || true

    echo
    echo "===== /etc/os-release ====="
    cat /etc/os-release 2>/dev/null || true

    echo
    echo "===== whoami / id ====="
    whoami || true
    id || true

    echo
    echo "===== PATH ====="
    echo "${PATH}"

    echo
    echo "===== CUDA_VISIBLE_DEVICES ====="
    echo "${CUDA_VISIBLE_DEVICES:-unset}"
} | tee "${RUN_DIR}/system_info.txt"

criu_exp_info "Checking NVIDIA GPU visibility"

if command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1; then
    if "${NVIDIA_SMI_BIN}" > "${NVIDIA_SMI_BEFORE}" 2>&1; then
        criu_exp_info "nvidia-smi succeeded"
        cat "${NVIDIA_SMI_BEFORE}"
    else
        criu_exp_error "nvidia-smi failed; see ${NVIDIA_SMI_BEFORE}"
        CHECK_FAILED=1
    fi

    {
        echo "===== nvidia-smi query ====="
        "${NVIDIA_SMI_BIN}" \
            --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free \
            --format=csv \
            2>/dev/null || true
    } > "${RUN_DIR}/nvidia_smi_query.txt"
else
    criu_exp_error "nvidia-smi is unavailable"
    CHECK_FAILED=1
fi

criu_exp_info "Checking CUDA compiler"

if command -v "${NVCC_BIN}" >/dev/null 2>&1; then
    "${NVCC_BIN}" --version | tee "${RUN_DIR}/nvcc_version.txt"
else
    criu_exp_error "nvcc is unavailable"
    CHECK_FAILED=1
fi

criu_exp_info "Checking CRIU"

if command -v "${CRIU_BIN}" >/dev/null 2>&1; then
    "${CRIU_BIN}" --version | tee "${RUN_DIR}/criu_version.txt"

    set +e
    "${CRIU_SUDO}" "${CRIU_BIN}" check > "${RUN_DIR}/criu_check.log" 2>&1
    CRIU_CHECK_EXIT=$?
    set -e

    if [[ "${CRIU_CHECK_EXIT}" -eq 0 ]]; then
        criu_exp_info "sudo criu check passed"
    else
        criu_exp_error "sudo criu check failed; see ${RUN_DIR}/criu_check.log"
        CHECK_FAILED=1
    fi

    set +e
    "${CRIU_SUDO}" "${CRIU_BIN}" check --all > "${RUN_DIR}/criu_check_all.log" 2>&1
    CRIU_CHECK_ALL_EXIT=$?
    set -e

    if [[ "${CRIU_CHECK_ALL_EXIT}" -eq 0 ]]; then
        criu_exp_info "sudo criu check --all passed"
    else
        criu_exp_warn "sudo criu check --all failed; see ${RUN_DIR}/criu_check_all.log"
    fi
else
    criu_exp_error "CRIU is unavailable"
    CHECK_FAILED=1
fi

criu_exp_info "Checking cuda-checkpoint utility"

if command -v "${CUDA_CHECKPOINT_BIN}" >/dev/null 2>&1; then
    set +e
    "${CUDA_CHECKPOINT_BIN}" --help > "${RUN_DIR}/cuda_checkpoint_help.txt" 2>&1
    CUDA_CHECKPOINT_HELP_EXIT=$?
    set -e

    if [[ "${CUDA_CHECKPOINT_HELP_EXIT}" -eq 0 ]]; then
        criu_exp_info "cuda-checkpoint --help succeeded"
    else
        criu_exp_warn "cuda-checkpoint exists but --help returned ${CUDA_CHECKPOINT_HELP_EXIT}"
    fi
else
    criu_exp_warn "cuda-checkpoint utility not found"
    criu_exp_warn "CUDA_CHECKPOINT_MODE=${CUDA_CHECKPOINT_MODE:-unset}; direct CRIU CUDA plugin mode may still be tested"
fi

criu_exp_info "Checking sudo access"

set +e
sudo -n true >/dev/null 2>&1
SUDO_NONINTERACTIVE_EXIT=$?
set -e

if [[ "${SUDO_NONINTERACTIVE_EXIT}" -eq 0 ]]; then
    criu_exp_info "sudo is available without prompting"
else
    criu_exp_warn "sudo may require an interactive password prompt"
    criu_exp_warn "If later CRIU scripts appear to hang, run 'sudo true' first"
fi

criu_exp_info "Checking whether benchmark binaries already exist"

if [[ -x "${CUDA_BENCH_BIN}" ]]; then
    criu_exp_info "CUDA benchmark binary exists: ${CUDA_BENCH_BIN}"
else
    criu_exp_warn "CUDA benchmark binary not found yet: ${CUDA_BENCH_BIN}"
    criu_exp_warn "Run ./scripts/01_build.sh next"
fi

if [[ -x "${CPU_BASELINE_BIN}" ]]; then
    criu_exp_info "CPU baseline binary exists: ${CPU_BASELINE_BIN}"
else
    criu_exp_warn "CPU baseline binary not found yet: ${CPU_BASELINE_BIN}"
    criu_exp_warn "Run ./scripts/01_build.sh next"
fi

criu_exp_print_result_locations

if [[ "${CHECK_FAILED}" -ne 0 ]]; then
    criu_exp_error "Environment check completed with failures"
    criu_exp_error "Inspect run directory: ${RUN_DIR}"
    exit 1
fi

criu_exp_info "Environment check completed successfully"
criu_exp_info "Next step: ./scripts/01_build.sh"