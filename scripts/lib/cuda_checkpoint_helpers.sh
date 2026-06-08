#!/usr/bin/env bash
#
# Shared NVIDIA CUDA checkpoint helper functions.
#
# This file is meant to be sourced after scripts/lib/env.sh:
#
#   source scripts/lib/env.sh
#   source scripts/lib/cuda_checkpoint_helpers.sh

set -o pipefail

criu_exp_require_cuda_checkpoint() {
    criu_exp_require_command "${CUDA_CHECKPOINT_BIN}" "NVIDIA CUDA checkpoint utility (${CUDA_CHECKPOINT_BIN})"
}

criu_exp_cuda_checkpoint_help() {
    if ! command -v "${CUDA_CHECKPOINT_BIN}" >/dev/null 2>&1; then
        criu_exp_error "cuda-checkpoint utility not found: ${CUDA_CHECKPOINT_BIN}"
        return 1
    fi

    "${CUDA_CHECKPOINT_BIN}" --help
}

criu_exp_cuda_get_state() {
    local pid="$1"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_get_state requires a PID"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1

    "${CUDA_CHECKPOINT_BIN}" --get-state --pid "${pid}"
}

criu_exp_cuda_get_restore_tid() {
    local pid="$1"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_get_restore_tid requires a PID"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1

    "${CUDA_CHECKPOINT_BIN}" --get-restore-tid --pid "${pid}"
}

criu_exp_cuda_lock() {
    local pid="$1"
    local timeout_ms="${2:-30000}"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_lock requires a PID"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    criu_exp_info "Locking CUDA APIs for pid=${pid}, timeout_ms=${timeout_ms}"

    "${CUDA_CHECKPOINT_BIN}" \
        --action lock \
        --pid "${pid}" \
        --timeout "${timeout_ms}"
}

criu_exp_cuda_checkpoint_state() {
    local pid="$1"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_checkpoint_state requires a PID"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    criu_exp_info "Checkpointing CUDA state for pid=${pid}"

    "${CUDA_CHECKPOINT_BIN}" \
        --action checkpoint \
        --pid "${pid}"
}

criu_exp_cuda_restore_state() {
    local pid="$1"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_restore_state requires a PID"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    criu_exp_info "Restoring CUDA state for pid=${pid}"

    "${CUDA_CHECKPOINT_BIN}" \
        --action restore \
        --pid "${pid}"
}

criu_exp_cuda_unlock() {
    local pid="$1"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_unlock requires a PID"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    criu_exp_info "Unlocking CUDA APIs for pid=${pid}"

    "${CUDA_CHECKPOINT_BIN}" \
        --action unlock \
        --pid "${pid}"
}

criu_exp_cuda_toggle() {
    local pid="$1"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_toggle requires a PID"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    criu_exp_info "Toggling CUDA checkpoint state for pid=${pid}"

    "${CUDA_CHECKPOINT_BIN}" \
        --toggle \
        --pid "${pid}"
}

criu_exp_cuda_suspend_for_criu() {
    local pid="$1"
    local timing_file="$2"
    local timeout_ms="${CUDA_LOCK_TIMEOUT_MS:-30000}"

    local start_ns
    local end_ns
    local suspend_ms
    local exit_code

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_suspend_for_criu requires a PID"
        return 1
    fi

    if [[ -z "${timing_file}" ]]; then
        criu_exp_error "criu_exp_cuda_suspend_for_criu requires a timing file"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    criu_exp_info "Suspending CUDA state for CRIU: pid=${pid}"

    start_ns="$(criu_exp_now_ns)"

    set +e
    if [[ "${CUDA_CHECKPOINT_MODE}" == "cuda-checkpoint-wrapper" ]]; then
        criu_exp_cuda_lock "${pid}" "${timeout_ms}" && \
        criu_exp_cuda_checkpoint_state "${pid}"
        exit_code=$?
    else
        criu_exp_cuda_toggle "${pid}"
        exit_code=$?
    fi
    set -e

    end_ns="$(criu_exp_now_ns)"
    suspend_ms="$(criu_exp_elapsed_ms "${start_ns}" "${end_ns}")"

    {
        printf "CUDA_SUSPEND_START_NS=%s\n" "${start_ns}"
        printf "CUDA_SUSPEND_END_NS=%s\n" "${end_ns}"
        printf "CUDA_SUSPEND_WALL_MS=%s\n" "${suspend_ms}"
        printf "CUDA_SUSPEND_EXIT_CODE=%s\n" "${exit_code}"
    } >> "${timing_file}"

    if [[ "${exit_code}" -ne 0 ]]; then
        criu_exp_error "CUDA suspend failed: exit_code=${exit_code}, suspend_wall_ms=${suspend_ms}"
        return "${exit_code}"
    fi

    criu_exp_info "CUDA suspend completed: suspend_wall_ms=${suspend_ms}"
    return 0
}

criu_exp_cuda_resume_after_criu() {
    local pid="$1"
    local timing_file="$2"

    local start_ns
    local end_ns
    local resume_ms
    local exit_code

    if [[ -z "${pid}" ]]; then
        criu_exp_error "criu_exp_cuda_resume_after_criu requires a PID"
        return 1
    fi

    if [[ -z "${timing_file}" ]]; then
        criu_exp_error "criu_exp_cuda_resume_after_criu requires a timing file"
        return 1
    fi

    criu_exp_require_cuda_checkpoint || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    criu_exp_info "Resuming CUDA state after CRIU: pid=${pid}"

    start_ns="$(criu_exp_now_ns)"

    set +e
    if [[ "${CUDA_CHECKPOINT_MODE}" == "cuda-checkpoint-wrapper" ]]; then
        criu_exp_cuda_restore_state "${pid}" && \
        criu_exp_cuda_unlock "${pid}"
        exit_code=$?
    else
        criu_exp_cuda_toggle "${pid}"
        exit_code=$?
    fi
    set -e

    end_ns="$(criu_exp_now_ns)"
    resume_ms="$(criu_exp_elapsed_ms "${start_ns}" "${end_ns}")"

    {
        printf "CUDA_RESUME_START_NS=%s\n" "${start_ns}"
        printf "CUDA_RESUME_END_NS=%s\n" "${end_ns}"
        printf "CUDA_RESUME_WALL_MS=%s\n" "${resume_ms}"
        printf "CUDA_RESUME_EXIT_CODE=%s\n" "${exit_code}"
    } >> "${timing_file}"

    if [[ "${exit_code}" -ne 0 ]]; then
        criu_exp_error "CUDA resume failed: exit_code=${exit_code}, resume_wall_ms=${resume_ms}"
        return "${exit_code}"
    fi

    criu_exp_info "CUDA resume completed: resume_wall_ms=${resume_ms}"
    return 0
}

criu_exp_cuda_prepare_for_dump_if_needed() {
    local pid="$1"
    local timing_file="$2"

    case "${CUDA_CHECKPOINT_MODE}" in
        auto)
            if command -v "${CUDA_CHECKPOINT_BIN}" >/dev/null 2>&1; then
                criu_exp_info "CUDA_CHECKPOINT_MODE=auto: using cuda-checkpoint toggle before CRIU dump"
                criu_exp_cuda_suspend_for_criu "${pid}" "${timing_file}"
            else
                criu_exp_warn "CUDA_CHECKPOINT_MODE=auto: cuda-checkpoint not found; relying on CRIU CUDA plugin only"
                return 0
            fi
            ;;

        criu-plugin)
            criu_exp_info "CUDA_CHECKPOINT_MODE=criu-plugin: relying on CRIU CUDA plugin only"
            return 0
            ;;

        cuda-checkpoint-wrapper)
            criu_exp_info "CUDA_CHECKPOINT_MODE=cuda-checkpoint-wrapper: using explicit lock/checkpoint sequence"
            criu_exp_cuda_suspend_for_criu "${pid}" "${timing_file}"
            ;;

        *)
            criu_exp_error "Unsupported CUDA_CHECKPOINT_MODE: ${CUDA_CHECKPOINT_MODE}"
            return 1
            ;;
    esac
}

criu_exp_cuda_resume_after_restore_if_needed() {
    local pid="$1"
    local timing_file="$2"

    case "${CUDA_CHECKPOINT_MODE}" in
        auto)
            if command -v "${CUDA_CHECKPOINT_BIN}" >/dev/null 2>&1; then
                criu_exp_info "CUDA_CHECKPOINT_MODE=auto: using cuda-checkpoint toggle after CRIU restore"
                criu_exp_cuda_resume_after_criu "${pid}" "${timing_file}"
            else
                criu_exp_warn "CUDA_CHECKPOINT_MODE=auto: cuda-checkpoint not found; relying on CRIU CUDA plugin only"
                return 0
            fi
            ;;

        criu-plugin)
            criu_exp_info "CUDA_CHECKPOINT_MODE=criu-plugin: no explicit cuda-checkpoint resume"
            return 0
            ;;

        cuda-checkpoint-wrapper)
            criu_exp_info "CUDA_CHECKPOINT_MODE=cuda-checkpoint-wrapper: using explicit restore/unlock sequence"
            criu_exp_cuda_resume_after_criu "${pid}" "${timing_file}"
            ;;

        *)
            criu_exp_error "Unsupported CUDA_CHECKPOINT_MODE: ${CUDA_CHECKPOINT_MODE}"
            return 1
            ;;
    esac
}