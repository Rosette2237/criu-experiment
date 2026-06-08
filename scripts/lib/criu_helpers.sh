#!/usr/bin/env bash
#
# Shared CRIU helper functions.
#
# This file is meant to be sourced after scripts/lib/env.sh:
#
#   source scripts/lib/env.sh
#   source scripts/lib/criu_helpers.sh

set -o pipefail

criu_exp_require_criu() {
    criu_exp_require_command "${CRIU_BIN}" "CRIU (${CRIU_BIN})"
}

criu_exp_check_criu_basic() {
    criu_exp_info "Running CRIU basic check"

    if ! "${CRIU_SUDO}" "${CRIU_BIN}" check; then
        criu_exp_error "CRIU basic check failed"
        return 1
    fi

    criu_exp_info "CRIU basic check passed"
    return 0
}

criu_exp_check_criu_all() {
    criu_exp_info "Running CRIU full check"

    if ! "${CRIU_SUDO}" "${CRIU_BIN}" check --all; then
        criu_exp_error "CRIU full check failed"
        return 1
    fi

    criu_exp_info "CRIU full check passed"
    return 0
}

criu_exp_assert_pid_alive() {
    local pid="$1"

    if [[ -z "${pid}" ]]; then
        criu_exp_error "PID is empty"
        return 1
    fi

    if ! kill -0 "${pid}" >/dev/null 2>&1; then
        criu_exp_error "Process is not alive: pid=${pid}"
        return 1
    fi

    return 0
}

criu_exp_build_common_args() {
    CRIU_COMMON_ARGS=()

    CRIU_COMMON_ARGS+=("-v${CRIU_LOG_LEVEL}")

    if [[ "${CRIU_SHELL_JOB}" == "1" ]]; then
        CRIU_COMMON_ARGS+=("--shell-job")
    fi

    if [[ "${CRIU_FILE_LOCKS}" == "1" ]]; then
        CRIU_COMMON_ARGS+=("--file-locks")
    fi

    if [[ "${CRIU_LINK_REMAP}" == "1" ]]; then
        CRIU_COMMON_ARGS+=("--link-remap")
    fi

    if [[ "${CRIU_TCP_ESTABLISHED}" == "1" ]]; then
        CRIU_COMMON_ARGS+=("--tcp-established")
    fi
}

criu_exp_build_dump_args() {
    local pid="$1"
    local image_dir="$2"
    local log_file="$3"

    CRIU_DUMP_ARGS=()

    criu_exp_build_common_args

    CRIU_DUMP_ARGS+=("dump")
    CRIU_DUMP_ARGS+=("-t" "${pid}")
    CRIU_DUMP_ARGS+=("-D" "${image_dir}")
    CRIU_DUMP_ARGS+=("-o" "${log_file}")

    CRIU_DUMP_ARGS+=("${CRIU_COMMON_ARGS[@]}")

    if [[ "${CRIU_LEAVE_RUNNING}" == "1" ]]; then
        CRIU_DUMP_ARGS+=("--leave-running")
    fi

    if [[ "${CRIU_TRACK_MEM}" == "1" ]]; then
        CRIU_DUMP_ARGS+=("--track-mem")
    fi

    if [[ -n "${EXTRA_CRIU_DUMP_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        local extra_args=( ${EXTRA_CRIU_DUMP_ARGS} )
        CRIU_DUMP_ARGS+=("${extra_args[@]}")
    fi
}

criu_exp_build_restore_args() {
    local image_dir="$1"
    local log_file="$2"

    CRIU_RESTORE_ARGS=()

    criu_exp_build_common_args

    CRIU_RESTORE_ARGS+=("restore")
    CRIU_RESTORE_ARGS+=("-D" "${image_dir}")
    CRIU_RESTORE_ARGS+=("-o" "${log_file}")

    CRIU_RESTORE_ARGS+=("${CRIU_COMMON_ARGS[@]}")

    if [[ "${CRIU_RESTORE_DETACHED}" == "1" ]]; then
        CRIU_RESTORE_ARGS+=("-d")
    fi

    if [[ "${CRIU_EXT_TTY}" == "1" ]]; then
        CRIU_RESTORE_ARGS+=("--ext-tty")
    fi

    if [[ -n "${EXTRA_CRIU_RESTORE_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        local extra_args=( ${EXTRA_CRIU_RESTORE_ARGS} )
        CRIU_RESTORE_ARGS+=("${extra_args[@]}")
    fi
}

criu_exp_dump_process() {
    local pid="$1"
    local image_dir="$2"
    local log_file="$3"
    local timing_file="$4"

    local start_ns
    local end_ns
    local dump_ms
    local exit_code

    criu_exp_require_criu || return 1
    criu_exp_assert_pid_alive "${pid}" || return 1

    mkdir -p "${image_dir}"
    mkdir -p "$(dirname "${log_file}")"

    criu_exp_info "Starting CRIU dump for pid=${pid}"
    criu_exp_info "Checkpoint image directory: ${image_dir}"
    criu_exp_info "CRIU dump log: ${log_file}"

    criu_exp_build_dump_args "${pid}" "${image_dir}" "${log_file}"

    start_ns="$(criu_exp_now_ns)"

    set +e
    timeout "${CRIU_DUMP_TIMEOUT_SECONDS}s" \
        "${CRIU_SUDO}" "${CRIU_BIN}" "${CRIU_DUMP_ARGS[@]}"
    exit_code=$?
    set -e

    end_ns="$(criu_exp_now_ns)"
    dump_ms="$(criu_exp_elapsed_ms "${start_ns}" "${end_ns}")"

    {
        printf "DUMP_START_NS=%s\n" "${start_ns}"
        printf "DUMP_END_NS=%s\n" "${end_ns}"
        printf "DUMP_WALL_MS=%s\n" "${dump_ms}"
        printf "DUMP_EXIT_CODE=%s\n" "${exit_code}"
    } >> "${timing_file}"

    if [[ "${exit_code}" -ne 0 ]]; then
        criu_exp_error "CRIU dump failed: exit_code=${exit_code}, dump_wall_ms=${dump_ms}"
        return "${exit_code}"
    fi

    criu_exp_info "CRIU dump completed: dump_wall_ms=${dump_ms}"
    return 0
}

criu_exp_restore_process() {
    local image_dir="$1"
    local log_file="$2"
    local timing_file="$3"

    local start_ns
    local end_ns
    local restore_ms
    local exit_code

    criu_exp_require_criu || return 1

    if [[ ! -d "${image_dir}" ]]; then
        criu_exp_error "Checkpoint image directory does not exist: ${image_dir}"
        return 1
    fi

    mkdir -p "$(dirname "${log_file}")"

    criu_exp_info "Starting CRIU restore"
    criu_exp_info "Checkpoint image directory: ${image_dir}"
    criu_exp_info "CRIU restore log: ${log_file}"

    criu_exp_build_restore_args "${image_dir}" "${log_file}"

    start_ns="$(criu_exp_now_ns)"

    set +e
    timeout "${CRIU_RESTORE_TIMEOUT_SECONDS}s" \
        "${CRIU_SUDO}" "${CRIU_BIN}" "${CRIU_RESTORE_ARGS[@]}"
    exit_code=$?
    set -e

    end_ns="$(criu_exp_now_ns)"
    restore_ms="$(criu_exp_elapsed_ms "${start_ns}" "${end_ns}")"

    {
        printf "RESTORE_START_NS=%s\n" "${start_ns}"
        printf "RESTORE_END_NS=%s\n" "${end_ns}"
        printf "RESTORE_WALL_MS=%s\n" "${restore_ms}"
        printf "RESTORE_EXIT_CODE=%s\n" "${exit_code}"
    } >> "${timing_file}"

    if [[ "${exit_code}" -ne 0 ]]; then
        criu_exp_error "CRIU restore failed: exit_code=${exit_code}, restore_wall_ms=${restore_ms}"
        return "${exit_code}"
    fi

    criu_exp_info "CRIU restore completed: restore_wall_ms=${restore_ms}"
    return 0
}

criu_exp_write_checkpoint_size() {
    local image_dir="$1"
    local output_file="$2"

    mkdir -p "$(dirname "${output_file}")"

    if [[ ! -d "${image_dir}" ]]; then
        {
            printf "CHECKPOINT_IMAGE_DIR=%s\n" "${image_dir}"
            printf "CHECKPOINT_IMAGE_EXISTS=0\n"
            printf "CHECKPOINT_IMAGE_BYTES=0\n"
            printf "CHECKPOINT_IMAGE_HUMAN=missing\n"
        } > "${output_file}"

        criu_exp_warn "Checkpoint image directory missing: ${image_dir}"
        return 1
    fi

    local bytes
    local human

    bytes="$(du -sb "${image_dir}" 2>/dev/null | awk '{print $1}')"
    human="$(du -sh "${image_dir}" 2>/dev/null | awk '{print $1}')"

    if [[ -z "${bytes}" ]]; then
        bytes="0"
    fi

    if [[ -z "${human}" ]]; then
        human="unknown"
    fi

    {
        printf "CHECKPOINT_IMAGE_DIR=%s\n" "${image_dir}"
        printf "CHECKPOINT_IMAGE_EXISTS=1\n"
        printf "CHECKPOINT_IMAGE_BYTES=%s\n" "${bytes}"
        printf "CHECKPOINT_IMAGE_HUMAN=%s\n" "${human}"
    } > "${output_file}"

    criu_exp_info "Checkpoint image size: ${human} (${bytes} bytes)"
    return 0
}

criu_exp_print_criu_command_preview() {
    local mode="$1"

    case "${mode}" in
        dump)
            printf "CRIU dump command:\n" >&2
            printf "  %q " "${CRIU_SUDO}" "${CRIU_BIN}" "${CRIU_DUMP_ARGS[@]}" >&2
            printf "\n" >&2
            ;;
        restore)
            printf "CRIU restore command:\n" >&2
            printf "  %q " "${CRIU_SUDO}" "${CRIU_BIN}" "${CRIU_RESTORE_ARGS[@]}" >&2
            printf "\n" >&2
            ;;
        *)
            criu_exp_warn "Unknown CRIU command preview mode: ${mode}"
            ;;
    esac
}