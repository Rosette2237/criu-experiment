#!/usr/bin/env bash
#
# Parse experiment outputs under results/raw/ and generate summary files.
#
# Run from repo root:
#
#   ./scripts/06_collect_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/env.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/lib/metrics_helpers.sh"

criu_exp_load_env
criu_exp_make_dirs

SUMMARY_CSV="${PARSED_RESULTS_DIR}/summary.csv"
SUMMARY_MD="${PARSED_RESULTS_DIR}/summary.md"

criu_exp_info "Collecting experiment results"
criu_exp_info "RAW_RESULTS_DIR=${RAW_RESULTS_DIR}"
criu_exp_info "SUMMARY_CSV=${SUMMARY_CSV}"
criu_exp_info "SUMMARY_MD=${SUMMARY_MD}"

if [[ ! -d "${RAW_RESULTS_DIR}" ]]; then
    criu_exp_die "Raw results directory does not exist: ${RAW_RESULTS_DIR}"
fi

if [[ ! -f "${REPO_ROOT}/tools/parse_results.py" ]]; then
    criu_exp_die "Missing parser: ${REPO_ROOT}/tools/parse_results.py"
fi

if [[ ! -f "${REPO_ROOT}/tools/summarize_results.py" ]]; then
    criu_exp_die "Missing summarizer: ${REPO_ROOT}/tools/summarize_results.py"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    criu_exp_die "Python not found: ${PYTHON_BIN}"
fi

criu_exp_info "Running result parser"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/parse_results.py" \
    --raw-dir "${RAW_RESULTS_DIR}" \
    --output-csv "${SUMMARY_CSV}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
    criu_exp_die "Summary CSV was not produced: ${SUMMARY_CSV}"
fi

criu_exp_info "Running result summarizer"

"${PYTHON_BIN}" "${REPO_ROOT}/tools/summarize_results.py" \
    --input-csv "${SUMMARY_CSV}" \
    --output-md "${SUMMARY_MD}"

if [[ ! -f "${SUMMARY_MD}" ]]; then
    criu_exp_die "Summary Markdown was not produced: ${SUMMARY_MD}"
fi

criu_exp_info "Result collection completed"
criu_exp_info "CSV summary: ${SUMMARY_CSV}"
criu_exp_info "Markdown summary: ${SUMMARY_MD}"

if [[ -f "${REPO_ROOT}/tools/plot_results.py" ]]; then
    criu_exp_info "Generating plots"

    set +e
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/plot_results.py" \
        --input-csv "${SUMMARY_CSV}" \
        --figures-dir "${FIGURES_DIR}"
    PLOT_EXIT_CODE=$?
    set -e

    if [[ "${PLOT_EXIT_CODE}" -eq 0 ]]; then
        criu_exp_info "Plots generated under: ${FIGURES_DIR}"
    else
        criu_exp_warn "Plot generation failed with exit code ${PLOT_EXIT_CODE}"
        criu_exp_warn "The CSV and Markdown summaries were still produced"
    fi
fi

printf "\n"
printf "Summary files:\n"
printf "  %s\n" "${SUMMARY_CSV}"
printf "  %s\n" "${SUMMARY_MD}"
printf "\n"

if [[ -s "${SUMMARY_MD}" ]]; then
    printf "Markdown summary preview:\n"
    printf "-------------------------\n"
    cat "${SUMMARY_MD}"
    printf "\n"
fi