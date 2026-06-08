#!/usr/bin/env python3
#
# Convert parsed CRIU/CUDA experiment CSV results into a Markdown summary.
#
# Example:
#
#   python3 tools/summarize_results.py \
#       --input-csv results/parsed/summary.csv \
#       --output-md results/parsed/summary.md

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str) -> Optional[float]:
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def safe_int(value: str) -> Optional[int]:
    number = safe_float(value)

    if number is None:
        return None

    return int(number)


def format_ms(value: str) -> str:
    number = safe_float(value)

    if number is None:
        return ""

    return f"{number:.3f}"


def format_bytes(value: str) -> str:
    number = safe_float(value)

    if number is None:
        return ""

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(number)

    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{int(number)} B"


def markdown_table(headers: List[str], rows: Iterable[List[str]]) -> str:
    rows = list(rows)

    if not rows:
        return "_No rows._\n"

    output = []
    output.append("| " + " | ".join(headers) + " |")
    output.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        escaped = [str(cell).replace("|", "\\|") for cell in row]
        output.append("| " + " | ".join(escaped) + " |")

    return "\n".join(output) + "\n"


def summarize_counts(rows: List[Dict[str, str]]) -> str:
    by_type = Counter(row.get("run_type", "unknown") for row in rows)
    by_status = Counter(row.get("status", "unknown") for row in rows)

    output = []
    output.append("## Run counts\n")
    output.append("### By run type\n")

    type_rows = [[run_type, str(count)] for run_type, count in sorted(by_type.items())]
    output.append(markdown_table(["Run type", "Count"], type_rows))

    output.append("\n### By status\n")

    status_rows = [[status, str(count)] for status, count in sorted(by_status.items())]
    output.append(markdown_table(["Status", "Count"], status_rows))

    return "\n".join(output)


def summarize_cuda_baselines(rows: List[Dict[str, str]]) -> str:
    baseline_rows = [
        row for row in rows
        if row.get("run_type") == "cuda_baseline"
    ]

    output = []
    output.append("## CUDA baseline runs\n")

    table_rows = []

    for row in baseline_rows:
        table_rows.append([
            row.get("run_id", ""),
            row.get("status", ""),
            row.get("matrix_size", ""),
            row.get("iterations", ""),
            row.get("completed_iterations", ""),
            format_ms(row.get("total_program_ms", "")),
            format_ms(row.get("kernel_total_ms", "")),
            row.get("passed", ""),
            row.get("run_dir", ""),
        ])

    output.append(markdown_table(
        [
            "Run ID",
            "Status",
            "Matrix size",
            "Iterations",
            "Completed",
            "Program ms",
            "Kernel ms",
            "Passed",
            "Run dir",
        ],
        table_rows,
    ))

    return "\n".join(output)


def summarize_cuda_criu(rows: List[Dict[str, str]]) -> str:
    criu_rows = [
        row for row in rows
        if row.get("run_type") == "cuda_criu_checkpoint_restore"
    ]

    output = []
    output.append("## CUDA CRIU checkpoint/restore runs\n")

    table_rows = []

    for row in criu_rows:
        table_rows.append([
            row.get("run_id", ""),
            row.get("status", ""),
            row.get("matrix_size", ""),
            row.get("iterations", ""),
            row.get("completed_iterations", ""),
            format_ms(row.get("dump_wall_ms", "")),
            format_ms(row.get("restore_wall_ms", "")),
            format_ms(row.get("cuda_suspend_wall_ms", "")),
            format_ms(row.get("cuda_resume_wall_ms", "")),
            format_ms(row.get("cuda_program_total_wall_ms", "")),
            format_bytes(row.get("checkpoint_image_bytes", "")),
            row.get("passed", ""),
            row.get("run_dir", ""),
        ])

    output.append(markdown_table(
        [
            "Run ID",
            "Status",
            "Matrix size",
            "Iterations",
            "Completed",
            "Dump ms",
            "Restore ms",
            "CUDA suspend ms",
            "CUDA resume ms",
            "Total wall ms",
            "Checkpoint size",
            "Passed",
            "Run dir",
        ],
        table_rows,
    ))

    return "\n".join(output)


def summarize_cpu_criu(rows: List[Dict[str, str]]) -> str:
    cpu_rows = [
        row for row in rows
        if row.get("run_type") == "cpu_criu_baseline"
    ]

    output = []
    output.append("## CPU-only CRIU baseline runs\n")

    table_rows = []

    for row in cpu_rows:
        table_rows.append([
            row.get("run_id", ""),
            row.get("status", ""),
            row.get("iterations", ""),
            row.get("completed_iterations", ""),
            format_ms(row.get("dump_wall_ms", "")),
            format_ms(row.get("restore_wall_ms", "")),
            format_bytes(row.get("checkpoint_image_bytes", "")),
            row.get("passed", ""),
            row.get("run_dir", ""),
        ])

    output.append(markdown_table(
        [
            "Run ID",
            "Status",
            "Iterations",
            "Completed",
            "Dump ms",
            "Restore ms",
            "Checkpoint size",
            "Passed",
            "Run dir",
        ],
        table_rows,
    ))

    return "\n".join(output)


def summarize_scaling(rows: List[Dict[str, str]]) -> str:
    criu_rows = [
        row for row in rows
        if row.get("run_type") == "cuda_criu_checkpoint_restore"
        and row.get("status") == "success"
    ]

    by_size: Dict[int, List[Dict[str, str]]] = defaultdict(list)

    for row in criu_rows:
        matrix_size = safe_int(row.get("matrix_size", ""))

        if matrix_size is not None:
            by_size[matrix_size].append(row)

    output = []
    output.append("## CUDA CRIU scaling summary\n")

    table_rows = []

    for matrix_size in sorted(by_size):
        group = by_size[matrix_size]

        dump_values = [
            safe_float(row.get("dump_wall_ms", ""))
            for row in group
            if safe_float(row.get("dump_wall_ms", "")) is not None
        ]

        restore_values = [
            safe_float(row.get("restore_wall_ms", ""))
            for row in group
            if safe_float(row.get("restore_wall_ms", "")) is not None
        ]

        checkpoint_sizes = [
            safe_float(row.get("checkpoint_image_bytes", ""))
            for row in group
            if safe_float(row.get("checkpoint_image_bytes", "")) is not None
        ]

        gpu_memory = [
            safe_float(row.get("gpu_memory_allocated_bytes", ""))
            for row in group
            if safe_float(row.get("gpu_memory_allocated_bytes", "")) is not None
        ]

        avg_dump = sum(dump_values) / len(dump_values) if dump_values else None
        avg_restore = sum(restore_values) / len(restore_values) if restore_values else None
        avg_checkpoint_size = sum(checkpoint_sizes) / len(checkpoint_sizes) if checkpoint_sizes else None
        avg_gpu_memory = sum(gpu_memory) / len(gpu_memory) if gpu_memory else None

        table_rows.append([
            str(matrix_size),
            str(len(group)),
            f"{avg_dump:.3f}" if avg_dump is not None else "",
            f"{avg_restore:.3f}" if avg_restore is not None else "",
            format_bytes(str(avg_checkpoint_size)) if avg_checkpoint_size is not None else "",
            format_bytes(str(avg_gpu_memory)) if avg_gpu_memory is not None else "",
        ])

    output.append(markdown_table(
        [
            "Matrix size",
            "Successful runs",
            "Avg dump ms",
            "Avg restore ms",
            "Avg checkpoint size",
            "Avg explicit GPU allocation",
        ],
        table_rows,
    ))

    return "\n".join(output)


def summarize_environment(rows: List[Dict[str, str]]) -> str:
    output = []
    output.append("## Environment observed\n")

    if not rows:
        output.append("_No environment data available._\n")
        return "\n".join(output)

    latest = rows[-1]

    table_rows = [
        ["Hostname", latest.get("hostname", "")],
        ["OS", latest.get("os_pretty_name", "")],
        ["Kernel", latest.get("kernel", "")],
        ["NVIDIA driver", latest.get("nvidia_driver_version", "")],
        ["CUDA toolkit", latest.get("cuda_toolkit_version", "")],
        ["CRIU", latest.get("criu_version", "")],
        ["cuda-checkpoint", latest.get("cuda_checkpoint_version", "")],
        ["CUDA_VISIBLE_DEVICES", latest.get("cuda_visible_devices", "")],
        ["Selected CUDA device", latest.get("selected_cuda_device", "")],
        ["CUDA checkpoint mode", latest.get("cuda_checkpoint_mode", "")],
    ]

    output.append(markdown_table(["Field", "Value"], table_rows))

    return "\n".join(output)


def write_summary(rows: List[Dict[str, str]], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# CRIU CUDA checkpoint experiment summary\n")
    lines.append(f"Parsed runs: **{len(rows)}**\n")

    lines.append(summarize_counts(rows))
    lines.append(summarize_environment(rows))
    lines.append(summarize_cuda_baselines(rows))
    lines.append(summarize_cpu_criu(rows))
    lines.append(summarize_cuda_criu(rows))
    lines.append(summarize_scaling(rows))

    output_md.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CRIU/CUDA experiment CSV results.")
    parser.add_argument("--input-csv", required=True, help="Input CSV path from parse_results.py.")
    parser.add_argument("--output-md", required=True, help="Output Markdown summary path.")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_md = Path(args.output_md)

    rows = read_csv(input_csv)
    write_summary(rows, output_md)

    print(f"Read rows: {len(rows)}")
    print(f"Wrote Markdown summary: {output_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())