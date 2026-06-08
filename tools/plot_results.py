#!/usr/bin/env python3
#
# Generate simple plots from parsed CRIU/CUDA experiment results.
#
# Example:
#
#   python3 tools/plot_results.py \
#       --input-csv results/parsed/summary.csv \
#       --figures-dir results/figures

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    value_float = safe_float(value)

    if value_float is None:
        return None

    return int(value_float)


def filter_successful_cuda_criu(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row for row in rows
        if row.get("run_type") == "cuda_criu_checkpoint_restore"
        and row.get("status") == "success"
    ]


def group_average_by_matrix_size(
    rows: List[Dict[str, str]],
    value_key: str,
) -> List[Tuple[int, float]]:
    grouped: Dict[int, List[float]] = {}

    for row in rows:
        matrix_size = safe_int(row.get("matrix_size", ""))
        value = safe_float(row.get(value_key, ""))

        if matrix_size is None or value is None:
            continue

        grouped.setdefault(matrix_size, []).append(value)

    averaged = []

    for matrix_size, values in grouped.items():
        if not values:
            continue

        averaged.append((matrix_size, sum(values) / len(values)))

    return sorted(averaged)


def import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as exc:
        print(f"Could not import matplotlib: {exc}")
        print("Install matplotlib if plots are needed:")
        print("  python3 -m pip install matplotlib")
        return None


def plot_line(
    plt,
    points: List[Tuple[int, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    if not points:
        print(f"Skipping plot with no data: {output_path}")
        return False

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Wrote plot: {output_path}")
    return True


def plot_dump_time(rows: List[Dict[str, str]], figures_dir: Path, plt) -> bool:
    points = group_average_by_matrix_size(rows, "dump_wall_ms")

    return plot_line(
        plt=plt,
        points=points,
        title="CRIU dump time vs matrix size",
        xlabel="Matrix size N",
        ylabel="Average dump time, ms",
        output_path=figures_dir / "checkpoint_time_vs_matrix_size.png",
    )


def plot_restore_time(rows: List[Dict[str, str]], figures_dir: Path, plt) -> bool:
    points = group_average_by_matrix_size(rows, "restore_wall_ms")

    return plot_line(
        plt=plt,
        points=points,
        title="CRIU restore time vs matrix size",
        xlabel="Matrix size N",
        ylabel="Average restore time, ms",
        output_path=figures_dir / "restore_time_vs_matrix_size.png",
    )


def plot_checkpoint_size(rows: List[Dict[str, str]], figures_dir: Path, plt) -> bool:
    points = group_average_by_matrix_size(rows, "checkpoint_image_bytes")

    points_mib = [
        (matrix_size, bytes_value / (1024.0 * 1024.0))
        for matrix_size, bytes_value in points
    ]

    return plot_line(
        plt=plt,
        points=points_mib,
        title="Checkpoint image size vs matrix size",
        xlabel="Matrix size N",
        ylabel="Average checkpoint image size, MiB",
        output_path=figures_dir / "checkpoint_size_vs_matrix_size.png",
    )


def plot_gpu_memory(rows: List[Dict[str, str]], figures_dir: Path, plt) -> bool:
    points = group_average_by_matrix_size(rows, "gpu_memory_allocated_bytes")

    points_mib = [
        (matrix_size, bytes_value / (1024.0 * 1024.0))
        for matrix_size, bytes_value in points
    ]

    return plot_line(
        plt=plt,
        points=points_mib,
        title="Explicit GPU allocation vs matrix size",
        xlabel="Matrix size N",
        ylabel="Explicit GPU allocation, MiB",
        output_path=figures_dir / "gpu_allocation_vs_matrix_size.png",
    )


def plot_total_wall_time(rows: List[Dict[str, str]], figures_dir: Path, plt) -> bool:
    points = group_average_by_matrix_size(rows, "cuda_program_total_wall_ms")

    return plot_line(
        plt=plt,
        points=points,
        title="Total checkpointed program wall time vs matrix size",
        xlabel="Matrix size N",
        ylabel="Average total wall time, ms",
        output_path=figures_dir / "total_wall_time_vs_matrix_size.png",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CRIU/CUDA checkpoint experiment results.")
    parser.add_argument("--input-csv", required=True, help="Input CSV path from parse_results.py.")
    parser.add_argument("--figures-dir", required=True, help="Directory where plot PNG files should be written.")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    figures_dir = Path(args.figures_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_csv)
    cuda_criu_rows = filter_successful_cuda_criu(rows)

    if not cuda_criu_rows:
        print("No successful CUDA CRIU checkpoint/restore rows found. No plots generated.")
        return 0

    plt = import_matplotlib()

    if plt is None:
        return 1

    generated = 0

    generated += int(plot_dump_time(cuda_criu_rows, figures_dir, plt))
    generated += int(plot_restore_time(cuda_criu_rows, figures_dir, plt))
    generated += int(plot_checkpoint_size(cuda_criu_rows, figures_dir, plt))
    generated += int(plot_gpu_memory(cuda_criu_rows, figures_dir, plt))
    generated += int(plot_total_wall_time(cuda_criu_rows, figures_dir, plt))

    print(f"Generated {generated} plot file(s) under {figures_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())