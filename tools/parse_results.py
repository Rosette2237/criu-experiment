#!/usr/bin/env python3
#
# Parse raw CRIU/CUDA experiment result directories into a single CSV file.
#
# Example:
#
#   python3 tools/parse_results.py \
#       --raw-dir results/raw \
#       --output-csv results/parsed/summary.csv

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CSV_COLUMNS = [
    "run_id",
    "run_type",
    "status",
    "matrix_size",
    "iterations",
    "completed_iterations",
    "block_size",
    "device_id",
    "sleep_ms_between_iterations",
    "verify",
    "verify_samples",
    "passed",
    "exit_code",
    "gpu_memory_allocated_bytes",
    "host_memory_allocated_bytes",
    "matrix_bytes",
    "total_program_ms",
    "kernel_total_ms",
    "cuda_init_ms",
    "device_allocation_ms",
    "h2d_copy_ms",
    "d2h_copy_ms",
    "verification_ms",
    "cuda_program_total_wall_ms",
    "dump_wall_ms",
    "restore_wall_ms",
    "cuda_suspend_wall_ms",
    "cuda_resume_wall_ms",
    "checkpoint_image_bytes",
    "checkpoint_image_human",
    "dump_exit_code",
    "restore_exit_code",
    "cuda_suspend_exit_code",
    "cuda_resume_exit_code",
    "timestamp_utc",
    "hostname",
    "kernel",
    "os_pretty_name",
    "nvidia_driver_version",
    "cuda_toolkit_version",
    "criu_version",
    "cuda_checkpoint_version",
    "cuda_visible_devices",
    "selected_cuda_device",
    "cuda_checkpoint_mode",
    "run_dir",
]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}
    except Exception:
        return {}


def read_key_value_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}

    if not path.exists():
        return data

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()
    except Exception:
        return {}

    return data


def nested_get(data: Dict[str, Any], path: str, default: Any = "") -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def stringify(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def infer_run_type(run_dir: Path, cuda_json: Dict[str, Any], cpu_json: Dict[str, Any]) -> str:
    name = run_dir.name

    if "cuda_criu" in name:
        return "cuda_criu_checkpoint_restore"

    if "cuda_baseline" in name:
        return "cuda_baseline"

    if "cpu_criu" in name:
        return "cpu_criu_baseline"

    if cuda_json:
        if (run_dir / "criu_dump.log").exists() or (run_dir / "criu_restore.log").exists():
            return "cuda_criu_checkpoint_restore"
        return "cuda_baseline"

    if cpu_json:
        return "cpu_criu_baseline"

    if "env_check" in name:
        return "env_check"

    if "matrix_sweep" in name:
        return "matrix_sweep"

    return "unknown"


def infer_status(
    run_type: str,
    program_json: Dict[str, Any],
    timing: Dict[str, str],
    checkpoint_size: Dict[str, str],
) -> str:
    if run_type in {"env_check", "matrix_sweep"}:
        return "metadata"

    if not program_json:
        return "missing_program_json"

    passed = program_json.get("passed", False)
    exit_code = program_json.get("exit_code", "")

    if passed is True and str(exit_code) == "0":
        if run_type == "cuda_criu_checkpoint_restore":
            dump_exit = timing.get("DUMP_EXIT_CODE", "")
            restore_exit = timing.get("RESTORE_EXIT_CODE", "")

            if dump_exit not in {"", "0"}:
                return "dump_failed"

            if restore_exit not in {"", "0"}:
                return "restore_failed"

        return "success"

    return "failed"


def parse_one_run(run_dir: Path) -> Optional[Dict[str, str]]:
    env_json = read_json(run_dir / "env.json")

    cuda_baseline_json = read_json(run_dir / "cuda_baseline.json")
    checkpoint_restore_json = read_json(run_dir / "checkpoint_restore.json")
    cpu_baseline_json = read_json(run_dir / "cpu_criu_baseline.json")

    timing = read_key_value_file(run_dir / "checkpoint_timing.env")
    checkpoint_size = read_key_value_file(run_dir / "checkpoint_size.txt")

    if checkpoint_restore_json:
        program_json = checkpoint_restore_json
    elif cuda_baseline_json:
        program_json = cuda_baseline_json
    elif cpu_baseline_json:
        program_json = cpu_baseline_json
    else:
        program_json = {}

    run_type = infer_run_type(run_dir, cuda_baseline_json or checkpoint_restore_json, cpu_baseline_json)
    status = infer_status(run_type, program_json, timing, checkpoint_size)

    if run_type == "unknown" and not env_json and not program_json:
        return None

    experiment_config = env_json.get("experiment_config", {})
    if not isinstance(experiment_config, dict):
        experiment_config = {}

    row = {
        "run_id": stringify(env_json.get("run_id", run_dir.name)),
        "run_type": run_type,
        "status": status,
        "matrix_size": stringify(program_json.get("matrix_size", experiment_config.get("matrix_size", ""))),
        "iterations": stringify(program_json.get("iterations", experiment_config.get("iterations", ""))),
        "completed_iterations": stringify(program_json.get("completed_iterations", "")),
        "block_size": stringify(program_json.get("block_size", experiment_config.get("block_size", ""))),
        "device_id": stringify(program_json.get("device_id", experiment_config.get("selected_cuda_device", ""))),
        "sleep_ms_between_iterations": stringify(
            program_json.get(
                "sleep_ms_between_iterations",
                experiment_config.get("sleep_ms_between_iterations", ""),
            )
        ),
        "verify": stringify(program_json.get("verify", experiment_config.get("verify", ""))),
        "verify_samples": stringify(program_json.get("verify_samples", experiment_config.get("verify_samples", ""))),
        "passed": stringify(program_json.get("passed", "")),
        "exit_code": stringify(program_json.get("exit_code", "")),
        "gpu_memory_allocated_bytes": stringify(program_json.get("gpu_memory_allocated_bytes", "")),
        "host_memory_allocated_bytes": stringify(program_json.get("host_memory_allocated_bytes", "")),
        "matrix_bytes": stringify(program_json.get("matrix_bytes", "")),
        "total_program_ms": stringify(nested_get(program_json, "timing_ms.total_program_ms", "")),
        "kernel_total_ms": stringify(nested_get(program_json, "timing_ms.kernel_total_ms", "")),
        "cuda_init_ms": stringify(nested_get(program_json, "timing_ms.cuda_init_ms", "")),
        "device_allocation_ms": stringify(nested_get(program_json, "timing_ms.device_allocation_ms", "")),
        "h2d_copy_ms": stringify(nested_get(program_json, "timing_ms.h2d_copy_ms", "")),
        "d2h_copy_ms": stringify(nested_get(program_json, "timing_ms.d2h_copy_ms", "")),
        "verification_ms": stringify(nested_get(program_json, "timing_ms.verification_ms", "")),
        "cuda_program_total_wall_ms": stringify(timing.get("CUDA_PROGRAM_TOTAL_WALL_MS", "")),
        "dump_wall_ms": stringify(timing.get("DUMP_WALL_MS", "")),
        "restore_wall_ms": stringify(timing.get("RESTORE_WALL_MS", "")),
        "cuda_suspend_wall_ms": stringify(timing.get("CUDA_SUSPEND_WALL_MS", "")),
        "cuda_resume_wall_ms": stringify(timing.get("CUDA_RESUME_WALL_MS", "")),
        "checkpoint_image_bytes": stringify(checkpoint_size.get("CHECKPOINT_IMAGE_BYTES", "")),
        "checkpoint_image_human": stringify(checkpoint_size.get("CHECKPOINT_IMAGE_HUMAN", "")),
        "dump_exit_code": stringify(timing.get("DUMP_EXIT_CODE", "")),
        "restore_exit_code": stringify(timing.get("RESTORE_EXIT_CODE", "")),
        "cuda_suspend_exit_code": stringify(timing.get("CUDA_SUSPEND_EXIT_CODE", "")),
        "cuda_resume_exit_code": stringify(timing.get("CUDA_RESUME_EXIT_CODE", "")),
        "timestamp_utc": stringify(env_json.get("timestamp_utc", "")),
        "hostname": stringify(env_json.get("hostname", "")),
        "kernel": stringify(env_json.get("kernel", "")),
        "os_pretty_name": stringify(env_json.get("os_pretty_name", "")),
        "nvidia_driver_version": stringify(env_json.get("nvidia_driver_version", "")),
        "cuda_toolkit_version": stringify(env_json.get("cuda_toolkit_version", "")),
        "criu_version": stringify(env_json.get("criu_version", "")),
        "cuda_checkpoint_version": stringify(env_json.get("cuda_checkpoint_version", "")),
        "cuda_visible_devices": stringify(env_json.get("cuda_visible_devices", "")),
        "selected_cuda_device": stringify(env_json.get("selected_cuda_device", "")),
        "cuda_checkpoint_mode": stringify(experiment_config.get("cuda_checkpoint_mode", "")),
        "run_dir": str(run_dir),
    }

    return row


def iter_run_dirs(raw_dir: Path) -> Iterable[Path]:
    if not raw_dir.exists():
        return []

    return sorted(path for path in raw_dir.iterdir() if path.is_dir())


def write_csv(rows: Iterable[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CRIU/CUDA experiment result directories.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw run directories.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_csv = Path(args.output_csv)

    rows = []

    for run_dir in iter_run_dirs(raw_dir):
        row = parse_one_run(run_dir)

        if row is not None:
            rows.append(row)

    write_csv(rows, output_csv)

    print(f"Parsed {len(rows)} run directories")
    print(f"Wrote CSV: {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())