#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

print("== checkpoint metrics ==")
for name in ["cpu_criu_metrics.csv", "gpu_criugpu_metrics.csv"]:
    path = RESULTS / name
    rows = read_csv(path)
    if not rows:
        print(f"{name}: missing")
        continue

    print(f"\n{name}")
    for r in rows:
        print(
            f"run_id={r['run_id']} mode={r['mode']} gpus={r['gpus']} "
            f"dump_s={r['dump_wall_s']} restore_s={r['restore_wall_s']} "
            f"ckpt_GB={int(r['checkpoint_bytes']) / (1024**3):.3f} "
            f"app_csv={r['app_csv']}"
        )

print("\n== app progress summaries ==")
for path in sorted(RESULTS.glob("*_app.csv")):
    rows = read_csv(path)
    if not rows:
        continue

    groups = defaultdict(list)
    for r in rows:
        key = (r.get("mode", ""), r.get("gpu", "cpu"))
        groups[key].append(r)

    print(f"\n{path.name}")
    for key, vals in sorted(groups.items()):
        last = vals[-1]
        mode, gpu = key
        print(
            f"mode={mode} gpu={gpu} rows={len(vals)} "
            f"last_iter={last.get('iter')} last_wall_s={last.get('wall_s')} "
            f"last_checksum={last.get('checksum')}"
        )