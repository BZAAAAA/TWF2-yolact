#!/usr/bin/env python3
"""Queue every (config, fold) pair, skipping runs that already finished.

    python run_all.py --configs configs/*.yaml --folds 5 --out runs

Each run writes ``runs/<config>_f<fold>/metrics.json`` and ``done.json``; an
interrupted job is resumed by re-running the same command.  This is the script
that keeps 40 unattended GPU-hours from being lost to a single crash.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    for cfg in args.configs:
        stem = Path(cfg).stem
        for fold in range(args.folds):
            for seed in args.seeds:
                out = Path(args.out) / f"{stem}_f{fold}_s{seed}"
                if (out / "done.json").exists():
                    print(f"skip   {out}")
                    continue
                jobs.append((cfg, fold, seed, out))

    print(f"\n{len(jobs)} job(s) to run")
    for cfg, fold, seed, out in jobs:
        cmd = [args.python, "train.py", "--config", cfg, "--fold", str(fold),
               "--seed", str(seed), "--out", str(out)]
        print("run   ", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! job failed (rc={rc}): {out} -- stopping so the error is visible")
            return rc
    print("all jobs finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
