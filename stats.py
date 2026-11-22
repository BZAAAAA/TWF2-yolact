#!/usr/bin/env python3
"""Aggregate fold results: mean +/- std and paired significance tests.

    python stats.py --runs runs/ --group-by variant

Reads every ``runs/**/metrics.json`` and prints a markdown table with
mean +/- sample std (ddof=1) per configuration, plus a paired comparison
between two configurations matched by fold index.

Note on n=5: the exact two-sided Wilcoxon signed-rank test cannot go below
p = 0.0625 with five pairs.  Text claims must therefore read e.g.
"p = 0.0625 (exact, n = 5)" and never "p < 0.05".
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["APdet", "APseg", "ARdet", "ARseg", "mAPdet", "mAPseg"]


def load_runs(root: Path):
    runs = []
    for p in root.rglob("metrics.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        d["path"] = str(p)
        runs.append(d)
    return runs


def group_key(run, mode):
    if mode == "variant":
        return run.get("variant", "unknown")
    if mode == "config":
        return Path(run.get("config", "?")).stem
    return run.get("path", "?")


def exact_wilcoxon(a, b):
    """Two-sided exact signed-rank p-value by full enumeration (n <= 12)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    w_obs = np.abs(np.sum(ranks[d > 0]))
    total = 0
    hits = 0
    for signs in itertools.product([-1, 1], repeat=n):
        w = abs(np.sum(ranks[np.asarray(signs) > 0]))
        total += 1
        hits += int(w >= w_obs - 1e-9)
    return hits / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--group-by", default="variant", choices=["variant", "config", "path"])
    ap.add_argument("--compare", nargs=2, default=None,
                    help="two group keys to test (paired by fold)")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs))
    if not runs:
        print("no metrics.json found")
        return 1

    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r, args.group_by)].append(r)

    print("| configuration | n | " + " | ".join(METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for key in sorted(groups):
        rs = groups[key]
        cells = []
        for m in METRICS:
            vals = [r["metrics"][m] for r in rs if m in r.get("metrics", {})]
            cells.append(f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}" if len(vals) > 1
                         else (f"{vals[0]:.2f}" if vals else "-"))
        print(f"| {key} | {len(rs)} | " + " | ".join(cells) + " |")

    if args.compare:
        a_key, b_key = args.compare
        by_fold_a = {r.get("fold"): r for r in groups.get(a_key, [])}
        by_fold_b = {r.get("fold"): r for r in groups.get(b_key, [])}
        shared = sorted(set(by_fold_a) & set(by_fold_b), key=lambda x: (x is None, x))
        print(f"\npaired comparison {a_key} vs {b_key} on {len(shared)} folds")
        for m in METRICS:
            a = [by_fold_a[f]["metrics"][m] for f in shared]
            b = [by_fold_b[f]["metrics"][m] for f in shared]
            if len(a) < 3:
                continue
            print(f"  {m}: Δ = {np.mean(a) - np.mean(b):+.2f}  exact p = {exact_wilcoxon(a, b):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
