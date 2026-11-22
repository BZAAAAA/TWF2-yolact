#!/usr/bin/env python3
"""Build deterministic K-fold splits for the tooth dataset.

    python make_folds.py --list-file data/list.txt --k 5 --seed 42 --out data/folds.json

``list.txt`` holds one ``<image_path> <labelme_json>`` per line.  Splits are
stratified by the number of annotated instances per image so that folds are
comparable; the same file is used by every model and is the file to publish
together with the code (Reviewer 1, point 6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def instance_count(json_path: str) -> int:
    try:
        d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return 0
    return sum(1 for s in d.get("shapes", []) if len(s.get("points") or []) >= 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-file", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = [ln.strip() for ln in Path(args.list_file).read_text().splitlines() if ln.strip()]
    rng = random.Random(args.seed)

    # stratify into 3 bins by instance count (0-1, 2-3, >=4)
    strata: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for ln in lines:
        parts = ln.split()
        n = instance_count(parts[1]) if len(parts) > 1 else 0
        strata[0 if n <= 1 else (1 if n <= 3 else 2)].append(ln)

    folds = [{"train": [], "val": []} for _ in range(args.k)]
    for _bin, items in strata.items():
        rng.shuffle(items)
        for i, item in enumerate(items):
            folds[i % args.k]["val"].append(item)
    for f in folds:
        val_set = set(f["val"])
        f["train"] = [ln for ln in lines if ln not in val_set]
        assert not (set(f["train"]) & set(f["val"])), "fold leakage"

    payload = {
        "source": args.list_file,
        "k": args.k,
        "seed": args.seed,
        "n_images": len(lines),
        "fingerprint": hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:16],
        "folds": folds,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: k={args.k} n={len(lines)} "
          f"vals={[len(f['val']) for f in folds]} fingerprint={payload['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
