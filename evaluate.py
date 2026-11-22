#!/usr/bin/env python3
"""Evaluate a checkpoint on one split and print the six manuscript metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from twf2yolact.data import ToothSegDataset, collate
from twf2yolact.engine import build_model, evaluate_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--list-file", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = build_model(cfg).to(args.device)
    ckpt = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    ds = ToothSegDataset(args.list_file, int(cfg["train"]["input_size"]), train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    metrics = evaluate_model(model, dl, args.device, cfg)
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
