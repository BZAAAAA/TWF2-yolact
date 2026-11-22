#!/usr/bin/env python3
"""Train one configuration on one fold (or on a fixed split).

Example
-------
    python train.py --config configs/exp4_full.yaml --fold 0 \
        --out runs/full_f0 --epochs 100

Writes ``<out>/metrics.json`` (six metrics, mean IoU-free timing, params) and
``<out>/last.pth``.  Safe to interrupt: a ``<out>/done.json`` marker makes the
run skippable by ``run_all``.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from twf2yolact.data import ToothSegDataset, collate
from twf2yolact.engine import build_model, evaluate_model, train_one_epoch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None, help="fold index in folds.json")
    ap.add_argument("--list-file", default=None, help="fixed split: one list file per set")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or cfg["train"]["epochs"])
    size = int(cfg["train"]["input_size"])

    # ----- data ---------------------------------------------------------- #
    if args.fold is not None:
        folds = json.loads(Path(cfg["data"]["folds"]).read_text())
        tr, va = folds["folds"][args.fold]["train"], folds["folds"][args.fold]["val"]
        tr_list, va_list = out / f"train_f{args.fold}.txt", out / f"val_f{args.fold}.txt"
        tr_list.write_text("\n".join(tr))
        va_list.write_text("\n".join(va))
    else:
        tr_list = Path(cfg["data"]["train_list"])
        va_list = Path(cfg["data"]["val_list"])

    tr_ds = ToothSegDataset(tr_list, size, train=True)
    va_ds = ToothSegDataset(va_list, size, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=int(cfg["train"]["batch"]), shuffle=True,
                       num_workers=int(cfg["train"].get("workers", 4)),
                       collate_fn=collate, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=1, shuffle=False,
                       num_workers=int(cfg["train"].get("workers", 4)),
                       collate_fn=collate)

    # ----- model --------------------------------------------------------- #
    model = build_model(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=float(cfg["train"]["lr"]),
                          momentum=float(cfg["train"].get("momentum", 0.9)),
                          weight_decay=float(cfg["train"].get("weight_decay", 5e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    t0 = time.time()
    best = -1.0
    for epoch in range(epochs):
        loss = train_one_epoch(model, tr_dl, opt, args.device, cfg, epoch)
        sched.step()
        if (epoch + 1) % int(cfg["train"].get("eval_every", 10)) == 0 or epoch + 1 == epochs:
            metrics = evaluate_model(model, va_dl, args.device, cfg)
            print(f"[epoch {epoch}] loss {loss:.4f} " +
                  " ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
            if metrics["mAPseg"] > best:
                best = metrics["mAPseg"]
                torch.save({"model": model.state_dict(), "cfg": cfg}, out / "best.pth")

    metrics = evaluate_model(model, va_dl, args.device, cfg)
    result = {
        "config": args.config, "fold": args.fold, "seed": seed, "epochs": epochs,
        "input_size": size, "params": n_params,
        "wall_time_s": time.time() - t0, "metrics": metrics,
        "variant": cfg.get("model", {}).get("variant", "full"),
        "eca_dirs": cfg.get("model", {}).get("eca_dirs", ["h", "w"]),
        "ewt_mode": cfg.get("model", {}).get("ewt_mode", "concat"),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "done.json").write_text(json.dumps({"status": "ok"}))
    torch.save({"model": model.state_dict(), "cfg": cfg}, out / "last.pth")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
