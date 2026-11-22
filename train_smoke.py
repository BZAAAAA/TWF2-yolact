# -*- coding: utf-8 -*-
"""End-to-end training/evaluation check on synthetic data (no dataset needed).

    D:\\anaconda3\\envs\\yolact\\python.exe train_smoke.py

Runs one optimisation step and one evaluation pass for the full model at a small
input size, so that the loss, the backward pass and the metric code can be
verified before a real run is launched.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from twf2yolact.engine import build_model, evaluate_model, train_one_epoch

SIZE = 256
N_IMG = 2


class ToyDataset(Dataset):
    """Random images with 1-2 rectangular instances, in the collate format."""

    def __init__(self, n=N_IMG, size=SIZE, seed=0):
        self.n, self.size = n, size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        r = np.random.RandomState(idx)
        img = torch.from_numpy(r.rand(3, self.size, self.size).astype(np.float32))
        masks, boxes = [], []
        for _ in range(r.randint(1, 3)):
            x0, y0 = r.randint(0, self.size // 2), r.randint(0, self.size // 2)
            w, h = r.randint(32, self.size // 3), r.randint(32, self.size // 3)
            m = np.zeros((self.size, self.size), np.float32)
            m[y0:y0 + h, x0:x0 + w] = 1.0
            masks.append(m)
            boxes.append([x0, y0, x0 + w, y0 + h])
        return {
            "image": img,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.zeros(len(boxes), dtype=torch.long),
            "masks": torch.from_numpy(np.stack(masks)),
            "orig_size": torch.tensor([self.size, self.size]),
            "index": idx,
        }


def collate(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "masks": [b["masks"] for b in batch],
        "orig_sizes": [b["orig_size"] for b in batch],
        "indices": [b["index"] for b in batch],
    }


def main() -> int:
    torch.manual_seed(0)
    cfg = {
        "train": {"input_size": SIZE, "batch": N_IMG, "lr": 1e-3,
                  "momentum": 0.9, "weight_decay": 5e-4, "workers": 0},
        "model": {"num_classes": 1, "mask_dim": 32, "pretrained": False,
                  "conf_thresh": 0.05, "nms_thresh": 0.6, "max_dets": 100,
                  "variant": "full"},
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = build_model(cfg).to(device)
    loader = DataLoader(ToyDataset(), batch_size=N_IMG, collate_fn=collate)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"],
                          momentum=0.9, weight_decay=5e-4)

    t0 = time.time()
    loss = train_one_epoch(model, loader, opt, device, cfg, epoch=0, log_every=0)
    print(f"one epoch ok  mean loss {loss:.4f}  ({time.time() - t0:.1f}s)")

    metrics = evaluate_model(model, loader, device, cfg)
    print("eval ok:", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in metrics.items()})
    print("TRAIN SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
