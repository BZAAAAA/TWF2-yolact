"""Shared engine: model builder, training epoch, evaluation loop."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from .metrics import Evaluator
from .model import TWF2YOLACT, make_priors, postprocess


def build_model(cfg: dict, variant: Optional[str] = None) -> TWF2YOLACT:
    m = cfg.get("model", cfg)
    return TWF2YOLACT(
        num_classes=int(m.get("num_classes", 1)),
        mask_dim=int(m.get("mask_dim", 32)),
        eca_dirs=tuple(m.get("eca_dirs", ("h", "w"))),
        ewt_mode=m.get("ewt_mode", "concat"),
        variant=variant or m.get("variant", "full"),
        pretrained=bool(m.get("pretrained", False)),
        conf_thresh=float(m.get("conf_thresh", 0.05)),
        nms_thresh=float(m.get("nms_thresh", 0.6)),
        max_dets=int(m.get("max_dets", 100)),
    )


def train_one_epoch(model, loader, optimizer, device, cfg, epoch: int, log_every: int = 20):
    from .losses import yolact_loss

    model.train()
    size = int(cfg.get("train", {}).get("input_size", 550))
    priors = None
    running, n = 0.0, 0
    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        out = model(images)
        if priors is None:
            # derive the anchor grid from the real feature-map shapes instead of
            # guessing size/8, size/16, ... (not integer for sizes like 550)
            priors = make_priors(out["levels"], (size, size), device)
        losses = yolact_loss(out, batch, priors, (size, size))
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        running += float(loss.detach())
        n += 1
        if log_every and step % log_every == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss {float(loss.detach()):.4f} "
                  f"(box {float(losses['box']):.3f} cls {float(losses['cls']):.3f} "
                  f"mask {float(losses['mask']):.3f}) pos {losses['num_pos']}")
    return running / max(n, 1)


@torch.no_grad()
def evaluate_model(model, loader, device, cfg, max_images: Optional[int] = None):
    """Return the six metrics of the manuscript on ``loader``."""
    model.eval()
    ev = Evaluator(max_dets=int(cfg.get("model", cfg).get("max_dets", 100)))
    size = int(cfg.get("train", {}).get("input_size", 550))
    seen = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        out = model(images)
        for i in range(images.shape[0]):
            single = {k: (v[i:i + 1] if torch.is_tensor(v) else v)
                      for k, v in out.items() if k != "levels"}
            single["levels"] = out["levels"]
            boxes, scores, classes, masks = postprocess(single, (size, size), cfg["model"])
            ev.add(boxes.numpy(), scores.numpy(), masks.numpy(),
                   batch["boxes"][i].numpy(), batch["masks"][i].numpy())
            seen += 1
            if max_images and seen >= max_images:
                return ev.compute()
    return ev.compute()
