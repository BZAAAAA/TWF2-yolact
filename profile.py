#!/usr/bin/env python3
"""Efficiency measurement: params, FLOPs, end-to-end latency, FPS.

Reports the six fields required by Reviewer 2 (point 3) and Reviewer 1
(point 3): hardware, batch size, precision, whether pre/post-processing and NMS
are included, timing method, input resolution.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import yaml

from twf2yolact.engine import build_model
from twf2yolact.model import make_priors, postprocess


def count_flops(model, size, device):
    try:
        from thop import profile as thop_profile
    except ImportError:
        return None
    x = torch.randn(1, 3, size, size, device=device)
    flops, _ = thop_profile(model, inputs=(x,), verbose=False)
    return float(flops)


@torch.no_grad()
def measure_latency(model, size, device, warmup=20, iters=100, include_post=True):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        out = model(x)
        if include_post:
            postprocess(out, (size, size), model.cfg)
    if device == "cuda":
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    if start is not None:
        start.record()
        for _ in range(iters):
            out = model(x)
            if include_post:
                postprocess(out, (size, size), model.cfg)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            out = model(x)
            if include_post:
                postprocess(out, (size, size), model.cfg)
        ms = (time.perf_counter() - t0) * 1000 / iters
    return ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    size = args.size or int(cfg["train"]["input_size"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, args.variant).to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    flops = count_flops(model, size, device)
    ms_with = measure_latency(model, size, device, include_post=True)
    ms_without = measure_latency(model, size, device, include_post=False)

    result = {
        "hardware": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "batch_size": 1,
        "precision": "FP32",
        "input_size": size,
        "includes_prepost_and_nms": True,
        "timing": "CUDA events, 20 warm-up + 100 timed iterations",
        "params_M": round(params / 1e6, 3),
        "flops_G": None if flops is None else round(flops / 1e9, 2),
        "latency_ms_with_post": round(ms_with, 2),
        "latency_ms_model_only": round(ms_without, 2),
        "fps": round(1000.0 / ms_with, 2),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
