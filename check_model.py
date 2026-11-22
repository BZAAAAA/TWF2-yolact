# -*- coding: utf-8 -*-
"""Parameter/forward sanity check for the ablation variants of the manuscript.

    D:\\anaconda3\\envs\\yolact\\python.exe check_model.py

Prints the parameter count of each variant together with the value reported in
Table 2 of the manuscript, and runs one forward pass so that shape errors show
up here rather than during training.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from twf2yolact.model import TWF2YOLACT

# Table 2, Params (M): Exp.1-Exp.4
CASES = [
    ("Exp.1 yolact",   dict(variant="yolact"),    550, 33.39),
    ("Exp.2 dfem",     dict(variant="dfem"),      550, 44.47),
    ("Exp.3 +fsd",     dict(variant="dfem_fsd"),  550, 47.29),
    ("Exp.4 +mfiem",   dict(variant="full"),      550, 49.60),
    # Table 6 Exp.3: inverse reconstruction instead of concatenation
    ("Table6 iwt",     dict(variant="dfem_fsd", ewt_mode="iwt"), 550, 38.99),
    # BraTS runs at 3x240x240
    ("BraTS 240",      dict(variant="full"),      240, 49.60),
]


def main() -> None:
    print("%-16s %10s %8s %7s  %s" % ("case", "params(M)", "table(M)", "diff", "shapes"))
    for tag, kwargs, hw, expected in CASES:
        model = TWF2YOLACT(**kwargs).eval()
        n = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            out = model(torch.randn(1, 3, hw, hw))
        shapes = "cls%s coeff%s proto%s" % (
            tuple(out["cls"].shape), tuple(out["coeff"].shape), tuple(out["proto"].shape))
        print("%-16s %10.2f %8.2f %+7.2f  %s" % (tag, n, expected, n - expected, shapes))


if __name__ == "__main__":
    main()
