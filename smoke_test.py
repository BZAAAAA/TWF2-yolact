#!/usr/bin/env python3
"""CPU smoke test — run this first, before any GPU job.

Checks
  1. Haar DWT/IWT round-trip
  2. all four ablation variants forward without shape errors
  3. head/prior/mask bookkeeping is consistent (postprocess runs)
  4. ECA does not change the tensor shape
No dataset and no GPU are required.
"""
from __future__ import annotations

import sys

import torch

from twf2yolact.model import (ECA, EWTBlock, TWF2YOLACT, haar_dwt2, haar_iwt2,
                              make_priors, postprocess)


def check_dwt():
    x = torch.randn(2, 8, 32, 32)
    A, Dh, Dw, Dd = haar_dwt2(x)
    y = haar_iwt2(A, Dh, Dw, Dd, (32, 32))
    err = (x - y).abs().max().item()
    assert err < 1e-5, f"DWT/IWT round-trip error {err}"
    print(f"  dwt round-trip ok (max err {err:.2e})")


def check_eca():
    x = torch.randn(2, 16, 8, 8)
    assert ECA(16)(x).shape == x.shape
    e = EWTBlock(16, 32, ("h", "w"), "concat")(x)
    assert e.shape == (2, 32, 8, 8), e.shape
    e2 = EWTBlock(16, 32, ("h", "w"), "iwt")(x)
    assert e2.shape == (2, 32, 8, 8), e2.shape
    print("  eca / ewt_block (concat, iwt) shape ok")


def check_forward(size=256):
    x = torch.randn(1, 3, size, size)
    for variant in ("yolact", "dfem", "dfem_fsd", "full"):
        model = TWF2YOLACT(variant=variant, mask_dim=32, eca_dirs=("h", "w"))
        model.eval()
        with torch.no_grad():
            out = model(x)
        n_anchors = out["cls"].shape[1]
        priors = make_priors(out["levels"], (size, size), out["cls"].device)
        assert priors.shape[0] == n_anchors, (priors.shape, n_anchors)
        assert out["coeff"].shape[:2] == (1, n_anchors)
        boxes, scores, classes, masks = postprocess(out, (size, size), model.cfg)
        assert masks.shape[0] == boxes.shape[0]
        print(f"  {variant:9s} ok  anchors={n_anchors}  dets={boxes.shape[0]}")


def main() -> int:
    print("1) DWT/IWT")
    check_dwt()
    print("2) modules")
    check_eca()
    print("3) forward for all variants (input 256x256)")
    check_forward()
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
