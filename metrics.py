"""COCO-style evaluation for the six metrics used in the paper.

Reported per model (identical to the table headers of the manuscript):
    APdet   @ IoU 0.75            APseg   @ IoU 0.75
    ARdet   @ IoU 0.50:0.95       ARseg   @ IoU 0.50:0.95   (maxDets = 100)
    mAPdet  @ IoU 0.50:0.95       mAPseg  @ IoU 0.50:0.95

The implementation is deliberately dependency-free (no pycocotools) so that the
metric definition is auditable.  IMPORTANT: numbers are only comparable with the
numbers already published if this file implements the same definitions as the
script that produced them; verify on one fold before trusting the output.
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _iou_matrix(dets: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """dets/gts: (N,4) xyxy -> (N,M) IoU."""
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)))
    lt = np.maximum(dets[:, None, :2], gts[None, :, :2])
    rb = np.minimum(dets[:, None, 2:], gts[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_d = np.clip(dets[:, 2] - dets[:, 0], 0, None) * np.clip(dets[:, 3] - dets[:, 1], 0, None)
    area_g = np.clip(gts[:, 2] - gts[:, 0], 0, None) * np.clip(gts[:, 3] - gts[:, 1], 0, None)
    union = area_d[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _mask_iou(det_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(det_mask, gt_mask).sum()
    union = np.logical_or(det_mask, gt_mask).sum()
    return float(inter / union) if union else 0.0


def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP."""
    if recall.size == 0:
        return 0.0
    r = np.linspace(0, 1, 101)
    p = np.zeros_like(r)
    for i, t in enumerate(r):
        mask = recall >= t
        p[i] = precision[mask].max() if mask.any() else 0.0
    return float(p.mean())


class Evaluator:
    """Accumulate detections/GT per image, then compute the six metrics."""

    def __init__(self, max_dets: int = 100):
        self.max_dets = max_dets
        self.records = []          # (det_boxes, det_scores, det_masks, gt_boxes, gt_masks)

    def add(self, det_boxes, det_scores, det_masks, gt_boxes, gt_masks):
        order = np.argsort(-np.asarray(det_scores))[: self.max_dets]
        self.records.append((
            np.asarray(det_boxes, dtype=np.float32)[order],
            np.asarray(det_scores, dtype=np.float32)[order],
            np.asarray(det_masks)[order] if len(order) else np.zeros((0, 1, 1)),
            np.asarray(gt_boxes, dtype=np.float32),
            np.asarray(gt_masks),
        ))

    # ------------------------------------------------------------------ #
    def _accumulate(self, use_masks: bool):
        """Return precision/recall curves per IoU threshold."""
        curves = {t: {"tp": [], "fp": [], "scores": [], "n_gt": 0} for t in IOU_THRESHOLDS}
        for det_b, det_s, det_m, gt_b, gt_m in self.records:
            n_gt = len(gt_b)
            for t in IOU_THRESHOLDS:
                c = curves[t]
                c["n_gt"] += n_gt
                if n_gt == 0:
                    # every detection is a false positive; keep tp/fp aligned
                    c["fp"].extend([1] * len(det_s))
                    c["tp"].extend([0] * len(det_s))
                    c["scores"].extend(det_s.tolist())
                    continue
                if use_masks:
                    iou = np.zeros((len(det_m), n_gt))
                    for i in range(len(det_m)):
                        for j in range(n_gt):
                            iou[i, j] = _mask_iou(det_m[i] > 0.5, gt_m[j] > 0.5)
                else:
                    iou = _iou_matrix(det_b, gt_b)
                used = np.zeros(n_gt, dtype=bool)
                for i in range(len(det_b)):
                    c["scores"].append(float(det_s[i]))
                    j = int(np.argmax(iou[i])) if n_gt else -1
                    if n_gt and iou[i, j] >= t and not used[j]:
                        used[j] = True
                        c["tp"].append(1)
                        c["fp"].append(0)
                    else:
                        c["tp"].append(0)
                        c["fp"].append(1)
        out = {}
        for t, c in curves.items():
            if not c["scores"]:
                out[float(t)] = (np.array([]), np.array([]), 0)
                continue
            order = np.argsort(-np.asarray(c["scores"]))
            tp = np.asarray(c["tp"])[order].cumsum()
            fp = np.asarray(c["fp"])[order].cumsum()
            recall = tp / max(c["n_gt"], 1)
            precision = tp / np.maximum(tp + fp, 1e-9)
            out[float(t)] = (recall, precision, c["n_gt"])
        return out

    def _metrics(self, use_masks: bool):
        curves = self._accumulate(use_masks)
        aps = {t: _ap_101(*v[:2]) for t, v in curves.items()}
        ars = {}
        for t, (recall, _, _) in curves.items():
            ars[t] = float(recall[-1]) if recall.size else 0.0
        return aps, ars

    def compute(self):
        ap_d, ar_d = self._metrics(use_masks=False)
        ap_s, ar_s = self._metrics(use_masks=True)
        mean = lambda d: float(np.mean(list(d.values())))
        return {
            "APdet": ap_d[0.75], "APseg": ap_s[0.75],
            "ARdet": float(np.mean(list(ar_d.values()))),
            "ARseg": float(np.mean(list(ar_s.values()))),
            "mAPdet": mean(ap_d), "mAPseg": mean(ap_s),
        }
