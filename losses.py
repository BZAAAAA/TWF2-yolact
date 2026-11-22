"""YOLACT-style training losses (box / class / mask)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def match_priors(priors, gt_boxes, pos_thr=0.5):
    """Return (labels, matched_idx) for one image; -1 means background."""
    n = priors.shape[0]
    if gt_boxes.numel() == 0:
        return torch.zeros(n, dtype=torch.long), torch.full((n,), -1, dtype=torch.long)
    iou = box_iou(priors, gt_boxes)                      # (N, G)
    best_iou, best_idx = iou.max(dim=1)
    labels = (best_iou >= pos_thr).long()
    matched = best_idx.clone()
    matched[labels == 0] = -1
    # every GT keeps at least its best prior
    for g in range(gt_boxes.shape[0]):
        j = iou[:, g].argmax()
        if labels[j] == 0:
            labels[j], matched[j] = 1, g
    return labels, matched


def hard_negative_mining(cls_prob, labels, neg_pos_ratio=3):
    pos = labels > 0
    num_pos = int(pos.sum())
    num_neg = min(int((~pos).sum()), neg_pos_ratio * max(num_pos, 1))
    loss_neg, idx_neg = cls_prob[~pos].sort(descending=True)
    keep = torch.zeros_like(labels, dtype=torch.bool)
    keep[pos] = True
    neg_indices = torch.where(~pos)[0][idx_neg[:num_neg]]
    keep[neg_indices] = True
    return keep, num_pos


def mask_targets(gt_masks, gt_boxes, img_hw):
    """Crop GT masks to their boxes and flatten to (G, h*w) at full resolution."""
    out = []
    for m, b in zip(gt_masks, gt_boxes):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_hw[1], x2), min(img_hw[0], y2)
        crop = m[y1:y2, x1:x2]
        if crop.numel() == 0:
            crop = torch.zeros(1, 1, device=m.device)
        out.append(crop.reshape(-1))
    return out


def yolact_loss(out, batch, priors, img_hw, neg_pos_ratio=3, w_box=1.5,
                w_cls=1.0, w_mask=1.0):
    """Total loss for a batch.  ``priors`` are the shared anchor boxes."""
    from .model import decode_boxes

    cls_logits, raw_box, coeff = out["cls"], out["box"], out["coeff"]
    proto = out["proto"]
    b = cls_logits.shape[0]
    device = cls_logits.device

    total_box = total_cls = total_mask = 0.0
    n_pos_total = 0

    for i in range(b):
        gt_boxes = batch["boxes"][i].to(device)
        gt_masks = batch["masks"][i].to(device)
        labels, matched = match_priors(priors, gt_boxes)
        pos = labels > 0
        n_pos = int(pos.sum())
        n_pos_total += n_pos

        cls_prob = torch.sigmoid(cls_logits[i]).squeeze(-1)
        keep, _ = hard_negative_mining(cls_prob, labels, neg_pos_ratio)
        target = labels.float().clamp(0, 1)
        total_cls = total_cls + F.binary_cross_entropy(
            cls_prob[keep].clamp(1e-4, 1 - 1e-4), target[keep], reduction="sum")

        if n_pos > 0:
            # decode all anchors, then keep the positive ones: priors are
            # indexed by anchor, so filtering first would misalign the two
            pbox = decode_boxes(raw_box[i], priors)[pos]
            gbox = gt_boxes[matched[pos]]
            total_box = total_box + F.smooth_l1_loss(pbox, gbox, reduction="sum", beta=0.1)

            hp, wp = proto.shape[-2:]
            c = coeff[i].reshape(-1, coeff.shape[-1])[pos]     # (P, K)
            p = proto[i].reshape(proto.shape[1], -1)          # (K, h*w)
            pred = torch.sigmoid(c @ p).reshape(n_pos, hp, wp)
            pred = F.interpolate(pred.unsqueeze(1), size=img_hw, mode="bilinear",
                                 align_corners=False).squeeze(1)
            # full-resolution GT masks of the matched instances (no cropping, so
            # the target always has the same shape as the prediction)
            target_pix = gt_masks[matched[pos]].float()
            total_mask = total_mask + F.binary_cross_entropy(
                pred.clamp(1e-4, 1 - 1e-4), target_pix, reduction="sum")

    n = max(n_pos_total, 1)
    loss_cls = total_cls / n
    loss_box = total_box / n
    loss_mask = total_mask / n
    return {"loss": w_box * loss_box + w_cls * loss_cls + w_mask * loss_mask,
            "box": loss_box.detach() if torch.is_tensor(loss_box) else loss_box,
            "cls": loss_cls.detach() if torch.is_tensor(loss_cls) else loss_cls,
            "mask": loss_mask.detach() if torch.is_tensor(loss_mask) else loss_mask,
            "num_pos": n_pos_total}
