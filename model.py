"""TWF2-YOLACT: reference implementation.

Modules follow Section 3 of the manuscript:
  * EWT_block  -> 2D Haar DWT + ECA on Dh/Dw + sub-band recombination (Eq. 1-5)
  * FSD-AFM    -> adaptive frequency/spatial fusion (Eq. 6-8)
  * MFIEM      -> ECA + AF_i + As2M_i (= MSHM) multi-layer interaction (Eq. 9)

Every place where the manuscript leaves a gap is marked  # ASSUMPTION An
and listed in SPEC_GAPS.md.  Defaults follow our best reading of the text and
are configurable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class ConvBNReLU(nn.Module):
    """C{k}x{k}BR in the paper (conv -> BN -> ReLU)."""

    def __init__(self, cin: int, cout: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ECA(nn.Module):
    """Efficient Channel Attention (1D conv over the pooled channel vector)."""

    def __init__(self, channels: int, k: int = 5):
        super().__init__()
        k = max(3, int(k) | 1)                      # keep the kernel odd
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        b, c = x.shape[:2]
        y = x.mean(dim=(2, 3)).view(b, 1, c)
        y = torch.sigmoid(self.conv(y)).view(b, c, 1, 1)
        return x * y


def haar_dwt2(x: torch.Tensor):
    """Single-level orthonormal 2D Haar transform (Eqs. 1-4).

        A  = (a + b + c + d) / 2      low-low
        Dh = (a - b + c - d) / 2      horizontal detail
        Dw = (a + b - c - d) / 2      vertical detail
        Dd = (a - b - c + d) / 2      diagonal detail
    with a,b,c,d the 2x2 pixel groups.  Odd sizes are right/bottom padded.
    """
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    return ((a + b + c + d) * 0.5, (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5, (a - b - c + d) * 0.5)


def haar_iwt2(A, Dh, Dw, Dd, out_hw):
    """Inverse of :func:`haar_dwt2`, resampled back to ``out_hw`` if needed."""
    a = (A + Dh + Dw + Dd) * 0.5
    b = (A - Dh + Dw - Dd) * 0.5
    c = (A + Dh - Dw - Dd) * 0.5
    d = (A - Dh - Dw + Dd) * 0.5
    out = torch.empty(a.shape[0], a.shape[1], a.shape[2] * 2, a.shape[3] * 2,
                      dtype=a.dtype, device=a.device)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    if out.shape[-2:] != tuple(out_hw):
        out = F.interpolate(out, size=out_hw, mode="nearest")
    return out


class EWTBlock(nn.Module):
    """One stage of the frequency branch (Fig. 5, Eq. 5).

    mode='concat' -> Eq. (5): concatenate A, Dh', Dw', Dd then C1x1BR.
    mode='iwt'    -> Table 6 Exp.3: inverse Haar reconstruction instead.
    """

    def __init__(self, cin: int, cout: int, eca_dirs=("h", "w"),
                 mode: str = "concat", eca_k: int = 5):
        super().__init__()
        assert mode in ("concat", "iwt")
        self.mode = mode
        self.eca_h = ECA(cin, eca_k) if "h" in eca_dirs else None
        self.eca_w = ECA(cin, eca_k) if "w" in eca_dirs else None
        self.eca_d = ECA(cin, eca_k) if "d" in eca_dirs else None
        # ASSUMPTION A1: sub-bands are 2x smaller; the projection is upsampled by
        # 2 so that the frequency branch matches the spatial branch resolution.
        self.proj = ConvBNReLU(4 * cin if mode == "concat" else cin, cout, 1)

    def forward(self, x):
        hw = x.shape[-2:]
        A, Dh, Dw, Dd = haar_dwt2(x)
        if self.eca_h is not None:
            Dh = self.eca_h(Dh)
        if self.eca_w is not None:
            Dw = self.eca_w(Dw)
        if self.eca_d is not None:
            Dd = self.eca_d(Dd)
        if self.mode == "concat":
            y = self.proj(torch.cat([A, Dh, Dw, Dd], dim=1))
            return F.interpolate(y, size=hw, mode="nearest")
        return self.proj(haar_iwt2(A, Dh, Dw, Dd, hw))


class FSDAFM(nn.Module):
    """Frequency-Spatial Domain Adaptive Fusion Module (Fig. 6, Eqs. 6-8)."""

    def __init__(self, c_freq: int, c_spat: int, cout: int = 256):
        super().__init__()
        self.p_freq = ConvBNReLU(c_freq, cout, 1)
        self.p_spat = ConvBNReLU(c_spat, cout, 1)
        self.br1 = ConvBNReLU(2 * cout, cout, 3)
        self.br2 = ConvBNReLU(cout, cout, 3)
        self.w = nn.Conv2d(cout, 2, 1)

    def forward(self, freq, spat):
        fp = self.p_freq(freq)
        fk = self.p_spat(spat)
        fpk = self.br2(self.br1(torch.cat([fk, fp], dim=1)))   # Eq. (6)
        w = torch.softmax(self.w(fpk), dim=1)                  # Eq. (7)
        wp, wk = w[:, 0:1], w[:, 1:2]
        add = fp * wp + fk * wk
        mul = (fp * wp) * (fk * wk)
        return add + mul                                       # Eq. (8)


class AF(nn.Module):
    """AF_i, stage 2 of MFIEM (Eq. 9): soft-max weighted multi-scale mixing.

    ``factors`` are the resampling factors applied to (C3, C4, C5) before they
    are brought to the level's own resolution.
    """

    def __init__(self, cin: int = 256, factors=(2.0, 2.0, 1.0)):
        super().__init__()
        self.factors = factors
        self.g = nn.ModuleList([ConvBNReLU(cin, cin, 1) for _ in range(3)])
        self.mix = nn.Conv2d(3 * cin, 3, 1)

    def forward(self, c3, c4, c5, size):
        feats = []
        for f, s in zip((c3, c4, c5), self.factors):
            feats.append(F.interpolate(f, scale_factor=s, mode="bilinear",
                                       align_corners=False) if s != 1.0 else f)
        feats = [F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                 if f.shape[-2:] != tuple(size) else f for f in feats]
        gs = [g(f) for g, f in zip(self.g, feats)]
        w = torch.softmax(self.mix(torch.cat(gs, dim=1)), dim=1)
        # ASSUMPTION A4: Eq. (9) repeats one weight symbol (Gw) for all three
        # scales and the prose in Sec. 3.4 has an empty symbol; we use the three
        # soft-max channels as the per-scale weights.
        return sum(w[:, i:i + 1] * gs[i] for i in range(3))


class As2M(nn.Module):
    """As2M_i of the MSHM module (Fig. 8, stage 3)."""

    def __init__(self, cin: int = 256, cout: int = 256, matmul: bool = False):
        super().__init__()
        self.matmul = matmul
        self.proj_c = nn.Conv2d(cin, cout, 1, bias=False)
        self.proj_p = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn_c, self.bn_p = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)
        self.proj_xpc = nn.Conv2d(cout, cout, 1, bias=False)
        self.bn_xpc = nn.BatchNorm2d(cout)

    def forward(self, c, p):
        xc = self.bn_c(self.proj_c(c))
        xp = self.bn_p(self.proj_p(p))
        # ASSUMPTION A5: "matrix multiplied" is implemented as the Hadamard
        # product (set cfg.model.as2m_matmul=true for a true matmul).
        xpc = xc * xp
        xpc = self.bn_xpc(self.proj_xpc(xpc))
        s = torch.sigmoid(xpc)
        return s * xp + (1.0 - s) * xc


class MFIEM(nn.Module):
    """Multi-layer Feature Interaction Enhancement Module (Fig. 7)."""

    def __init__(self, c3=512, c4=1024, c5=2048, cout=256, matmul=False):
        super().__init__()
        self.eca = nn.ModuleList([ECA(c3), ECA(c4), ECA(c5)])
        self.proj = nn.ModuleList([nn.Conv2d(c3, cout, 1), nn.Conv2d(c4, cout, 1),
                                   nn.Conv2d(c5, cout, 1)])
        # ASSUMPTION A3: the paper says "upsample/downsample" with factors
        # 2/2/1, 1/2/1/2 and 1/4/1/2/1 for AF_1..AF_3 (Sec. 3.4, Stage 1).
        self.af = nn.ModuleList([AF(cout, (2.0, 2.0, 1.0)),
                                 AF(cout, (0.5, 1.0, 2.0)),
                                 AF(cout, (0.25, 0.5, 1.0))])
        self.as2m = nn.ModuleList([As2M(cout, cout, matmul) for _ in range(3)])
        self.reduce = ConvBNReLU(3 * cout, cout, 1)
        self.eca_out = ECA(cout)

    def forward(self, c3, c4, c5, p3, p4, p5):
        cs = [proj(eca(c)) for eca, proj, c in zip(self.eca, self.proj, (c3, c4, c5))]
        ys = []
        for i, (p, size) in enumerate(zip((p3, p4, p5),
                                          (p3.shape[-2:], p4.shape[-2:], p5.shape[-2:]))):
            pi = self.af[i](cs[0], cs[1], cs[2], size)
            ys.append(self.as2m[i](cs[i], pi))
        y1, y2, y3 = ys
        y2 = F.interpolate(y2, size=y1.shape[-2:], mode="bilinear", align_corners=False)
        y3 = F.interpolate(y3, size=y1.shape[-2:], mode="bilinear", align_corners=False)
        return self.eca_out(self.reduce(torch.cat([y1, y2, y3], dim=1)))


# --------------------------------------------------------------------------- #
# backbone: ResNet50 + five-stage frequency branch (Fig. 4a)
# --------------------------------------------------------------------------- #
class TwoBranchBackbone(nn.Module):
    """Spatial branch (ResNet50 C1-C5) + frequency branch (five EWT_blocks)."""

    def __init__(self, eca_dirs=("h", "w"), ewt_mode="concat", use_dfem=True,
                 use_fsd=True, pretrained=False):
        super().__init__()
        net = resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.use_dfem, self.use_fsd = use_dfem, use_fsd
        if use_dfem:
            self.ewt = nn.ModuleList([
                EWTBlock(64, 64, eca_dirs, ewt_mode),
                EWTBlock(64, 256, eca_dirs, ewt_mode),
                EWTBlock(256, 512, eca_dirs, ewt_mode),
                EWTBlock(512, 1024, eca_dirs, ewt_mode),
                EWTBlock(1024, 2048, eca_dirs, ewt_mode),
            ])
            # ASSUMPTION A2: Sec. 3.1 states one module per branch per stage but
            # not how the two branches are combined; we add them element-wise at
            # every stage and apply FSD-AFM only at C5 (Sec. 3.3).
            self.fsd = FSDAFM(2048, 2048, 256) if use_fsd else None

    def forward(self, x):
        s = self.stem(x)                                  # C1
        if self.use_dfem:
            s = s + self.ewt[0](s)
        feats = []
        f_last = s_last = None
        for i, stage in enumerate((self.layer1, self.layer2, self.layer3, self.layer4)):
            f = self.ewt[i + 1](s) if self.use_dfem else None   # branch reads stage input
            s = stage(s)
            f_last, s_last = f, s
            if self.use_dfem:
                # the block reads the stage input; the stage output is half the size
                if f.shape[-2:] != s.shape[-2:]:
                    f = F.interpolate(f, size=s.shape[-2:], mode="nearest")
                s = s + f
            feats.append(s)
        if self.use_dfem and self.fsd is not None:
            if f_last.shape[-2:] != s_last.shape[-2:]:     # C4 branch -> C5 resolution
                f_last = F.interpolate(f_last, size=s_last.shape[-2:], mode="nearest")
            feats[-1] = self.fsd(f_last, s_last)          # Eq. (6)-(8) at C5
        return feats[1], feats[2], feats[3]               # C3, C4, C5


# --------------------------------------------------------------------------- #
# YOLACT heads
# --------------------------------------------------------------------------- #
class FPN(nn.Module):
    def __init__(self, in_chs=(512, 1024, 2048), out=256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out, 1) for c in in_chs])
        self.smooth = nn.ModuleList([ConvBNReLU(out, out, 3) for _ in range(3)])
        self.p6 = ConvBNReLU(out, out, 3, s=2)
        self.p7 = ConvBNReLU(out, out, 3, s=2)

    def forward(self, c3, c4, c5):
        l3, l4, l5 = [lat(c) for lat, c in zip(self.lat, (c3, c4, c5))]
        p5 = l5
        p4 = l4 + F.interpolate(p5, size=l4.shape[-2:], mode="nearest")
        p3 = l3 + F.interpolate(p4, size=l3.shape[-2:], mode="nearest")
        p3, p4, p5 = [s(p) for s, p in zip(self.smooth, (p3, p4, p5))]
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))
        return p3, p4, p5, p6, p7


class Protonet(nn.Module):
    def __init__(self, cin=256, mask_dim=32):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(cin, 256, 3), ConvBNReLU(256, 256, 3),
            ConvBNReLU(256, 256, 3), nn.Conv2d(256, mask_dim, 1))

    def forward(self, x):
        return self.body(x)


class PredictionHead(nn.Module):
    def __init__(self, cin=256, num_classes=1, num_priors=3, mask_dim=32):
        super().__init__()
        self.num_classes, self.num_priors = num_classes, num_priors
        self.mask_dim = mask_dim
        self.body = ConvBNReLU(cin, 256, 3)
        self.cls = nn.Conv2d(256, num_priors * num_classes, 3, 1, 1)
        self.box = nn.Conv2d(256, num_priors * 4, 3, 1, 1)
        self.mask = nn.Conv2d(256, num_priors * mask_dim, 3, 1, 1)

    def forward(self, x):
        f = self.body(x)
        b, _, h, w = f.shape
        cls = self.cls(f).permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
        box = self.box(f).permute(0, 2, 3, 1).reshape(b, -1, 4)
        # one coefficient vector per anchor, aligned with cls/box
        msk = self.mask(f).permute(0, 2, 3, 1).reshape(b, -1, self.mask_dim)
        return cls, box, msk, (h, w)


class TWF2YOLACT(nn.Module):
    """Full model.  ``variant`` selects the ablation rows of Table 1/2."""

    VARIANTS = {
        "yolact":   dict(use_dfem=False, use_fsd=False, use_mfiem=False),
        "dfem":     dict(use_dfem=True, use_fsd=False, use_mfiem=False),
        "dfem_fsd": dict(use_dfem=True, use_fsd=True, use_mfiem=False),
        "full":     dict(use_dfem=True, use_fsd=True, use_mfiem=True),
    }

    def __init__(self, num_classes=1, mask_dim=32, eca_dirs=("h", "w"),
                 ewt_mode="concat", variant="full", pretrained=False,
                 conf_thresh=0.05, nms_thresh=0.6, max_dets=100):
        super().__init__()
        flags = dict(self.VARIANTS[variant])
        self.cfg = dict(num_classes=num_classes, mask_dim=mask_dim,
                        eca_dirs=tuple(eca_dirs), ewt_mode=ewt_mode,
                        variant=variant, conf_thresh=conf_thresh,
                        nms_thresh=nms_thresh, max_dets=max_dets)
        self.backbone = TwoBranchBackbone(eca_dirs, ewt_mode, flags["use_dfem"],
                                          flags["use_fsd"], pretrained)
        # FSD-AFM returns 256 channels at C5 (Sec. 3.3); without it C5 keeps 2048
        c5_ch = 256 if flags["use_fsd"] else 2048
        self.fpn = FPN(in_chs=(512, 1024, c5_ch))
        self.mfiem = MFIEM(c5=c5_ch) if flags["use_mfiem"] else None
        self.protonet = Protonet(256, mask_dim)
        # one head per FPN level, as in YOLACT (P3-P7)
        self.heads = nn.ModuleList(
            [PredictionHead(256, num_classes, 3, mask_dim) for _ in range(5)])

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5, p6, p7 = self.fpn(c3, c4, c5)
        proto_in = self.mfiem(c3, c4, c5, p3, p4, p5) if self.mfiem is not None else p3
        proto = self.protonet(proto_in)
        outs = [h(p) for h, p in zip(self.heads, (p3, p4, p5, p6, p7))]
        return {
            "cls": torch.cat([o[0] for o in outs], 1),
            "box": torch.cat([o[1] for o in outs], 1),
            "coeff": torch.cat([o[2] for o in outs], 1),
            "proto": proto,
            "levels": [o[3] for o in outs],
        }


# --------------------------------------------------------------------------- #
# priors, decoding, mask assembly
# --------------------------------------------------------------------------- #
def make_priors(levels, img_hw, device, aspect=((1.0, 1.0), (1.0, 2.0), (2.0, 1.0))):
    """Anchor grid in image coordinates, ordered position-major then anchor.

    Matches the flattening used by :class:`PredictionHead` (h, w, num_priors).
    """
    priors = []
    for (h, w) in levels:
        stride = img_hw[0] / float(h)                     # per-level stride
        gy, gx = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device), indexing="ij")
        cx = (gx.reshape(-1).float() + 0.5) * stride
        cy = (gy.reshape(-1).float() + 0.5) * stride
        lvl = []
        for aw, ah in aspect:
            lvl.append(torch.stack([cx, cy,
                                    torch.full_like(cx, aw * stride * 4),
                                    torch.full_like(cx, ah * stride * 4)], 1))
        priors.append(torch.stack(lvl, 1).reshape(-1, 4))   # (h*w*A, 4)
    return torch.cat(priors, 0)


def decode_boxes(raw, priors):
    """(cx,cy,w,h) offsets relative to the anchor box -> xyxy."""
    cx = priors[:, 0] + raw[:, 0] * priors[:, 2]
    cy = priors[:, 1] + raw[:, 1] * priors[:, 3]
    w = priors[:, 2] * torch.exp(raw[:, 2].clamp(-4, 4))
    h = priors[:, 3] * torch.exp(raw[:, 3].clamp(-4, 4))
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


@torch.no_grad()
def postprocess(out, img_hw, cfg):
    """Single-image inference: boxes, scores, classes, masks (all on cpu)."""
    from torchvision.ops import nms

    proto = out["proto"][0]
    scores_all = torch.sigmoid(out["cls"][0]).max(dim=1).values
    keep = scores_all > cfg["conf_thresh"]
    if keep.sum() == 0:
        return (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long),
                torch.zeros(0, *img_hw))
    # decode every anchor first, then apply the confidence mask: the priors array
    # is indexed by anchor, so filtering before decoding would misalign the two
    priors = make_priors(out["levels"], img_hw, out["box"].device)
    boxes = decode_boxes(out["box"][0], priors)[keep]
    scores = scores_all[keep]
    classes = torch.sigmoid(out["cls"][0][keep]).argmax(1)
    coeff = out["coeff"][0].reshape(-1, out["coeff"].shape[-1])[keep]
    idx = nms(boxes, scores, cfg["nms_thresh"])[: cfg["max_dets"]]
    boxes, scores, classes, coeff = boxes[idx], scores[idx], classes[idx], coeff[idx]

    hp, wp = proto.shape[-2:]
    masks = torch.sigmoid(coeff @ proto.reshape(proto.shape[0], -1)).reshape(-1, hp, wp)
    masks = F.interpolate(masks.unsqueeze(1), size=img_hw, mode="bilinear",
                          align_corners=False).squeeze(1)
    ys = torch.arange(img_hw[0], device=masks.device).view(1, -1, 1)
    xs = torch.arange(img_hw[1], device=masks.device).view(1, 1, -1)
    crop = ((xs >= boxes[:, 0:1].unsqueeze(-1)) & (xs < boxes[:, 2:3].unsqueeze(-1)) &
            (ys >= boxes[:, 1:2].unsqueeze(-1)) & (ys < boxes[:, 3:4].unsqueeze(-1)))
    masks = (masks > 0.5).float() * crop.float()
    return boxes.cpu(), scores.cpu(), classes.cpu(), masks.cpu()
