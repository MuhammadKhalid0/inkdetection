"""Segmentation losses.  A0 default = Dice+BCE (equal weight)."""
import torch
import torch.nn.functional as F


def dice_bce(logits, y, w: float = 0.5, eps: float = 1.0):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(-1, -2))
    denom = p.sum(dim=(-1, -2)) + y.sum(dim=(-1, -2))
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return w * bce + (1.0 - w) * dice.mean()


def tversky_bce(logits, y, a: float = 0.7, b: float = 0.3, w: float = 0.5, eps: float = 1.0):
    p = torch.sigmoid(logits)
    tp = (p * y).sum(dim=(-1, -2))
    fp = (p * (1 - y)).sum(dim=(-1, -2))
    fn = ((1 - p) * y).sum(dim=(-1, -2))
    tv = 1.0 - (tp + eps) / (tp + a * fp + b * fn + eps)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    return w * bce + (1.0 - w) * tv.mean()


def dice_bce_masked(logits, y, u=None, w: float = 0.5, eps: float = 1.0):
    """Variant A of the uncertainty handling (plan 2026-07-17 §6.3).

    ``u`` is the per-pixel trimap (2 = confident ink core, 0 = confident
    background, 1 = uncertain band).  Uncertain pixels are EXCLUDED from both
    the BCE and the Dice terms.  With ``u=None`` (e.g. smoke test) this is
    exactly ``dice_bce``.
    """
    if u is None:
        return dice_bce(logits, y, w=w, eps=eps)
    valid = (u != 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    p = torch.sigmoid(logits) * valid
    yv = y * valid
    inter = (p * yv).sum(dim=(-1, -2))
    denom = p.sum(dim=(-1, -2)) + yv.sum(dim=(-1, -2))
    dice = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return w * bce + (1.0 - w) * dice.mean()


def soft_ce3(logits, y, u=None):
    """Variant B: 3-class CE with soft targets (plan 2026-07-17 §6.4).

    Channels: 0 = ink, 1 = background, 2 = uncertain.  Targets:
      confident ink (u==2)        -> [0.90, 0.05, 0.05]
      confident background (u==0) -> [0.05, 0.90, 0.05]
      uncertain band (u==1)       -> [0.25, 0.25, 0.50]   (the low-prob class)
    With ``u=None`` (smoke test) the trimap is derived from ``y`` (no band).
    """
    if u is None:
        u = torch.where(y > 0.5, torch.full_like(y, 2.0), torch.zeros_like(y))
    B, _, H, W = logits.shape
    t = logits.new_zeros(B, 3, H, W)
    ink = (u == 2).squeeze(1)
    bg = (u == 0).squeeze(1)
    unc = (u == 1).squeeze(1)
    for cls_mask, vec in ((ink, (0.90, 0.05, 0.05)),
                          (bg, (0.05, 0.90, 0.05)),
                          (unc, (0.25, 0.25, 0.50))):
        for c in range(3):
            t[:, c][cls_mask] = vec[c]
    logp = F.log_softmax(logits, dim=1)
    return -(t * logp).sum(dim=1).mean()


def build_loss(cfg_train):
    name = cfg_train.get("loss", "dice_bce")
    if name == "dice_bce":
        return dice_bce
    if name == "tversky_bce":
        a = cfg_train.get("tversky_alpha", 0.7)
        b = cfg_train.get("tversky_beta", 0.3)
        return lambda logits, y: tversky_bce(logits, y, a=a, b=b)
    if name == "dice_bce_masked":
        return dice_bce_masked
    if name == "soft_ce3":
        return soft_ce3
    raise ValueError(f"unknown loss {name}")


# Losses whose signature takes the uncertainty trimap as a third argument.
LOSSES_NEED_UNC = {"dice_bce_masked", "soft_ce3"}
