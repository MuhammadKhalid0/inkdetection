"""Evaluation metrics: F-beta (primary F0.5), Dice, precision, recall, plus a
threshold sweep.  All metrics are computed over valid-mask pixels only.
"""
import numpy as np


def pixel_metrics(prob, gt, mask, tau, beta=0.5, eps=1e-7):
    """prob,gt,mask: 2D arrays.  gt,mask in {0,1}.  Returns dict at threshold tau."""
    m = mask > 0
    p = (prob[m] >= tau).astype(np.float64)
    g = (gt[m] > 0).astype(np.float64)
    tp = float((p * g).sum())
    fp = float((p * (1 - g)).sum())
    fn = float(((1 - p) * g).sum())
    tn = float(((1 - p) * (1 - g)).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    b2 = beta * beta
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    # IoU per class (2026-07-17, supervisor request): foreground = ink,
    # background = papyrus. Reported alongside F0.5; never used for selection.
    iou_ink = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    return {"tau": float(tau), "fbeta": fbeta, "dice": dice,
            "precision": precision, "recall": recall,
            "iou_ink": iou_ink, "iou_bg": iou_bg,
            "miou": 0.5 * (iou_ink + iou_bg)}


def sweep_threshold(prob, gt, mask, taus=None, beta=0.5):
    """Sweep tau, return (best_dict, all_rows).  Best by F-beta."""
    if taus is None:
        taus = np.round(np.arange(0.1, 0.91, 0.05), 2)
    rows = [pixel_metrics(prob, gt, mask, t, beta=beta) for t in taus]
    best = max(rows, key=lambda r: r["fbeta"])
    return best, rows
