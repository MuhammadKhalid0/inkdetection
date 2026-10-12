"""Visualize a saved probability map against ground truth (validation) or alone (test).

Usage:
    python scripts/visualize.py --prob outputs/a0/val_prob_best.npy --seg 20241113090990 --tau 0.4
    python scripts/visualize.py --prob outputs/a0/test_prob.npy --seg <test> --no-gt --tau 0.4

Produces a stacked PNG: probability heatmap | thresholded prediction |
[ground truth | overlay(TP green/FP red/FN blue)] into outputs/<run>/figs/.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data.segments import proc_dir  # noqa: E402


def load_gt(seg, label_file="ink.npy"):
    ink = np.load(os.path.join(proc_dir(seg), label_file))
    mask = np.load(os.path.join(proc_dir(seg), "mask.npy"))
    return ink, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--no-gt", action="store_true")
    ap.add_argument("--label-file", default="ink.npy",
                    help="ink.npy or ink_reg.npy (registered labels)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--downsample", type=int, default=2, help="spatial subsample for display")
    args = ap.parse_args()

    prob = np.load(args.prob)
    ds = max(1, args.downsample)
    prob_d = prob[::ds, ::ds]
    pred = (prob_d >= args.tau).astype(np.float32)

    have_gt = not args.no_gt
    if have_gt:
        ink, mask = load_gt(args.seg, args.label_file)
        ink_d = (ink[::ds, ::ds] > 0).astype(np.float32)
        mask_d = (mask[::ds, ::ds] > 0)
        # metrics on full-res
        m = mask > 0
        p = (prob[m] >= args.tau)
        g = ink[m] > 0
        tp = float((p & g).sum()); fp = float((p & ~g).sum()); fn = float((~p & g).sum())
        prec = tp / (tp + fp + 1e-7); rec = tp / (tp + fn + 1e-7)
        f05 = 1.25 * prec * rec / (0.25 * prec + rec + 1e-7)
        title = f"{args.seg}  F0.5={f05:.3f}  P={prec:.3f}  R={rec:.3f}  @tau={args.tau}"
    else:
        title = f"{args.seg}  (no labels)  @tau={args.tau}"

    n_panels = 4 if have_gt else 2
    H, W = prob_d.shape
    aspect = W / max(H, 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=(min(24, 4 * aspect), 3 * n_panels))
    if n_panels == 2:
        axes = list(axes)

    axes[0].imshow(prob_d, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("probability")
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title(f"prediction (tau={args.tau})")

    if have_gt:
        axes[2].imshow(ink_d, cmap="gray")
        axes[2].set_title("ground truth ink")
        overlay = np.zeros((*pred.shape, 3), dtype=np.float32)
        pb = pred > 0.5; gb = ink_d > 0.5
        overlay[pb & gb] = [0, 1, 0]     # TP green
        overlay[pb & ~gb] = [1, 0, 0]    # FP red
        overlay[~pb & gb] = [0, 0.4, 1]  # FN blue
        overlay[~mask_d] = 0
        axes[3].imshow(overlay)
        axes[3].set_title("overlay  TP=green FP=red FN=blue")

    for a in axes:
        a.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.prob), "figs",
                                   f"{args.seg}_tau{args.tau}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")
    print(title)


if __name__ == "__main__":
    main()
