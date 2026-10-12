"""Plot training loss + validation scores for a run — plan 2026-07-17.

Reads <root>/<run>/train_log.csv and writes <root>/<run>/figs/train_curves.png:
  panel 1: train loss per epoch
  panel 2: val F0.5 (fixed tau + swept) per epoch, best epoch marked
  panel 3 (new runs only): val IoU ink/bg at the swept tau, if logged

Note: there is no separate 'validation loss' in this pipeline — validation is
full-segment inference scored by F0.5 (see engine/train.py), so the score
curves ARE the validation signal.

Usage:  python scripts/plot_train_curves.py --run s2_wordbox_t192
        (root resolves via $INK_OUTPUT_ROOT, falling back to the old outputs)
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from data.segments import OUTPUT_ROOT  # noqa: E402  (env-aware)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=OUTPUT_ROOT)
    args = ap.parse_args()

    log_path = os.path.join(args.root, args.run, "train_log.csv")
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        sys.exit(f"empty log: {log_path}")

    ep = np.array([int(r["epoch"]) for r in rows])
    loss = np.array([float(r["train_loss"]) for r in rows])
    f_fixed = np.array([float(r["val_f05"]) for r in rows])
    f_sweep = np.array([float(r["val_f05_sweep"]) for r in rows])
    has_iou = "val_iou_ink_sweep" in rows[0] and rows[0]["val_iou_ink_sweep"] != ""
    best_i = int(np.argmax(f_sweep))

    n_panels = 3 if has_iou else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), sharex=True)

    axes[0].plot(ep, loss, color="tab:red")
    axes[0].set_ylabel("train loss")
    axes[0].set_title(f"{args.run} — best swept F0.5 = {f_sweep[best_i]:.4f} @ e{ep[best_i]}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, f_fixed, label="val F0.5 @ fixed tau", color="tab:gray")
    axes[1].plot(ep, f_sweep, label="val F0.5 (swept)", color="tab:blue")
    axes[1].axvline(ep[best_i], color="tab:blue", ls="--", alpha=0.5)
    axes[1].scatter([ep[best_i]], [f_sweep[best_i]], color="tab:blue", zorder=3)
    axes[1].set_ylabel("val F0.5")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.3)

    if has_iou:
        iou_ink = np.array([float(r["val_iou_ink_sweep"]) for r in rows])
        iou_bg = np.array([float(r["val_iou_bg_sweep"]) for r in rows])
        axes[2].plot(ep, iou_ink, label="val IoU ink (swept tau)", color="tab:green")
        axes[2].plot(ep, iou_bg, label="val IoU background", color="tab:olive")
        axes[2].axvline(ep[best_i], color="tab:blue", ls="--", alpha=0.5)
        axes[2].set_ylabel("val IoU")
        axes[2].legend(loc="lower right", fontsize=8)
        axes[2].grid(alpha=0.3)

    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    out = os.path.join(args.root, args.run, "figs", "train_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
