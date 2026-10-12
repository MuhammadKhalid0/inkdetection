"""Render the TEST-segment ink PREDICTION as a 2D label-style image.

Yellow = predicted ink (prob >= tau), dark grey = papyrus inside mask.
No ground truth exists for the test segment -> qualitative only.

Usage:
  python scripts/show_pred.py --run a2bs --tau 0.5 --ds 2
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data.segments import TEST_SEGMENT, proc_infer_dir, OUTPUT_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="a2bs")
    ap.add_argument("--seg", default=TEST_SEGMENT)
    ap.add_argument("--prob", default=None)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--ds", type=int, default=2, help="downsample factor")
    args = ap.parse_args()

    prob_path = args.prob
    if prob_path is None:
        cands = sorted(glob.glob(os.path.join(OUTPUT_ROOT, args.run, "test_prob_*.npy")))
        if not cands:
            sys.exit(f"no test_prob_*.npy under outputs/{args.run}")
        prob_path = cands[0]

    ds = max(1, args.ds)
    prob = np.asarray(np.load(prob_path, mmap_mode="r")[::ds, ::ds])
    mask = np.asarray(np.load(os.path.join(proc_infer_dir(args.seg), "mask.npy"),
                              mmap_mode="r")[::ds, ::ds]) > 0
    pred = (prob >= args.tau) & mask
    H, W = pred.shape
    cov = 100 * pred.sum() / max(mask.sum(), 1)
    print(f"[pred] {prob_path} shape={prob.shape} tau={args.tau} coverage={cov:.2f}%")

    # orient long axis horizontal
    rot = W < H
    disp = np.rot90(pred) if rot else pred
    mkd = np.rot90(mask) if rot else mask
    rgb = np.zeros((*disp.shape, 3), dtype=np.float32)
    rgb[mkd] = 0.12
    rgb[disp] = [1.0, 0.85, 0.1]

    fig, ax = plt.subplots(figsize=(26, 26 * disp.shape[0] / disp.shape[1]))
    ax.imshow(rgb)
    ax.set_title(f"TEST prediction — {args.seg[:40]}  run={args.run}  "
                 f"tau={args.tau}  coverage={cov:.2f}%  (NO labels)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OUTPUT_ROOT, args.run, "figs", f"test_pred_label_tau{args.tau}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[pred] saved -> {out}")


if __name__ == "__main__":
    main()
