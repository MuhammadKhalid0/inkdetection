"""Readable qualitative visualization of the TEST segment prediction (no labels).

The test strip is huge (11140 x 49240).  This renders vertically-stacked
horizontal chunks: CT surface background with the prediction contour, and the
probability heatmap, so ink letterforms are legible for qualitative review.

Usage:
  python scripts/visualize_test.py --run a2bs --tau 0.6 --chunks 8
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
    ap.add_argument("--prob", default=None, help="explicit prob .npy (default outputs/<run>/test_prob_*.npy)")
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--ds", type=int, default=2, help="spatial downsample for display")
    args = ap.parse_args()

    prob_path = args.prob
    if prob_path is None:
        cands = sorted(glob.glob(os.path.join(OUTPUT_ROOT, args.run, "test_prob_*.npy")))
        if not cands:
            sys.exit(f"no test_prob_*.npy under outputs/{args.run}")
        prob_path = cands[0]
    prob = np.load(prob_path, mmap_mode="r")
    proc = proc_infer_dir(args.seg)
    vol = np.load(os.path.join(proc, "vol.npy"), mmap_mode="r")
    mask = np.load(os.path.join(proc, "mask.npy"), mmap_mode="r")
    H, W = mask.shape
    ds = max(1, args.ds)
    print(f"[viz] prob={prob_path} shape={prob.shape} tau={args.tau}")

    # CT background: central-band mean, downsampled
    slab = np.asarray(vol[26:38, ::ds, ::ds], dtype=np.float32).mean(0)
    mk = np.asarray(mask[::ds, ::ds]) > 0
    pr = np.asarray(prob[::ds, ::ds])
    pb98 = np.percentile(slab[mk], 98) if mk.any() else 255.0
    bg = np.clip(slab / max(pb98, 1e-3), 0, 1)
    Hd, Wd = bg.shape
    frac = float((pr[mk] >= args.tau).mean()) if mk.any() else 0.0
    print(f"[viz] predicted-ink coverage in mask @tau={args.tau}: {100*frac:.1f}%")

    edges = np.linspace(0, Wd, args.chunks + 1).astype(int)
    fig, axes = plt.subplots(args.chunks, 2, figsize=(26, 2.6 * args.chunks))
    for i in range(args.chunks):
        x0, x1 = edges[i], edges[i + 1]
        b = bg[:, x0:x1]
        p = pr[:, x0:x1]
        m = mk[:, x0:x1]
        pth = (p >= args.tau) & m

        ax = axes[i, 0]
        ax.imshow(b, cmap="gray", vmin=0, vmax=1)
        ax.contour(pth.astype(float), levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"CT + prediction contour   x[{x0*ds}:{x1*ds}]", fontsize=8)
        ax.axis("off")

        ax = axes[i, 1]
        pv = np.where(m, p, np.nan)
        ax.imshow(pv, cmap="magma", vmin=0, vmax=1)
        ax.set_title("ink probability", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"TEST {args.seg[:40]}  run={args.run}  (NO labels — qualitative)  "
                 f"tau={args.tau}  coverage={100*frac:.1f}%", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(OUTPUT_ROOT, args.run, "figs", f"test_tau{args.tau}_chunks.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[viz] saved -> {out}")


if __name__ == "__main__":
    main()
