"""Readable validation-segment visualization: the 1619x11479 strip is chopped into
vertically-stacked horizontal chunks so letters are legible.  Shows, per chunk,
the CT background with prediction contour and a TP/FP/FN overlay.

Usage:
  python scripts/visualize_val.py --run a2bs --label-file ink_reg.npy --tau 0.6
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
from data.segments import VAL_SEGMENT, proc_dir  # noqa: E402

OUT_ROOT = "/home/woody/iwi5/iwi5401h/vesuvius/outputs"


def f05(pred, gt, m):
    tp = float((pred & gt & m).sum()); fp = float((pred & ~gt & m).sum()); fn = float((~pred & gt & m).sum())
    p = tp / max(tp + fp, 1.0); r = tp / max(tp + fn, 1.0)
    return 1.25 * p * r / max(0.25 * p + r, 1e-9), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="a2bs")
    ap.add_argument("--label-file", default="ink_reg.npy")
    ap.add_argument("--tau", type=float, default=None, help="if unset, pick best-sweep tau on this label set")
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--ds", type=int, default=1)
    args = ap.parse_args()

    proc = proc_dir(VAL_SEGMENT)
    vol = np.load(os.path.join(proc, "vol.npy"), mmap_mode="r")
    slab = np.asarray(vol[26:38], dtype=np.float32).mean(axis=0)          # CT background
    ink = np.load(os.path.join(proc, args.label_file)) > 0
    mask = np.load(os.path.join(proc, "mask.npy")) > 0
    prob = np.load(os.path.join(OUT_ROOT, args.run, "val_prob_best.npy"))

    tau = args.tau
    if tau is None:
        tau = max(np.arange(0.3, 0.91, 0.05),
                  key=lambda t: f05(prob >= t, ink, mask)[0])
        tau = round(float(tau), 2)
    f, p, r = f05(prob >= tau, ink, mask)
    print(f"{args.run} vs {args.label_file}: F0.5={f:.3f} P={p:.3f} R={r:.3f} @tau={tau}")

    H, W = mask.shape
    ds = max(1, args.ds)
    edges = np.linspace(0, W, args.chunks + 1).astype(int)
    pb98 = np.percentile(slab[mask], 98)
    bg = np.clip(slab / max(pb98, 1e-3), 0, 1)

    fig, axes = plt.subplots(args.chunks, 2, figsize=(26, 2.3 * args.chunks))
    for i in range(args.chunks):
        x0, x1 = edges[i], edges[i + 1]
        b = bg[::ds, x0:x1:ds]
        pr = (prob[::ds, x0:x1:ds] >= tau)
        gt = ink[::ds, x0:x1:ds]
        mk = mask[::ds, x0:x1:ds]

        ax = axes[i, 0]
        ax.imshow(b, cmap="gray", vmin=0, vmax=1)
        ax.contour(pr.astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title(f"CT + prediction contour   x[{x0}:{x1}]", fontsize=8)
        ax.axis("off")

        ov = np.zeros((*pr.shape, 3), dtype=np.float32)
        ov[pr & gt] = [0, 1, 0]      # TP green
        ov[pr & ~gt] = [1, 0, 0]     # FP red
        ov[~pr & gt] = [0, 0.45, 1]  # FN blue
        ov[~mk] = 0
        ax = axes[i, 1]
        ax.imshow(ov)
        ax.set_title("TP=green  FP=red  FN=blue", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"VAL {VAL_SEGMENT}  run={args.run}  labels={args.label_file}  "
                 f"F0.5={f:.3f} P={p:.3f} R={r:.3f} @tau={tau}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(OUT_ROOT, args.run, "figs",
                       f"val_{args.label_file.replace('.npy','')}_tau{tau}_chunks.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
