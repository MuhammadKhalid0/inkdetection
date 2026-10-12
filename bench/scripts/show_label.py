"""Render the ground-truth ink label(s) of a segment as a 2D image.

Usage:
  python scripts/show_label.py --seg 20241113090990          # original + registered
  python scripts/show_label.py --seg 20241113090990 --which reg
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
from data.segments import VAL_SEGMENT, proc_dir, OUTPUT_ROOT  # noqa: E402


def load(seg, name):
    p = os.path.join(proc_dir(seg), name)
    return np.load(p, mmap_mode="r") if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=VAL_SEGMENT)
    ap.add_argument("--which", choices=["orig", "reg", "both"], default="both")
    ap.add_argument("--ds", type=int, default=1, help="downsample factor")
    args = ap.parse_args()

    ds = max(1, args.ds)
    ink = load(args.seg, "ink.npy")
    inkr = load(args.seg, "ink_reg.npy")
    mask = load(args.seg, "mask.npy")
    if ink is None:
        sys.exit(f"no ink.npy for {args.seg}")

    ink = np.asarray(ink[::ds, ::ds]) > 0
    inkr = (np.asarray(inkr[::ds, ::ds]) > 0) if inkr is not None else None
    mk = (np.asarray(mask[::ds, ::ds]) > 0) if mask is not None else None
    H, W = ink.shape
    print(f"[label] {args.seg} shape={ink.shape} ink%={100*ink.mean():.2f}")

    panels = []
    if args.which in ("orig", "both"):
        panels.append(("ink.npy (original)", ink))
    if args.which in ("reg", "both") and inkr is not None:
        panels.append(("ink_reg.npy (registered)", inkr))

    # segment is very wide/tall -> orient so the long axis is horizontal
    rot = W < H
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(24, 5 * n), squeeze=False)
    for i, (title, img) in enumerate(panels):
        disp = np.rot90(img) if rot else img
        rgb = np.zeros((*disp.shape, 3), dtype=np.float32)
        if mk is not None:
            mkd = np.rot90(mk) if rot else mk
            rgb[mkd] = 0.12  # dark grey = papyrus inside mask
        rgb[disp] = [1.0, 0.85, 0.1]  # yellow = ink
        ax = axes[i, 0]
        ax.imshow(rgb)
        ax.set_title(f"{title}   ink%={100*img.mean():.2f}", fontsize=11)
        ax.axis("off")

    fig.suptitle(f"Ink label — {args.seg}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUTPUT_ROOT, "labels", f"label_{args.seg[:16]}_{args.which}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[label] saved -> {out}")


if __name__ == "__main__":
    main()
