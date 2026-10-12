"""Interpretability probe 2 (plan §14.4): effective receptive field (ERF)
across the tile ladder — model-agnostic, gradient-based, no training.

For each checkpoint: sample ink-positive val tiles, backprop |d logit_center /
d input| and average -> a 2D map of how far input pixels influence one output
pixel. Radial statistics (r50/r90 = radius containing 50/90% of gradient mass)
show whether bigger tiles are actually *used* as context — the mechanistic
reading of the +0.16 tile ladder and the t320 turn.

fp32 throughout (clean gradients), batch 1.

Usage:  python -m scripts.interp_erf [--n-tiles 32]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import MANIFEST_ROOT, VAL_SEGMENT, proc_dir  # noqa: E402
from data.dataset import normalize_volume  # noqa: E402
from models.wrapper import build_model  # noqa: E402

W = "/home/woody/iwi5/iwi5401h/vesuvius"
MODELS = [  # (root, run, val manifest)
    (f"{W}/outputs", "a2bs_reg", "val_reg.csv"),
    (f"{W}/outputs", "a2bs_t128", "val_reg_t128.csv"),
    (f"{W}/outputs", "lean_t192_s42", "val_reg_t192.csv"),
    (f"{W}/outputs4", "t256_lean_s42", "val_reg_t256.csv"),
]


def erf_for(root, run, manifest, n_tiles, device, rng):
    ckpt = torch.load(os.path.join(root, run, "best.pt"), map_location=device,
                      weights_only=False)
    cfg = ckpt["cfg"]
    d = cfg["data"]
    tile, depth = int(d["tile"]), int(d["depth"])
    center, n_layers = int(d.get("center_layer", 32)), int(d.get("n_layers", 65))
    z0 = max(0, min(center - depth // 2, n_layers - depth))
    model = build_model(cfg).to(device).float()
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = [r for r in csv.DictReader(open(os.path.join(MANIFEST_ROOT, manifest)))
            if float(r["ink_frac"]) >= 0.05]
    pick = rng.choice(len(rows), size=min(n_tiles, len(rows)), replace=False)
    vol = np.load(os.path.join(proc_dir(VAL_SEGMENT), "vol.npy"), mmap_mode="r")

    acc = np.zeros((tile, tile), dtype=np.float64)
    tc = tile // 2
    for i in pick:
        r = rows[i]
        y, x = int(r["y"]), int(r["x"])
        xv = np.asarray(vol[z0:z0 + depth, y:y + tile, x:x + tile], dtype=np.float32)
        xv = normalize_volume(xv, d.get("norm_mode", "global"), float(d.get("normalize", 255.0)))
        xb = torch.from_numpy(xv).unsqueeze(0).unsqueeze(0).to(device)
        xb.requires_grad_(True)
        logits = model(xb)
        out = logits[0, 0, tc - 1:tc + 2, tc - 1:tc + 2].mean()   # 3x3 centre
        model.zero_grad(set_to_none=True)
        out.backward()
        acc += xb.grad.detach().abs().sum(dim=(0, 1, 2)).cpu().numpy()
    acc /= acc.sum()

    # radial mass profile around the centre
    yy, xx = np.mgrid[0:tile, 0:tile]
    rad = np.sqrt((yy - tc) ** 2 + (xx - tc) ** 2)
    # measure to the corner radius: at small tiles >10% of gradient mass can sit
    # beyond the inscribed circle (i.e. the ERF fills the whole tile)
    rs = np.arange(0, int(np.ceil(tile / np.sqrt(2))) + 2)
    cum = np.array([acc[rad <= r].sum() for r in rs])
    r50 = float(rs[min(np.searchsorted(cum, 0.5), len(rs) - 1)])
    r90 = float(rs[min(np.searchsorted(cum, 0.9), len(rs) - 1)])
    print(f"[{run}] tile {tile}: r50={r50:.0f}px r90={r90:.0f}px")
    return {"run": run, "tile": tile, "r50": r50, "r90": r90,
            "map": acc, "rs": rs, "cum": cum}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tiles", type=int, default=32)
    ap.add_argument("--out", default=f"{W}/output_inter")
    # 2026-07-20: optional explicit model list so the probe can run on the
    # nnFormer tree (models=<root>:<run>:<val manifest>, comma-separated).
    # Omitted -> the historical UNETR++ ladder above, byte-identical behaviour.
    ap.add_argument("--models", default=None)
    args = ap.parse_args()
    global MODELS
    if args.models:
        MODELS = [tuple(m.split(":")) for m in args.models.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    out_dir = os.path.join(args.out, "erf")
    os.makedirs(out_dir, exist_ok=True)

    results = [erf_for(*m, args.n_tiles, device, rng) for m in MODELS]

    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4.2 * (len(results) + 1), 4))
    for ax, r in zip(axes, results):
        im = ax.imshow(np.log10(r["map"] + 1e-12), cmap="inferno")
        ax.set_title(f"{r['run']}\ntile {r['tile']} — r50={r['r50']:.0f} r90={r['r90']:.0f}px",
                     fontsize=9)
        ax.axis("off")
    for r in results:
        axes[-1].plot(r["rs"], r["cum"], label=f"t{r['tile']} ({r['run']})")
    axes[-1].axhline(0.9, color="k", ls=":", lw=1)
    axes[-1].set_xlabel("radius from centre (px)"); axes[-1].set_ylabel("cum. gradient mass")
    axes[-1].legend(fontsize=7); axes[-1].grid(alpha=0.3)
    axes[-1].set_title("ERF radial profiles")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "erf_maps.png"), dpi=130, bbox_inches="tight")

    with open(os.path.join(out_dir, "erf_stats.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k in ("run", "tile", "r50", "r90")}
                   for r in results], f, indent=2)
    print(f"-> {out_dir}/erf_maps.png")


if __name__ == "__main__":
    main()
