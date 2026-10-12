"""Interpretability probe 4 (plan §14.4): the learned depth readout.

Reads the depth-collapse weighting straight out of each checkpoint (no GPU):
  * surface_relative collapse -> softmax(phi_rel) over relative depth r in
    [-(D-1), +(D-1)], plotted against its Gaussian init (sigma0 = D/4);
  * learned collapse -> softmax(phi) over absolute depth.

Stats per model: peak offset from r=0, half-width, entropy vs init.

Usage:  python -m scripts.interp_phirel [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = "/home/woody/iwi5/iwi5401h/vesuvius"
DEFAULT_RUNS = [
    (f"{W}/outputs", "a2bs_reg"), (f"{W}/outputs", "a2bs_t128"),
    (f"{W}/outputs", "lean_t192_s42"), (f"{W}/outputs", "lean_t192_s43"),
    (f"{W}/outputs", "lean_t192_s44"),
    (f"{W}/outputs4", "t256_lean_s42"), (f"{W}/outputs4", "t256_lean_s43"),
    (f"{W}/outputs4", "t256_lean_s44"), (f"{W}/outputs4", "t256_nosurf_s42"),
]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{W}/output_inter")
    args = ap.parse_args()
    out_dir = os.path.join(args.out, "phirel")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    stats = []
    for root, run in DEFAULT_RUNS:
        p = os.path.join(root, run, "best.pt")
        if not os.path.exists(p):
            print(f"[skip] {run}"); continue
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if "collapse.phi_rel" in sd:
            phi = sd["collapse.phi_rel"].float().numpy()
            L = len(phi); D = (L + 1) // 2
            grid = np.arange(L) - (D - 1)
            w = softmax(phi)
            init = softmax(-0.5 * (grid / max(1.0, D / 4.0)) ** 2)
            axes[0].plot(grid, w, label=run, alpha=0.8)
            if run == DEFAULT_RUNS[2][1]:
                axes[0].plot(grid, init, "k--", lw=1, label="Gaussian init")
            peak = float(grid[w.argmax()])
            ent = float(-(w * np.log(w + 1e-12)).sum() / np.log(L))
            ent0 = float(-(init * np.log(init + 1e-12)).sum() / np.log(L))
            stats.append({"run": run, "kind": "surface_relative", "peak_r": peak,
                          "entropy": ent, "entropy_init": ent0,
                          "mass_pm4": float(w[(grid >= -4) & (grid <= 4)].sum())})
            print(f"[{run}] surfrel: peak r={peak:+.0f}  entropy {ent:.3f} (init {ent0:.3f})  "
                  f"mass|r|<=4: {w[(grid>=-4)&(grid<=4)].sum():.2f}")
        elif "collapse.phi" in sd:
            phi = sd["collapse.phi"].float().numpy()
            D = len(phi)
            w = softmax(phi)
            axes[1].plot(np.arange(D), w, label=run, alpha=0.8)
            peak = int(w.argmax())
            stats.append({"run": run, "kind": "learned_abs", "peak_d": peak,
                          "entropy": float(-(w * np.log(w + 1e-12)).sum() / np.log(D))})
            print(f"[{run}] learned-abs: peak d={peak} (window coord; surface ~13-15)")
        else:
            print(f"[{run}] no collapse weights (mode mean?)")

    axes[0].set_xlabel("relative depth r = d - s_hat"); axes[0].set_ylabel("readout weight")
    axes[0].set_title("surface-relative collapse (learned vs init)"); axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("absolute window depth d"); axes[1].set_title("learned absolute collapse")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phi_curves.png"), dpi=130, bbox_inches="tight")
    with open(os.path.join(out_dir, "phi_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"-> {out_dir}/phi_curves.png")


if __name__ == "__main__":
    main()
