"""
Figure 2: Ladder qualitative panel.
Shows the same validation crop at three stages: a0 (blobs) → a0s (letterforms emerge) → a1 (sharper).
Each column: volume slab | GT | probability map | binary prediction.
Crop: right-word region y0=1088 x0=7800 h=320 w=640 (letterform-dense, good mask coverage).
Output: plan(md_files)/fig_ladder_qualitative.png
"""
import sys, os
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data.segments import proc_dir

SEG = "20241113090990"
d   = proc_dir(SEG)

vol  = np.load(f"{d}/vol.npy",     mmap_mode="r")
ireg = np.load(f"{d}/ink_reg.npy")
mask = np.load(f"{d}/mask.npy",    mmap_mode="r")

# ── crop coordinates ─────────────────────────────────────────────────────────
# Right-word area where letterforms are clearly resolved by a0s/a1
# but a0 shows only blobs — chosen to maximise visual contrast between rungs.
CY0, CX0, CH, CW = 1088, 7800, 320, 640

def crop(arr):
    return np.asarray(arr[CY0:CY0+CH, CX0:CX0+CW])

slab_vol  = vol[28:38, CY0:CY0+CH, CX0:CX0+CW].mean(0).astype(np.float32)
gt        = crop(ireg) > 0
mk        = crop(mask)  > 0

# ── runs ─────────────────────────────────────────────────────────────────────
RUNS = [
    ("a0",     "a0\n(t64, baseline)",     0.40),
    ("a0s",    "a0s\n(+ stem & collapse)",0.45),
    ("a1",     "a1\n(+ depth attention)", 0.45),
    ("a1_reg", "a1_reg\n(+ registered labels)", 0.55),
]

probs = {}
for run, _, _ in RUNS:
    p = f"../outputs/{run}/val_prob_best.npy"
    probs[run] = crop(np.load(p, mmap_mode="r"))

# ── figure ───────────────────────────────────────────────────────────────────
COLS = ["Volume\n(mean z28–38)", "GT (reg)", "Probability map", "Prediction (τ)"]
nrows = len(RUNS)
ncols = len(COLS)

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 2.8, nrows * 2.4),
                         gridspec_kw=dict(hspace=0.04, wspace=0.04))

vmin_vol, vmax_vol = np.percentile(slab_vol[mk], [2, 98])

for r, (run, label, tau) in enumerate(RUNS):
    prob  = probs[run]
    pred  = (prob > tau) & mk
    
    for c, data in enumerate([slab_vol, gt.astype(float), prob, pred.astype(float)]):
        ax = axes[r][c]
        if c == 0:                                # grey volume
            ax.imshow(data, cmap="gray", vmin=vmin_vol, vmax=vmax_vol,
                      interpolation="bicubic")
        elif c == 1:                              # GT: binary white-on-black
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        elif c == 2:                              # probability: hot colourmap
            ax.imshow(data, cmap="inferno", vmin=0, vmax=1)
        else:                                     # prediction: white-on-black
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)

        ax.axis("off")

        # row label: draw as figure text anchored to the leftmost axis
        if c == 0:
            # get axis position in figure coordinates
            ax.text(-0.04, 0.5, label,
                    transform=ax.transAxes,
                    fontsize=8.5, va="center", ha="right",
                    fontweight="semibold",
                    bbox=dict(fc="white", ec="none", pad=2))
        # tau annotation on prediction column
        if c == 3:
            ax.text(0.98, 0.03, f"τ = {tau}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5,
                    color="white", bbox=dict(fc="black", alpha=0.45, pad=1.5))

    # F0.5 annotation right-most column
    f05_map = {"a0": 0.446, "a0s": 0.526, "a1": 0.627, "a1_reg": 0.637}
    axes[r][-1].text(0.02, 0.97, f"F₀.₅={f05_map[run]:.3f}",
                     transform=axes[r][-1].transAxes,
                     ha="left", va="top", fontsize=7.5,
                     color="yellow", bbox=dict(fc="black", alpha=0.45, pad=1.5))

# column headers
for c, hdr in enumerate(COLS):
    axes[0][c].set_title(hdr, fontsize=9.5, pad=4)

# overall title
fig.suptitle(
    "Validation crop  (seg 090990, y=1088–1408, x=7800–8440)  ·  "
    "right-word region  ·  same crop across all runs",
    fontsize=9, y=1.00)

# ── save ─────────────────────────────────────────────────────────────────────
OUT = "/home/hpc/iwi5/iwi5401h/plan(md_files)/fig_ladder_qualitative.png"
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
