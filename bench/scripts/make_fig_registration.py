"""
Figure 1: Label registration proof.
Left panel:  contrast-vs-shift curve (detrended slab contrast as a function of dy)
             for the segment with the clearest signal (20241025145701, dy_best=-35).
Right panel: before/after contour overlay on one 256×512 crop of segment 090990
             (the val segment, dy varies -54..-22 along x).
Output: plan(md_files)/fig_registration_proof.png
"""
import sys, os
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import measure
from data.segments import proc_dir

# ── helpers ─────────────────────────────────────────────────────────────────
def detrended_slab(vol, mask, zlo, zhi, x0, x1):
    slab = np.asarray(vol[zlo:zhi, :, x0:x1], dtype=np.float32).mean(axis=0)
    mk = (np.asarray(mask[:, x0:x1]) > 0) & (slab > 10)
    s = slab.copy()
    s[~mk] = np.nan
    s -= np.nanmedian(s, axis=1, keepdims=True)
    s -= np.nanmedian(s, axis=0, keepdims=True)
    return s, mk

def contrast(s, mk, ik_patch):
    sel_in  = ik_patch & mk
    sel_out = (~ik_patch) & mk
    if sel_in.sum() < 500 or sel_out.sum() < 500:
        return np.nan
    return float(np.nanmean(s[sel_in]) - np.nanmean(s[sel_out]))

# ── segment for contrast-vs-shift curve: 145701 ─────────────────────────────
SEG_CURVE = "20241025145701"
d = proc_dir(SEG_CURVE)
vol  = np.load(f"{d}/vol.npy",  mmap_mode="r")
ink  = np.load(f"{d}/ink.npy")
mask = np.load(f"{d}/mask.npy", mmap_mode="r")
ireg = np.load(f"{d}/ink_reg.npy")

zlo, zhi = 28, 38                     # best band from reg_report
x0, x1   = 0, min(vol.shape[2], 6000)
s, mk = detrended_slab(vol, mask, zlo, zhi, x0, x1)
ik    = ink[:, x0:x1] > 0

dy_range = np.arange(-70, 16, 3)
contrasts = []
for dy in dy_range:
    ik_shifted = np.roll(ik, int(dy), axis=0)
    contrasts.append(contrast(s, mk, ik_shifted))
contrasts = np.array(contrasts)

# ── segment for before/after overlay: 090990 (val, large misreg) ────────────
SEG_OVERLAY = "20241113090990"
d2 = proc_dir(SEG_OVERLAY)
vol2  = np.load(f"{d2}/vol.npy",  mmap_mode="r")
ink2  = np.load(f"{d2}/ink.npy")
ireg2 = np.load(f"{d2}/ink_reg.npy")
mask2 = np.load(f"{d2}/mask.npy", mmap_mode="r")

# Crop: central-right area with clear letterforms and visible misreg
# dy at this region is ~-30 (x~8000) so offset is clearly visible
CY0, CX0, CH, CW = 700, 7500, 256, 512
slab_mid = np.asarray(vol2[28:38, CY0:CY0+CH, CX0:CX0+CW], dtype=np.float32).mean(0)
ink_crop  = ink2 [CY0:CY0+CH, CX0:CX0+CW] > 0
ireg_crop = ireg2[CY0:CY0+CH, CX0:CX0+CW] > 0

# ── figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4.5))
gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.4], wspace=0.08)

# ─── Panel A: contrast vs dy ─────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
valid = ~np.isnan(contrasts)
ax0.plot(dy_range[valid], contrasts[valid], color="#1f77b4", lw=2.0, zorder=3)
ax0.axvline(-35, color="#d62728", lw=1.5, ls="--", label="dy = −35 (estimated)")
ax0.axhline(0,   color="gray",    lw=0.8, ls=":")
ax0.fill_betweenx([contrasts[valid].min()-0.5, contrasts[valid].max()+0.5],
                  -47, -32, alpha=0.12, color="#d62728", label="per-column dy range")
ax0.set_xlabel("Label shift  dy  (pixels)", fontsize=11)
ax0.set_ylabel("Detrended ink-vs-papyrus contrast", fontsize=10)
ax0.set_title("(a)  Contrast-vs-shift scan\nsegment 145701, z-band [28,38]", fontsize=10)
ax0.legend(fontsize=8.5, framealpha=0.85)
ax0.set_xlim(dy_range[0]-2, dy_range[-1]+2)
ax0.grid(True, lw=0.4, alpha=0.5)
ax0.set_facecolor("#fafafa")

# ─── Panels B/C: before/after contour overlay ────────────────────────────────
gs2 = gs[1].subgridspec(1, 2, wspace=0.03)

for col, (ik_patch, title) in enumerate([
        (ink_crop,  "(b)  Before registration\n(original ink.npy)"),
        (ireg_crop, "(c)  After registration\n(ink_reg.npy)")]):
    ax = fig.add_subplot(gs2[col])
    # show mean-slab volume as grey background
    vmin, vmax = np.percentile(slab_mid, [2, 98])
    ax.imshow(slab_mid, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bicubic")
    # draw label contour
    for contour in measure.find_contours(ik_patch.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0],
                color="#ff4444" if col == 0 else "#00cc66",
                lw=0.9, alpha=0.9)
    ax.set_title(title, fontsize=9.5)
    ax.axis("off")

# shared legend for overlay panels
red_patch   = mpatches.Patch(color="#ff4444", label="label contour (orig)")
green_patch = mpatches.Patch(color="#00cc66", label="label contour (registered)")
fig.legend(handles=[red_patch, green_patch],
           loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.72, -0.02), framealpha=0.9)

# ── save ────────────────────────────────────────────────────────────────────
OUT = "/home/hpc/iwi5/iwi5401h/plan(md_files)/fig_registration_proof.png"
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
