"""Raw-data verification of the circled FP region on VAL 20241113090990
(2026-07-20, per Omar's annotated screenshot: dense FP cluster upper-left/mid).

Question: is this false positive cluster (a) real unlabeled ink, (b) a labeling
gap around already-labeled letterforms (ascenders/diacritics), or (c) a model
artifact (crackle/noise misread as ink)? Answered from RAW CT data only, using
the project's OWN validated ink-vs-substrate contrast metric (identical to
scripts/interp_error_anatomy.py: component mean intensity minus its 8px-ring
background mean, on the z26-38 central slab) — so the number is directly
comparable to the "contrast>=3.0 = real ink" threshold already used to explain
FN components in this segment.

Method:
  1. Cross-architecture consensus FP mask: pixels where >=2 of 3 independent
     checkpoints (UNETR++ c1_t256, UNETR++ t256_lean, nnFormer n_base after
     reg2 retrain) agree prob>=0.5, AND outside a 24px dilation of ink_reg
     (isolated from any labeled stroke) AND inside the segment mask.
  2. Connected components of that consensus-FP mask -> the actual "blobs".
  3. For each: contrast (same formula/threshold as error anatomy), median
     width, area, and per-model probability.
  4. Multi-z-slice raw-intensity crops (z=20,24,28,30,32,34,36,40,44) for the
     3 largest blobs, WITH the existing ink_reg label contour and the blob's
     own contour overlaid, so ink vs no-ink is checked visually across depth
     (letters should show a a stable crackle/thickening band around z28-34;
     an artifact should not).
  5. Reference bars: contrast distribution of REAL labeled ink components in
     the same segment (for calibration) and of the documented dead-label
     components (x~6177/9724, contrast should be near 0, sanity check).
  6. Symmetric FN check (per Omar): sample the top FN components (already
     labeled, model misses them) and confirm their contrast/depth profile
     looks like real ink (i.e. the label is right, the model is just
     under-confident) rather than a phantom label.

Outputs: nnFormer/docs/fp_region_verification.json + docs/figs/fpregion_*.png
"""
import json
import os

import numpy as np
from scipy import ndimage

WORK = "/home/woody/iwi5/iwi5401h"
PROC = f"{WORK}/vesuvius/data/proc/20241113090990"
DOCS = f"{WORK}/nnFormer/docs"
os.makedirs(os.path.join(DOCS, "figs"), exist_ok=True)

MAPS = {
    "unetrpp_c1_t256": f"{WORK}/vesuvius/outputs5/c1_t256_s42/val_prob_best.npy",
    "unetrpp_t256_lean": f"{WORK}/vesuvius/outputs4/t256_lean_s42/val_prob_best.npy",
    "nnformer_n_base_reg2": f"{WORK}/nnFormer/outputs/n_base_t192_s42/val_prob_best.npy",
}
TAUS = {"unetrpp_c1_t256": 0.70, "unetrpp_t256_lean": 0.65, "nnformer_n_base_reg2": 0.80}

CONTRAST_THRESH = 3.0     # identical to interp_error_anatomy.py
DEAD_PROB = 0.07
RING_ITERS = 8
ISOLATION_PX = 24         # min distance from any labeled ink to count as "isolated FP"
MIN_BLOB_PX = 3000

ink = np.load(os.path.join(PROC, "ink_reg.npy")) > 0
mask = np.load(os.path.join(PROC, "mask.npy")) > 0
vol = np.load(os.path.join(PROC, "vol.npy"), mmap_mode="r")   # [65, H, W]
H, W = ink.shape
print(f"val segment: H={H} W={W} ink%={ink.mean()*100:.2f} mask%={mask.mean()*100:.1f}")

probs = {}
for name, p in MAPS.items():
    arr = np.load(p).astype(np.float32)
    assert arr.shape == ink.shape, (name, arr.shape)
    probs[name] = arr

# --- consensus FP mask (>=2 of 3 models agree, isolated from labels) --------
votes = sum((probs[n] >= TAUS[n]).astype(np.uint8) for n in MAPS)
consensus_fp = (votes >= 2) & (~ink) & mask
halo = ndimage.binary_dilation(ink, iterations=ISOLATION_PX)
isolated_fp = consensus_fp & ~halo
print(f"consensus (>=2/3 models) FP px: {int(consensus_fp.sum())}; "
      f"isolated (>{ISOLATION_PX}px from any label): {int(isolated_fp.sum())}")


def contrast_and_width(mask2d, obj_slice=None):
    """Exact interp_error_anatomy.py formula: mean(component) - mean(8px ring),
    on the z26-38 central slab; width = 2*median(distance-to-edge)."""
    slab = np.asarray(vol[26:38], dtype=np.float32).mean(0)
    if obj_slice is None:
        obj_slice = tuple(slice(0, s) for s in mask2d.shape)
    pad = tuple(slice(max(0, s.start - 12), min(mask2d.shape[i], s.stop + 12))
                for i, s in enumerate(obj_slice))
    sub = mask2d[pad]
    ring = ndimage.binary_dilation(sub, iterations=RING_ITERS) & ~sub & mask[pad] & ~ink[pad]
    c = float(slab[pad][sub].mean() - slab[pad][ring].mean()) if ring.sum() else float("nan")
    dt = ndimage.distance_transform_edt(sub)
    width = float(np.median(dt[sub]) * 2) if sub.any() else 0.0
    return c, width


# --- 1. characterize the isolated-FP blobs ----------------------------------
lab, n = ndimage.label(isolated_fp, structure=np.ones((3, 3)))
objs = ndimage.find_objects(lab)
sizes = np.bincount(lab.ravel())[1:]
order = np.argsort(sizes)[::-1]
blob_report = []
for k in order:
    if sizes[k] < MIN_BLOB_PX:
        break
    idx = k + 1
    sl = objs[k]
    comp = lab[sl] == idx
    c, wid = contrast_and_width(lab == idx, sl)
    yc = (sl[0].start + sl[0].stop) / 2
    xc = (sl[1].start + sl[1].stop) / 2
    per_model = {name: round(float(probs[name][lab == idx].mean()), 3) for name in MAPS}
    blob_report.append({"x": int(xc), "y": int(yc), "area_px": int(sizes[k]),
                        "contrast": round(c, 2), "median_width_px": round(wid, 1),
                        "is_ink_like (contrast>=3.0)": bool(c >= CONTRAST_THRESH),
                        "prob_per_model": per_model})
print(f"\n[1] isolated consensus-FP blobs (>= {MIN_BLOB_PX} px): {len(blob_report)}")
for b in blob_report[:15]:
    print(" ", b)

# --- 2. reference: contrast of REAL labeled-ink components ------------------
ilab, ni = ndimage.label(ink, structure=np.ones((3, 3)))
isizes = np.bincount(ilab.ravel())[1:]
iobjs = ndimage.find_objects(ilab)
real_contrasts = []
for k in np.argsort(isizes)[::-1][:40]:
    if isizes[k] < MIN_BLOB_PX:
        break
    c, _ = contrast_and_width(ilab == (k + 1), iobjs[k])
    real_contrasts.append(c)
real_contrasts = np.array(real_contrasts)
print(f"\n[2] reference: top-40 REAL ink components contrast "
      f"mean={real_contrasts.mean():.2f} median={np.median(real_contrasts):.2f} "
      f"min={real_contrasts.min():.2f}")

# --- 3. sanity check on the two documented dead-label components -----------
print("\n[3] sanity check — documented dead-label components (should read ~0 contrast, low prob):")
for x0, y0 in [(6177, 557), (9724, 1046)]:
    win = (slice(max(0, y0 - 100), y0 + 100), slice(max(0, x0 - 150), x0 + 150))
    comp = ink[win]
    if comp.sum() < 100:
        print(f"  ({x0},{y0}): no ink component found in window (unexpected)")
        continue
    c, _ = contrast_and_width(comp, win)
    pm = {name: round(float(probs[name][win][comp].mean()), 3) for name in MAPS}
    print(f"  ({x0},{y0}): contrast={c:.2f}  prob={pm}")

# --- 4. symmetric FN check: do the documented low-recall FN components ------
#        actually look like real ink (they should - the model is just faint)
print("\n[4] symmetric check — top FN (missed, but LABELED) components; expect contrast>=3 (real, just faint):")
pred_union = (probs["unetrpp_c1_t256"] >= TAUS["unetrpp_c1_t256"])
fn = ink & ~pred_union & mask
fn_ids = ilab[fn]
fn_counts = np.bincount(fn_ids[fn_ids > 0], minlength=ni + 1)
for k in np.argsort(fn_counts)[::-1][:6]:
    if fn_counts[k] < MIN_BLOB_PX or k == 0:
        continue
    sl = iobjs[k - 1]
    c, wid = contrast_and_width(ilab == k, sl)
    yc, xc = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
    print(f"  component@({xc},{yc}) fn_px={int(fn_counts[k])} contrast={c:.2f} width={wid:.1f}px")

# --- 5. multi-z-slice figures for the 3 largest FP blobs --------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ZS = [20, 24, 28, 30, 32, 34, 36, 40, 44]
    for bi, b in enumerate(blob_report[:3]):
        x0, y0 = b["x"], b["y"]
        pad = 220
        y_lo, y_hi = max(0, y0 - pad), min(H, y0 + pad)
        x_lo, x_hi = max(0, x0 - pad), min(W, x0 + pad)
        fig, axes = plt.subplots(1, len(ZS), figsize=(3 * len(ZS), 3.4))
        blob_mask = (lab[y_lo:y_hi, x_lo:x_hi] > 0)
        ink_crop = ink[y_lo:y_hi, x_lo:x_hi]
        for ax, z in zip(axes, ZS):
            sl = np.asarray(vol[z, y_lo:y_hi, x_lo:x_hi], dtype=np.float32)
            ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
            if ink_crop.any():
                ax.contour(ink_crop, levels=[0.5], colors="lime", linewidths=1.0)
            ax.contour(blob_mask, levels=[0.5], colors="red", linewidths=1.2)
            ax.set_title(f"z={z}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"FP blob #{bi} @ (x={x0},y={y0}) area={b['area_px']}px "
                     f"contrast={b['contrast']:.2f}  (green=ink_reg label, red=FP blob outline)")
        out = os.path.join(DOCS, "figs", f"fpregion_blob{bi}_x{x0}_y{y0}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("fig ->", out)

    # segment-wide overview: consensus-FP (red) vs labels (green) vs mask edge
    sub = 4
    fig, ax = plt.subplots(figsize=(18, 3))
    base = np.asarray(vol[30, ::sub, ::sub], dtype=np.float32)
    rgb = np.stack([base, base, base], -1) / 255.0
    rgb[ink[::sub, ::sub]] = [0, 0.8, 0]
    rgb[isolated_fp[::sub, ::sub]] = [1, 0, 0]
    ax.imshow(rgb, aspect="auto")
    ax.set_title("val segment overview (z=30): green=ink_reg label, red=isolated consensus-FP blobs")
    out = os.path.join(DOCS, "figs", "fpregion_overview.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", out)
except Exception as e:
    print("figures skipped:", e)

json.dump({"isolated_fp_px": int(isolated_fp.sum()),
           "blobs": blob_report,
           "reference_real_ink_contrast": {"mean": float(real_contrasts.mean()),
                                           "median": float(np.median(real_contrasts)),
                                           "min": float(real_contrasts.min())},
           "contrast_thresh_used": CONTRAST_THRESH},
          open(os.path.join(DOCS, "fp_region_verification.json"), "w"), indent=1)
print("\n->", os.path.join(DOCS, "fp_region_verification.json"))
