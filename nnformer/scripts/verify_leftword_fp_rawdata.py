"""Raw-data verification of FP blobs IN/AROUND the left word (2026-07-20, per Omar).

Left word = the labeled component cluster at x in [1434,5300] (components 8,9,
11,12 of ink_reg; matches scripts/leftword_fp_diagnosis.py's crop). Unlike the
upper-band FP-region script (isolated blobs only), this one takes EVERY FP
blob touching or near the word -- including thin boundary slivers -- and asks
two questions per blob:
  (a) is it EXPLAINED by the already-validated re-registration field (i.e. it
      becomes a true positive under ink_reg2 -> it's a registration artifact,
      not a real model error or missing annotation)?
  (b) if NOT explained by reg2, is it ink-like by raw CT contrast (component
      mean intensity minus its 8px ring, z26-38 slab -- identical formula/
      threshold used everywhere else in this project)?

Every remaining_fp component gets: size, distance-to-nearest-label, contrast,
reg2 status, per-model probability, and a z-slice figure for the largest ones.

Outputs: docs/leftword_blob_verification.json + docs/figs/lwblob_*.png
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

XL0, XL1 = 1434, 5300     # the left-word crop (leftword_fp_diagnosis.py convention)
CONTRAST_THRESH = 3.0
RING_ITERS = 8
BOUNDARY_PX = 6            # blobs whose min-distance-to-label <= this = "boundary sliver"
MIN_BLOB_PX = 150          # small min -- we want the thin slivers too

ink = np.load(os.path.join(PROC, "ink_reg.npy")) > 0
ink2 = np.load(os.path.join(PROC, "ink_reg2.npy")) > 0
mask = np.load(os.path.join(PROC, "mask.npy")) > 0
vol = np.load(os.path.join(PROC, "vol.npy"), mmap_mode="r")
H, W = ink.shape
print(f"left word crop x[{XL0},{XL1})  ink px in crop: {int(ink[:, XL0:XL1].sum())}")

probs = {n: np.load(p).astype(np.float32) for n, p in MAPS.items()}
for n, p in probs.items():
    assert p.shape == ink.shape, n

# --- consensus FP (>=2/3 models), crop to the left-word x-range -------------
votes = sum((probs[n] >= TAUS[n]).astype(np.uint8) for n in MAPS)
fp_full = (votes >= 2) & (~ink) & mask
fp = np.zeros_like(fp_full)
fp[:, XL0:XL1] = fp_full[:, XL0:XL1]
print(f"consensus FP px in left-word crop: {int(fp.sum())}")

explained_by_reg2 = fp & ink2                # this pixel IS ink under the corrected label
remaining = fp & ~ink2
print(f"  explained by reg2 field (registration artifact): {int(explained_by_reg2.sum())} px "
      f"({explained_by_reg2.sum()/max(fp.sum(),1)*100:.1f}%)")
print(f"  remaining (not explained by better registration): {int(remaining.sum())} px "
      f"({remaining.sum()/max(fp.sum(),1)*100:.1f}%)")

dt_out = ndimage.distance_transform_edt(~ink)   # distance from every px to nearest ORIGINAL label


def contrast_and_width(comp_mask, obj_slice):
    slab = np.asarray(vol[26:38], dtype=np.float32).mean(0)
    pad = tuple(slice(max(0, s.start - 12), min(comp_mask.shape[i], s.stop + 12))
                for i, s in enumerate(obj_slice))
    sub = comp_mask[pad]
    ring = ndimage.binary_dilation(sub, iterations=RING_ITERS) & ~sub & mask[pad] & ~ink[pad]
    c = float(slab[pad][sub].mean() - slab[pad][ring].mean()) if ring.sum() else float("nan")
    dt = ndimage.distance_transform_edt(sub)
    width = float(np.median(dt[sub]) * 2) if sub.any() else 0.0
    return c, width


# --- connected components of the REMAINING (unexplained-by-reg2) FP mass ---
lab, n = ndimage.label(remaining, structure=np.ones((3, 3)))
objs = ndimage.find_objects(lab)
sizes = np.bincount(lab.ravel())[1:]
order = np.argsort(sizes)[::-1]
blobs = []
for k in order:
    if sizes[k] < MIN_BLOB_PX:
        break
    idx = k + 1
    sl = objs[k]
    comp = (lab == idx)
    c, wid = contrast_and_width(comp, sl)
    mind = float(dt_out[comp].min())
    yc, xc = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
    per_model = {name: round(float(probs[name][comp].mean()), 3) for name in MAPS}
    blobs.append({"x": int(xc), "y": int(yc), "area_px": int(sizes[k]),
                  "min_dist_to_label_px": round(mind, 1),
                  "kind": "boundary_sliver" if mind <= BOUNDARY_PX else "separate_blob",
                  "contrast": round(c, 2) if not np.isnan(c) else None,
                  "ink_like (contrast>=3.0)": bool(c >= CONTRAST_THRESH) if not np.isnan(c) else None,
                  "median_width_px": round(wid, 1),
                  "prob_per_model": per_model})

print(f"\n[components] {len(blobs)} remaining FP blobs (>= {MIN_BLOB_PX}px, not reg2-explained):")
n_sliver = sum(1 for b in blobs if b["kind"] == "boundary_sliver")
n_sep = len(blobs) - n_sliver
print(f"  {n_sliver} boundary slivers (<= {BOUNDARY_PX}px from a label) | {n_sep} separate blobs")
for b in blobs[:25]:
    print(" ", b)

# reference (same as before, recomputed for the crop's own labeled ink)
ilab, ni = ndimage.label(ink[:, XL0:XL1], structure=np.ones((3, 3)))
isizes = np.bincount(ilab.ravel())[1:]
iobjs = ndimage.find_objects(ilab)
real_c = []
for k in np.argsort(isizes)[::-1][:20]:
    if isizes[k] < MIN_BLOB_PX:
        break
    sl = iobjs[k]
    sl_full = (sl[0], slice(sl[1].start + XL0, sl[1].stop + XL0))
    comp_full = np.zeros_like(ink)
    comp_full[sl_full] = (ilab[sl] == (k + 1))
    c, _ = contrast_and_width(comp_full, sl_full)
    if not np.isnan(c):
        real_c.append(c)
real_c = np.array(real_c)
print(f"\n[reference] left-word real-ink contrast: mean={real_c.mean():.2f} median={np.median(real_c):.2f}")

# --- figures: largest separate blobs + largest boundary slivers ------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ZS = [20, 24, 28, 30, 32, 34, 36, 40, 44]
    sep_blobs = [b for b in blobs if b["kind"] == "separate_blob"][:2]
    sliver_blobs = [b for b in blobs if b["kind"] == "boundary_sliver"][:2]
    for tag, bl in (("sep", sep_blobs), ("sliver", sliver_blobs)):
        for bi, b in enumerate(bl):
            x0, y0 = b["x"], b["y"]
            pad = 180
            y_lo, y_hi = max(0, y0 - pad), min(H, y0 + pad)
            x_lo, x_hi = max(0, x0 - pad), min(W, x0 + pad)
            blob_mask = (lab[y_lo:y_hi, x_lo:x_hi] > 0)
            ink_crop = ink[y_lo:y_hi, x_lo:x_hi]
            ink2_crop = ink2[y_lo:y_hi, x_lo:x_hi]
            fig, axes = plt.subplots(1, len(ZS), figsize=(3 * len(ZS), 3.6))
            for ax, z in zip(axes, ZS):
                sl_img = np.asarray(vol[z, y_lo:y_hi, x_lo:x_hi], dtype=np.float32)
                ax.imshow(sl_img, cmap="gray", vmin=0, vmax=255)
                if ink_crop.any():
                    ax.contour(ink_crop, levels=[0.5], colors="lime", linewidths=1.0)
                if ink2_crop.any():
                    ax.contour(ink2_crop, levels=[0.5], colors="cyan", linewidths=0.8, linestyles="dashed")
                ax.contour(blob_mask, levels=[0.5], colors="red", linewidths=1.2)
                ax.set_title(f"z={z}", fontsize=9)
                ax.axis("off")
            fig.suptitle(f"{tag} blob @ (x={x0},y={y0}) area={b['area_px']}px contrast={b['contrast']} "
                         f"dist={b['min_dist_to_label_px']}px  "
                         f"(green=ink_reg, cyan dashed=ink_reg2, red=FP outline)")
            out = os.path.join(DOCS, "figs", f"lwblob_{tag}{bi}_x{x0}_y{y0}.png")
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print("fig ->", out)

    # word-wide overview: reg2-explained (yellow) vs remaining unexplained (red) vs label (green)
    fig, ax = plt.subplots(figsize=(16, 5))
    base = np.asarray(vol[30, :, XL0:XL1], dtype=np.float32)
    rgb = np.stack([base, base, base], -1) / 255.0
    rgb[ink[:, XL0:XL1]] = [0, 0.8, 0]
    rgb[explained_by_reg2[:, XL0:XL1]] = [1, 0.9, 0]
    rgb[remaining[:, XL0:XL1]] = [1, 0, 0]
    ax.imshow(rgb, aspect="auto")
    ax.set_title("left word: green=ink_reg label, yellow=FP explained by reg2 field "
                 "(registration artifact), red=FP unexplained (needs raw-data check)")
    out = os.path.join(DOCS, "figs", "lwblob_overview.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig ->", out)
except Exception as e:
    print("figures skipped:", e)

json.dump({"crop_x": [XL0, XL1],
           "fp_total_px": int(fp.sum()),
           "explained_by_reg2_px": int(explained_by_reg2.sum()),
           "remaining_px": int(remaining.sum()),
           "n_boundary_slivers": n_sliver, "n_separate_blobs": n_sep,
           "blobs": blobs,
           "reference_real_ink_contrast": {"mean": float(real_c.mean()),
                                           "median": float(np.median(real_c))}},
          open(os.path.join(DOCS, "leftword_blob_verification.json"), "w"), indent=1)
print("\n->", os.path.join(DOCS, "leftword_blob_verification.json"))
