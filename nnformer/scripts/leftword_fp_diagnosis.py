"""Left-word FP diagnosis on VAL 20241113090990 (2026-07-20, per Omar).

Claim under test: all models show FPs just ABOVE the left word, same place ->
suspected residual local label misregistration (the val warp was dy -54..-22
varying along x; steepest at the left edge, with dx +10 measured at x~1250).

Method (the project's validated prob-correlation registration, made LOCAL):
  A. per x-window: dy* = argmax_dy mean(prob | ink_reg shifted by dy)
     for 4 independent checkpoints (3x UNETR++, 1x nnFormer).
     Labels correct <=> dy* ~ 0 in every window for every model.
  B. FP anatomy in the left region (x in [1434, 5200]):
     - cross-model FP-mask IoU (same place across models?)
     - signed vertical offset of each FP pixel to the nearest label pixel in
       its column (negative = FP sits ABOVE the label)
     - overlap with the known bad mask strip x < 1434.
  C. overlay figure crops for visual confirmation.

Outputs: docs/leftword_diagnosis.json, docs/figs/leftword_fp_*.png
"""
import json
import os

import numpy as np

VAL = "20241113090990"
PROC = f"/home/woody/iwi5/iwi5401h/vesuvius/data/proc/{VAL}"
DOCS = "/home/woody/iwi5/iwi5401h/nnFormer/docs"
os.makedirs(os.path.join(DOCS, "figs"), exist_ok=True)

MODELS = {  # run label -> (prob path, postprocess json for tau)
    "unetrpp_c1_t256": "/home/woody/iwi5/iwi5401h/vesuvius/outputs5/c1_t256_s42",
    "unetrpp_t256_lean": "/home/woody/iwi5/iwi5401h/vesuvius/outputs4/t256_lean_s42",
    "unetrpp_a2bs_t192": "/home/woody/iwi5/iwi5401h/vesuvius/outputs/a2bs_t192",
    "nnformer_n_base": "/home/woody/iwi5/iwi5401h/nnFormer/outputs/n_base_t192_s42",
}

ink = np.load(os.path.join(PROC, "ink_reg.npy")) > 0
mask = np.load(os.path.join(PROC, "mask.npy")) > 0
H, W = ink.shape
print(f"val {VAL}: H={H} W={W} ink%={ink.mean()*100:.1f}")

probs, taus = {}, {}
for name, d in MODELS.items():
    p = np.load(os.path.join(d, "val_prob_best.npy"))
    assert p.shape == ink.shape, (name, p.shape)
    probs[name] = p.astype(np.float32)
    try:
        pp = json.load(open(os.path.join(d, "postprocess.json")))
        taus[name] = float(pp["postprocessed"]["tau"])
    except Exception:
        taus[name] = float(json.load(open(os.path.join(d, "summary.json")))["best_tau"])

# ---------- A. local dy map (prob-correlation registration, per x-window) ----
DYS = list(range(-60, 61, 2))
XWIN, XSTEP = 1500, 750
windows = [(x0, min(x0 + XWIN, W)) for x0 in range(0, W - XWIN // 2, XSTEP)]

def local_dy(prob, x0, x1):
    sub_p = prob[:, x0:x1]
    sub_i = ink[:, x0:x1]
    sub_m = mask[:, x0:x1]
    if sub_i.sum() < 5000:
        return None, None
    best, bdy = -1.0, 0
    for dy in DYS:
        s = np.roll(sub_i, dy, axis=0) & sub_m
        if not s.any():
            continue
        v = float(sub_p[s].mean())
        if v > best:
            best, bdy = v, dy
    # refine +-1
    for dy in (bdy - 1, bdy + 1):
        s = np.roll(sub_i, dy, axis=0) & sub_m
        v = float(sub_p[s].mean()) if s.any() else -1
        if v > best:
            best, bdy = v, dy
    s0 = ink[:, x0:x1] & sub_m
    return bdy, round(best / max(float(sub_p[s0].mean()), 1e-6), 3)  # gain vs dy=0

print("\n[A] local dy* per x-window (dy*=0 -> labels sit right; gain = peak/dy0)")
dymap = {}
hdr = "x-window        " + "".join(f"{n[:14]:>17}" for n in MODELS)
print(hdr)
for (x0, x1) in windows:
    row = f"[{x0:5d},{x1:5d}) "
    vals = {}
    for name in MODELS:
        dy, gain = local_dy(probs[name], x0, x1)
        vals[name] = (dy, gain)
        row += f"  {'—' if dy is None else f'{dy:+3d} (x{gain})':>15}"
    dymap[f"{x0}-{x1}"] = vals
    print(row)

# ---------- B. FP anatomy in the left region --------------------------------
XL0, XL1 = 1434, 5200
fp_masks = {}
print(f"\n[B] FP anatomy, left region x in [{XL0},{XL1}), per-model post tau")
for name in MODELS:
    fp = (probs[name] >= taus[name]) & (~ink) & mask
    fp_masks[name] = fp
    sub = fp[:, XL0:XL1]
    strip = fp[:, :XL0]
    # signed vertical offset to nearest label pixel per column (sampled)
    offs = []
    cols = np.where(sub.any(axis=0))[0][::5]
    for c in cols:
        fpy = np.where(sub[:, c])[0]
        iky = np.where(ink[:, XL0 + c])[0]
        if len(iky) == 0 or len(fpy) == 0:
            continue
        for y in fpy[::3]:
            j = iky[np.argmin(np.abs(iky - y))]
            offs.append(int(y - j))
    offs = np.array(offs) if offs else np.array([0])
    above = float((offs < 0).mean())
    print(f"  {name:<20} tau={taus[name]:.2f}  FP_left={int(sub.sum()):7d}px "
          f"FP_strip(x<{XL0})={int(strip.sum()):7d}px  "
          f"FP-above-label frac={above:.2f}  median|off|={int(np.median(np.abs(offs)))}px")

names = list(MODELS)
print("  cross-model FP IoU (left region):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = fp_masks[names[i]][:, XL0:XL1]
        b = fp_masks[names[j]][:, XL0:XL1]
        iou = float((a & b).sum()) / max(float((a | b).sum()), 1)
        print(f"    {names[i][:16]:<18} vs {names[j][:16]:<18} IoU={iou:.3f}")

# ---------- C. overlay crops -------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y0, y1 = 0, H
    for name in ("unetrpp_c1_t256", "nnformer_n_base"):
        fig, ax = plt.subplots(figsize=(16, 4))
        crop_p = probs[name][y0:y1, XL0:XL1]
        crop_i = ink[y0:y1, XL0:XL1]
        crop_fp = fp_masks[name][y0:y1, XL0:XL1]
        rgb = np.stack([crop_p, crop_p, crop_p], -1)
        rgb[crop_i] = [0, 0.8, 0]          # labels green
        rgb[crop_fp] = [1, 0, 0]           # FPs red
        ax.imshow(rgb, aspect="auto")
        ax.set_title(f"{name}  (green=ink_reg label, red=FP@tau{taus[name]:.2f}, gray=prob)  x[{XL0},{XL1})")
        out = os.path.join(DOCS, "figs", f"leftword_fp_{name}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("fig ->", out)
except Exception as e:
    print("figure skipped:", e)

json.dump({"windows": {k: {n: v for n, v in vv.items()} for k, vv in dymap.items()},
           "taus": taus},
          open(os.path.join(DOCS, "leftword_diagnosis.json"), "w"), indent=1, default=str)
print("->", os.path.join(DOCS, "leftword_diagnosis.json"))
