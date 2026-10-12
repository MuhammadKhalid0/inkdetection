"""Interpretability probe 1 (plan §14.4): error anatomy — decompose the
residual errors of a trained model into measurable strata, giving the
"attainable headroom" number. No model internals; CPU only.

FN strata (priority order — each pixel counted once):
  1. dead-label components : GT components with mean prob < 0.07 (probable
     label errors / destroyed ink — documented upstream candidates)
  2. boundary band         : FN within `edge_r` px of the stroke edge
                             (label-uncertainty zone; cf. the u1 trimap)
  3. low-contrast          : FN in components with ink-vs-ring contrast < 3.0
                             (of 255) in the central slab z26-38
  4. unexplained           : everything else  == the true attainable headroom

Also: FP anatomy (boundary FP within `edge_r` of GT vs isolated islands),
per-stratum recall, and a top-10 list of components by missing mass.

Usage:
    python -m scripts.interp_error_anatomy --root /...vesuvius/outputs4 --run t256_lean_s42
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import VAL_SEGMENT, proc_dir  # noqa: E402

W = "/home/woody/iwi5/iwi5401h/vesuvius"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=f"{W}/output_inter")
    ap.add_argument("--edge-r", type=int, default=4)
    ap.add_argument("--contrast-thresh", type=float, default=3.0)
    ap.add_argument("--dead-prob", type=float, default=0.07)
    args = ap.parse_args()

    rd = os.path.join(args.root, args.run)
    prob = np.load(os.path.join(rd, "val_prob_best.npy"))
    tau = float(json.load(open(os.path.join(rd, "summary.json")))["best_tau"])
    proc = proc_dir(VAL_SEGMENT)
    gt = np.load(os.path.join(proc, "ink_reg.npy")) > 0
    mask = np.load(os.path.join(proc, "mask.npy")) > 0
    vol = np.load(os.path.join(proc, "vol.npy"), mmap_mode="r")
    slab = np.asarray(vol[26:38], dtype=np.float32).mean(0)

    pred = (prob >= tau) & mask
    fn = gt & ~pred & mask
    fp = pred & ~gt

    # --- per-component properties ---------------------------------------
    lab, n = ndimage.label(gt, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    comp_prob = ndimage.mean(prob, lab, index=np.arange(1, n + 1))
    sizes = np.bincount(lab.ravel())[1:]
    dt_in = ndimage.distance_transform_edt(gt)          # distance to background
    contrast = np.zeros(n + 1, dtype=np.float32)
    width = np.zeros(n + 1, dtype=np.float32)
    for c in range(1, n + 1):
        sl = objs[c - 1]
        if sl is None: continue
        pad = tuple(slice(max(0, s.start - 12), s.stop + 12) for s in sl)
        sub = lab[pad] == c
        ring = ndimage.binary_dilation(sub, iterations=8) & ~sub & mask[pad] & (lab[pad] == 0)
        if ring.sum() > 0:
            contrast[c] = float(slab[pad][sub].mean() - slab[pad][ring].mean())
        width[c] = float(np.median(dt_in[pad][sub]) * 2)

    dead = np.zeros(n + 1, dtype=bool)
    dead[1:] = comp_prob < args.dead_prob
    lowc = np.zeros(n + 1, dtype=bool)
    lowc[1:] = contrast[1:] < args.contrast_thresh

    # --- stratify FN (priority order) ------------------------------------
    fn_ids = lab[fn]
    edge = dt_in[fn] <= args.edge_r
    s_dead = dead[fn_ids]
    s_edge = ~s_dead & edge
    s_lowc = ~s_dead & ~s_edge & lowc[fn_ids]
    s_unex = ~s_dead & ~s_edge & ~s_lowc

    n_gt = int(gt.sum()); n_fn = int(fn.sum())
    rec = 1 - n_fn / n_gt
    strata = {"dead_label_components": int(s_dead.sum()),
              f"boundary_band_<= {args.edge_r}px": int(s_edge.sum()),
              f"low_contrast_<{args.contrast_thresh}": int(s_lowc.sum()),
              "UNEXPLAINED (attainable headroom)": int(s_unex.sum())}

    # --- FP anatomy -------------------------------------------------------
    dt_out = ndimage.distance_transform_edt(~gt)
    fp_near = fp & (dt_out <= args.edge_r)
    lab_fp, n_fp_comp = ndimage.label(fp & (dt_out > args.edge_r), structure=np.ones((3, 3)))
    fp_sizes = np.bincount(lab_fp.ravel())[1:]

    rep = {"run": args.run, "tau": tau, "recall": rec,
           "gt_px": n_gt, "fn_px": n_fn, "fn_frac_of_gt": n_fn / n_gt,
           "fn_strata_px": strata,
           "fn_strata_frac_of_fn": {k: v / max(n_fn, 1) for k, v in strata.items()},
           "fn_strata_frac_of_gt": {k: v / n_gt for k, v in strata.items()},
           "fp_px": int(fp.sum()),
           "fp_boundary_px": int(fp_near.sum()),
           "fp_isolated_components": int(n_fp_comp),
           "fp_isolated_px": int(fp_sizes.sum()) if len(fp_sizes) else 0,
           "dead_components": [
               {"x_center": int((objs[c - 1][1].start + objs[c - 1][1].stop) / 2),
                "size": int(sizes[c - 1]), "mean_prob": float(comp_prob[c - 1])}
               for c in range(1, n + 1) if dead[c]]}

    # top components by FN mass
    fn_by_comp = np.bincount(fn_ids, minlength=n + 1)
    top = np.argsort(fn_by_comp[1:])[::-1][:10] + 1
    rep["top10_missing_components"] = [
        {"x_center": int((objs[c - 1][1].start + objs[c - 1][1].stop) / 2),
         "size": int(sizes[c - 1]), "fn_px": int(fn_by_comp[c]),
         "recall": 1 - fn_by_comp[c] / sizes[c - 1],
         "contrast": round(float(contrast[c]), 2), "median_width": round(float(width[c]), 1),
         "dead": bool(dead[c])} for c in top]

    out_dir = os.path.join(args.out, "error_anatomy")
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
    with open(os.path.join(out_dir, f"{args.run}_anatomy.json"), "w") as f:
        json.dump(rep, f, indent=2)

    lines = [f"# Error anatomy — {args.run} (tau {tau}, recall {rec:.3f})", "",
             "| FN stratum | px | % of FN | % of GT |", "|---|---|---|---|"]
    for k, v in strata.items():
        lines.append(f"| {k} | {v:,} | {v/max(n_fn,1):.1%} | {v/n_gt:.1%} |")
    lines += ["", f"FP: {rep['fp_px']:,} px total — {rep['fp_boundary_px']:,} within "
              f"{args.edge_r}px of GT (boundary), {rep['fp_isolated_px']:,} in "
              f"{rep['fp_isolated_components']} isolated islands."]
    with open(os.path.join(out_dir, f"{args.run}_anatomy.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure: FN colored by stratum (downsampled 2x)
    ds = 2
    m = np.zeros(gt[::ds, ::ds].shape, dtype=np.uint8)
    fnv = np.zeros_like(fn); fnv[fn] = (1 * s_dead + 2 * s_edge + 3 * s_lowc + 4 * s_unex)
    m = fnv[::ds, ::ds]
    fig, ax = plt.subplots(figsize=(22, 3.5))
    cmap = matplotlib.colors.ListedColormap(
        ["black", "magenta", "orange", "deepskyblue", "red"])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(f"{args.run} FN strata — magenta=dead-label, orange=boundary, "
                 f"blue=low-contrast, red=UNEXPLAINED")
    ax.axis("off")
    fig.savefig(os.path.join(out_dir, "figs", f"{args.run}_fn_map.png"), dpi=140,
                bbox_inches="tight")
    print(f"-> {out_dir}/{args.run}_anatomy.md")


if __name__ == "__main__":
    main()
