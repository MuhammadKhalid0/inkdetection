"""Morphological post-processing for saved probability maps.

Motivation (2026-07-04, measured on a2bs_t128 val): real letterforms on these
segments are >=42k px; thresholded FP islands are far smaller.  Removing
connected components below a minimum size raises F0.5 from 0.7185 to 0.7385
(tau 0.70, min_size 10k) — the legitimate, label-free version of a
"text-region prior".  Image-domain enhancement (CLAHE, background subtraction,
detrending) was tested and REDUCES ink/papyrus separability; ensembling and
hysteresis also hurt.  Component filtering is the one post-hoc lever that works.

Sweeps (tau, min_size) on a run's saved val probability map, writes the best
operating point to outputs/<run>/postprocess.json, and (optionally) applies it
to the test probability map for qualitative use.

Usage:
    python -m scripts.postprocess --run a2bs_t128                  # sweep val
    python -m scripts.postprocess --run a2bs_t128 --apply-test     # + clean test pred .npy
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import VAL_SEGMENT, proc_dir, OUTPUT_ROOT  # noqa: E402

TAUS = np.round(np.arange(0.50, 0.86, 0.05), 2)
MIN_SIZES = [0, 2000, 5000, 10000, 20000]


def f05(pred, gt, m):
    tp = float((pred & gt & m).sum())
    fp = float((pred & ~gt & m).sum())
    fn = float((~pred & gt & m).sum())
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    return 1.25 * p * r / max(0.25 * p + r, 1e-9), p, r


def component_filter(pred: np.ndarray, min_size: int) -> np.ndarray:
    """Remove 8-connected components smaller than min_size pixels."""
    if min_size <= 0:
        return pred
    lab, _ = ndimage.label(pred, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= min_size)[0]
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label-file", default="ink_reg.npy")
    ap.add_argument("--apply-test", action="store_true",
                    help="apply best (tau,min_size) to outputs/<run>/test_prob_*.npy")
    args = ap.parse_args()

    out_dir = os.path.join(OUTPUT_ROOT, args.run)
    prob = np.load(os.path.join(out_dir, "val_prob_best.npy"))
    proc = proc_dir(VAL_SEGMENT)
    gt = np.load(os.path.join(proc, args.label_file)) > 0
    mask = np.load(os.path.join(proc, "mask.npy")) > 0

    base = max(((f05(prob >= t, gt, mask)) + (t,) for t in TAUS), key=lambda z: z[0])
    print(f"[{args.run}] baseline (no filter): F0.5={base[0]:.4f} P={base[1]:.3f} "
          f"R={base[2]:.3f} @tau{base[3]:.2f}")

    best = None
    for tau in TAUS:
        pred0 = prob >= tau
        lab, _ = ndimage.label(pred0, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        for ms in MIN_SIZES:
            if ms == 0:
                pred = pred0
            else:
                keep = np.where(sizes >= ms)[0]
                keep = keep[keep > 0]
                pred = np.isin(lab, keep)
            f, p, r = f05(pred, gt, mask)
            if best is None or f > best["f05"]:
                best = {"tau": float(tau), "min_size": int(ms),
                        "f05": f, "precision": p, "recall": r}
    print(f"[{args.run}] best: F0.5={best['f05']:.4f} P={best['precision']:.3f} "
          f"R={best['recall']:.3f} @tau{best['tau']:.2f} min_size={best['min_size']}")

    payload = {"run": args.run, "label_file": args.label_file,
               "baseline": {"f05": base[0], "precision": base[1],
                            "recall": base[2], "tau": float(base[3])},
               "postprocessed": best,
               "note": "tau+min_size selected on the VALIDATION segment (the "
                       "designated selection set); component filter is label-free "
                       "at inference time."}
    with open(os.path.join(out_dir, "postprocess.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"-> {os.path.join(out_dir, 'postprocess.json')}")

    if args.apply_test:
        cands = sorted(glob.glob(os.path.join(out_dir, "test_prob_*.npy")))
        if not cands:
            sys.exit(f"no test_prob_*.npy under {out_dir}")
        tprob = np.load(cands[0], mmap_mode="r")
        pred = np.asarray(tprob) >= best["tau"]
        pred = component_filter(pred, best["min_size"])
        out = os.path.join(out_dir, f"test_pred_clean_tau{best['tau']:.2f}"
                                    f"_ms{best['min_size']}.npy")
        np.save(out, pred.astype(np.uint8))
        print(f"-> {out}  (coverage {100 * pred.mean():.2f}% of full frame)")


if __name__ == "__main__":
    main()
