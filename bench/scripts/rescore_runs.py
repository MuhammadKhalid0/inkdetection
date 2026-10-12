"""Re-score every completed run's saved val probability map against a label file.

Motivation (2026-07-02): the GT ink labels are misregistered (labels sit 10-45px
below the physical ink signal).  ``ink_reg.npy`` fixes this.  Every prior run
saved ``val_prob_best.npy`` (checkpoint chosen on ORIGINAL labels — noted in the
output), so the honest thing is to re-score all of them against the registered
labels with a fresh tau sweep and report both numbers side by side.

Usage:
    python -m scripts.rescore_runs                     # all runs, ink_reg.npy
    python -m scripts.rescore_runs --label-file ink.npy --runs a1 a2bs
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import VAL_SEGMENT, proc_dir  # noqa: E402

OUT_ROOT = "/home/woody/iwi5/iwi5401h/vesuvius/outputs"
TAUS = np.round(np.arange(0.30, 0.91, 0.05), 2)


def f05(pred, gt, m):
    tp = float((pred & gt & m).sum())
    fp = float((pred & ~gt & m).sum())
    fn = float((~pred & gt & m).sum())
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    f = 1.25 * p * r / max(0.25 * p + r, 1e-9)
    return f, p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-file", default="ink_reg.npy")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    proc = proc_dir(VAL_SEGMENT)
    gt_orig = np.load(os.path.join(proc, "ink.npy")) > 0
    gt_path = os.path.join(proc, args.label_file)
    if not os.path.exists(gt_path):
        sys.exit(f"missing {gt_path} — run scripts.register_labels first")
    gt_reg = np.load(gt_path) > 0
    mask = np.load(os.path.join(proc, "mask.npy")) > 0

    runs = args.runs or sorted(
        d for d in os.listdir(OUT_ROOT)
        if os.path.exists(os.path.join(OUT_ROOT, d, "val_prob_best.npy")))

    print(f"val={VAL_SEGMENT}  labels: orig=ink.npy  reg={args.label_file}")
    print(f"{'run':<10} {'F.5 orig':>9} {'tau':>5} {'F.5 reg':>9} {'tau':>5} {'dF':>7}   P/R (reg)")
    results = {}
    for run in runs:
        prob = np.load(os.path.join(OUT_ROOT, run, "val_prob_best.npy"))
        if prob.shape != gt_reg.shape:
            print(f"{run:<10} shape mismatch {prob.shape} vs {gt_reg.shape} — skipped")
            continue
        bo = max(((f05(prob >= t, gt_orig, mask)) + (t,) for t in TAUS), key=lambda z: z[0])
        br = max(((f05(prob >= t, gt_reg, mask)) + (t,) for t in TAUS), key=lambda z: z[0])
        results[run] = {"orig": {"f05": bo[0], "p": bo[1], "r": bo[2], "tau": bo[3]},
                        "reg": {"f05": br[0], "p": br[1], "r": br[2], "tau": br[3]}}
        print(f"{run:<10} {bo[0]:9.4f} {bo[3]:5.2f} {br[0]:9.4f} {br[3]:5.2f} {br[0]-bo[0]:+7.4f}   "
              f"{br[1]:.3f}/{br[2]:.3f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()
