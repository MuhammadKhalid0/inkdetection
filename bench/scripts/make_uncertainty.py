"""Build per-pixel label-confidence trimaps — plan 2026-07-17 §6.1.

Supervisor request C3 ('another class with a low-probability softmax'): mark
the pixels whose labels cannot be trusted so the loss can down-weight them
(u1 dice_bce_masked) or route them to a third softmax class (u2 soft_ce3).

Trimap unc.npy (uint8):  2 = confident ink core  ·  0 = confident background
                         1 = uncertain band

Construction per TRAIN segment (val never gets one — it is never trained on):
  core_ink = erosion(ink, r)      halo = dilation(ink, r)
  band     = halo & ~core_ink     (label-boundary noise, +-r px)
plus residual-misregistration widening on the 4 identity-gated segments
(|best dy| from data_issues_by_numbers.md §1 — labels sit BELOW the ink):
  suspect_bg  = rows up to dy ABOVE strokes (may be unlabelled ink)
  suspect_ink = bottom dy rows of stroke interiors (may be papyrus)

Erosion/dilation use a separable square filter (Chebyshev r) for speed.

Usage:  python -m scripts.make_uncertainty --r 8 --label-file ink_reg.npy
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import TRAIN_SEGMENTS, proc_dir  # noqa: E402

# |best dy| of segments the registration left at identity (gain below the 0.4
# acceptance gate) — their labels carry an uncorrected offset of this order.
RESIDUAL_DY = {
    "20241025145341": 30,
    "20241030152031": 20,
    "20241113080880": 20,
    "20241108120732": 10,
}


def _up_union(a, k):
    """union over shifts: out[i] = any(a[i+1 .. i+k]) — 'ink within k rows below'."""
    out = np.zeros_like(a)
    cur = a
    for _ in range(k):
        nxt = np.zeros_like(cur)
        nxt[:-1] = cur[1:]
        out |= nxt
        cur = nxt
    return out


def _up_all(a, k):
    """out[i] = all(a[i .. i+k]) — erosion from below along the row axis."""
    out = a.copy()
    for _ in range(k):
        nxt = np.zeros_like(out)
        nxt[:-1] = out[1:]
        out &= nxt
    return out


def build_trimap(seg, r, label_file):
    ink = np.load(os.path.join(proc_dir(seg), label_file)) > 0
    size = 2 * r + 1
    core = ndimage.minimum_filter(ink.astype(np.uint8), size=size) > 0   # erosion
    halo = ndimage.maximum_filter(ink.astype(np.uint8), size=size) > 0   # dilation
    band = halo & ~core

    dy = RESIDUAL_DY.get(seg, 0)
    if dy > 0:
        sus_bg = _up_union(ink, dy) & ~halo        # unlabelled rows above strokes
        sus_ink = ink & ~_up_all(ink, dy)          # bottom dy rows of stroke interiors
        band = band | sus_bg | sus_ink
        core = core & ~band

    unc = np.zeros(ink.shape, dtype=np.uint8)      # 0 = confident background
    unc[band] = 1
    unc[core & ~band] = 2

    n_ink = int(ink.sum())
    stats = {
        "segment": seg, "r": r, "residual_dy": dy,
        "ink_px": n_ink,
        "core_ink_px": int((unc == 2).sum()),
        "band_px": int((unc == 1).sum()),
        "core_keep_frac": float((unc == 2).sum()) / max(n_ink, 1),
        "band_frac_of_all": float((unc == 1).sum()) / unc.size,
    }
    return unc, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=8, help="boundary band half-width (px)")
    ap.add_argument("--label-file", default="ink_reg.npy")
    ap.add_argument("--segments", nargs="*", default=None)
    # 2026-07-20 (reg2 rebuild): write to a different file so the historical
    # unc.npy is preserved, and allow disabling the residual-dy widening — the
    # reg2 field warp corrects exactly the offsets the widening compensated.
    ap.add_argument("--out-file", default="unc.npy")
    ap.add_argument("--no-dy-widen", action="store_true",
                    help="skip RESIDUAL_DY widening (use with ink_reg2.npy)")
    args = ap.parse_args()
    if args.no_dy_widen:
        RESIDUAL_DY.clear()

    segs = args.segments or TRAIN_SEGMENTS
    all_stats = []
    for seg in segs:
        unc, stats = build_trimap(seg, args.r, args.label_file)
        np.save(os.path.join(proc_dir(seg), args.out_file), unc)
        with open(os.path.join(proc_dir(seg),
                               args.out_file.replace(".npy", "_stats.json")), "w") as f:
            json.dump(stats, f, indent=2)
        all_stats.append(stats)
        flag = ("  <-- core keeps <80% of ink; check strokes are not vanishing"
                if stats["core_keep_frac"] < 0.80 else "")
        print(f"[{seg}] core keeps {stats['core_keep_frac']:.1%} of ink px, "
              f"band = {stats['band_frac_of_all']:.2%} of all px, dy={stats['residual_dy']}{flag}")
    mean_keep = float(np.mean([s["core_keep_frac"] for s in all_stats]))
    print(f"== mean core-keep {mean_keep:.1%} | plan gate: if a segment loses >5% "
          f"of ink area AND r=8, rebuild that segment with --r 4 ==")


if __name__ == "__main__":
    main()
