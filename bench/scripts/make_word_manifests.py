"""Build word-box (annotation-anchored) training manifests — plan 2026-07-17 §5.

Supervisor request C1: train on squares placed ON the annotated words instead of
a blind grid, because parts of the background annotation are untrustworthy.

Per TRAIN segment (val is NEVER touched — evaluation stays full-segment):
  * 'word' tiles: for every 8-connected component of the registered ink label,
    draw k = max(4, ceil(area / per_area)) anchor pixels uniformly from the
    component, jitter the tile centre by +-jitter px, keep if mask_frac >= min_mask.
  * 'neg_clean' tiles: regular grid tiles (stride) with mask_frac >= min_mask,
    ink_frac == 0 AND >= neg_margin px away from any labelled ink.
  * 'neg_dirty' tiles: remaining grid negatives (contain ink or are within the
    margin) — recorded for bookkeeping, weight 0 in the sampler by default.

Output: data/manifests/train<suffix>.csv with columns
    segment,y,x,ink_frac,mask_frac,kind

Usage (plan defaults):
    python -m scripts.make_word_manifests --tile 192 --stride 96 --jitter 48 \
        --per-area 9216 --neg-margin 32 --min-mask 0.5 \
        --label-file ink_reg.npy --suffix _wb_t192 --seed 0
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import MANIFEST_ROOT, TRAIN_SEGMENTS, proc_dir  # noqa: E402
from data.tiling import _tile_starts  # noqa: E402


def wordbox_rows(seg, tile, stride, jitter, per_area, neg_margin, min_mask,
                 label_file, rng):
    ink = np.load(os.path.join(proc_dir(seg), label_file)) > 0
    mask = np.load(os.path.join(proc_dir(seg), "mask.npy")) > 0
    H, W = mask.shape
    area = tile * tile
    rows = []
    seen = set()

    # --- 'word' tiles: anchored on ink components -------------------------
    lab, n_comp = ndimage.label(ink, structure=np.ones((3, 3)))
    comp_sizes = np.bincount(lab.ravel())
    objs = ndimage.find_objects(lab)   # bbox per component — avoids O(n_comp*N) scans
    n_word = 0
    for c in range(1, n_comp + 1):
        sl = objs[c - 1]
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == c)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        k = max(4, math.ceil(comp_sizes[c] / per_area))
        pick = rng.integers(0, len(ys), size=k)
        for p in pick:
            cy, cx = int(ys[p]), int(xs[p])
            y0 = cy - tile // 2 + int(rng.integers(-jitter, jitter + 1))
            x0 = cx - tile // 2 + int(rng.integers(-jitter, jitter + 1))
            y0 = int(np.clip(y0, 0, H - tile))
            x0 = int(np.clip(x0, 0, W - tile))
            if (y0, x0) in seen:
                continue
            m = mask[y0:y0 + tile, x0:x0 + tile]
            mask_frac = float(m.sum()) / area
            if mask_frac < min_mask:
                continue
            ink_frac = float(ink[y0:y0 + tile, x0:x0 + tile].sum()) / area
            seen.add((y0, x0))
            rows.append((seg, y0, x0, round(ink_frac, 5), round(mask_frac, 5), "word"))
            n_word += 1

    # --- grid negatives, split clean/dirty by distance to labelled ink ----
    # 'clean' = zero ink in the tile AND zero overlap with the ink dilated by
    # neg_margin (separable square max-filter — Chebyshev distance, fast).
    ink_dil = ndimage.maximum_filter(ink.astype(np.uint8),
                                     size=2 * neg_margin + 1) > 0
    n_clean = n_dirty = 0
    for y0 in _tile_starts(H, tile, stride):
        for x0 in _tile_starts(W, tile, stride):
            m = mask[y0:y0 + tile, x0:x0 + tile]
            mask_frac = float(m.sum()) / area
            if mask_frac < min_mask:
                continue
            ifrac = float(ink[y0:y0 + tile, x0:x0 + tile].sum()) / area
            if ifrac >= 0.05:
                continue  # grid positives are not needed; words come from anchors
            if (y0, x0) in seen:
                continue
            if ifrac == 0.0 and not ink_dil[y0:y0 + tile, x0:x0 + tile].any():
                kind = "neg_clean"; n_clean += 1
            else:
                kind = "neg_dirty"; n_dirty += 1
            seen.add((y0, x0))
            rows.append((seg, y0, x0, round(ifrac, 5), round(mask_frac, 5), kind))

    print(f"[{seg}] comps={n_comp} word={n_word} neg_clean={n_clean} neg_dirty={n_dirty}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=192)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--jitter", type=int, default=48)
    ap.add_argument("--per-area", type=int, default=9216, help="one anchor per this many ink px")
    ap.add_argument("--neg-margin", type=int, default=32)
    ap.add_argument("--min-mask", type=float, default=0.5)
    ap.add_argument("--label-file", default="ink_reg.npy")
    ap.add_argument("--suffix", default="_wb_t192")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    all_rows = []
    for seg in TRAIN_SEGMENTS:
        all_rows.extend(wordbox_rows(seg, args.tile, args.stride, args.jitter,
                                     args.per_area, args.neg_margin, args.min_mask,
                                     args.label_file, rng))
    out = os.path.join(MANIFEST_ROOT, f"train{args.suffix}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "y", "x", "ink_frac", "mask_frac", "kind"])
        w.writerows(all_rows)
    kinds = [r[5] for r in all_rows]
    print(f"-> {out}: {len(all_rows)} tiles "
          f"(word={kinds.count('word')}, neg_clean={kinds.count('neg_clean')}, "
          f"neg_dirty={kinds.count('neg_dirty')})")


if __name__ == "__main__":
    main()
