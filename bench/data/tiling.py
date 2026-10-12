"""Build tile manifests for the fixed train/val split.

A tile is a (segment, y, x) top-left coordinate for a ``tile x tile`` in-plane
window (depth is added at read time).  We keep tiles whose valid-mask coverage
>= ``min_mask`` and record their ink fraction so the training sampler can
oversample ink-positive tiles.

Outputs:  data/manifests/{train,val}.csv   columns: segment,y,x,ink_frac,mask_frac

Usage:
    python -m data.tiling                     # tile=64 stride=32 min_mask=0.5
    python -m data.tiling --tile 64 --stride 32 --min-mask 0.5
"""
import argparse
import csv
import os

import numpy as np

from .segments import (MANIFEST_ROOT, TRAIN_SEGMENTS, VAL_SEGMENT, proc_dir)


def _tile_starts(size: int, tile: int, stride: int):
    if size < tile:
        return []
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)          # ensure edge coverage
    return starts


def tiles_for_segment(seg: str, tile: int, stride: int, min_mask: float,
                      label_file: str = "ink.npy"):
    ink = np.load(os.path.join(proc_dir(seg), label_file))   # (H,W) {0,1}
    mask = np.load(os.path.join(proc_dir(seg), "mask.npy"))  # (H,W) {0,1}
    H, W = mask.shape
    area = tile * tile
    rows = []
    for y in _tile_starts(H, tile, stride):
        for x in _tile_starts(W, tile, stride):
            m = mask[y:y + tile, x:x + tile]
            mask_frac = float(m.sum()) / area
            if mask_frac < min_mask:
                continue
            ink_frac = float(ink[y:y + tile, x:x + tile].sum()) / area
            rows.append((seg, y, x, round(ink_frac, 5), round(mask_frac, 5)))
    return rows


def write_manifest(path: str, segments, tile, stride, min_mask, label_file="ink.npy"):
    all_rows = []
    for seg in segments:
        rows = tiles_for_segment(seg, tile, stride, min_mask, label_file)
        pos = sum(1 for r in rows if r[3] >= 0.05)
        print(f"[{seg}] {len(rows)} tiles ({pos} ink>=5%)")
        all_rows.extend(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "y", "x", "ink_frac", "mask_frac"])
        w.writerows(all_rows)
    print(f"-> {path}: {len(all_rows)} tiles total\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--min-mask", type=float, default=0.5)
    ap.add_argument("--label-file", default="ink.npy",
                    help="ink.npy (original) or ink_reg.npy (registered labels); "
                         "ink_frac in the manifest is computed from this file")
    ap.add_argument("--suffix", default="",
                    help="manifest filename suffix, e.g. _reg -> train_reg.csv")
    args = ap.parse_args()
    print("== TRAIN ==")
    write_manifest(os.path.join(MANIFEST_ROOT, f"train{args.suffix}.csv"),
                   TRAIN_SEGMENTS, args.tile, args.stride, args.min_mask, args.label_file)
    print("== VAL ==")
    write_manifest(os.path.join(MANIFEST_ROOT, f"val{args.suffix}.csv"),
                   [VAL_SEGMENT], args.tile, args.stride, args.min_mask, args.label_file)


if __name__ == "__main__":
    main()
