"""Preprocess raw Vault segments -> fast, aligned arrays on $WORK.

For each labelled segment:
  * read the pre-stacked uint8 volume  (65, H0, W0) from Vault (memmap),
  * read inklabels.png (RGBA, R=G=B)  -> binary uint8 {0,1} at threshold 127,
  * read mask.png (grayscale)         -> binary uint8 {0,1} at threshold 127,
  * crop all three to the common min (H, W)  (masks are 1px smaller; one segment
    has a 16px height mismatch),
  * write vol.npy / ink.npy / mask.npy + meta.json to  data/proc/<seg>/.

Memory-safe: the volume is copied layer-by-layer via an output memmap, so peak
RAM is one layer (<=~250 MB even for the 22000x10580 segment).

Usage:
    python -m data.prepare                # all labelled segments
    python -m data.prepare --segments 20241025145341 20241113090990
    python -m data.prepare --dry-run      # just print shapes / alignment
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from .segments import (SEGMENTS, proc_dir, raw_ink_path, raw_mask_path,
                       raw_vol_path)

Image.MAX_IMAGE_PIXELS = None  # these papyrus maps are >100 MP


def _load_binary_png(path: str, channel0: bool) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if channel0 and arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 127).astype(np.uint8)


def _depth_profile(vol: np.ndarray, step: int = 50) -> list:
    # mean intensity per depth layer on a spatial subsample (fast)
    return [float(vol[d, ::step, ::step].mean()) for d in range(vol.shape[0])]


def prepare_segment(seg: str, dry_run: bool = False) -> dict:
    vol = np.load(raw_vol_path(seg), mmap_mode="r")          # (D, H0, W0) uint8
    ink = _load_binary_png(raw_ink_path(seg), channel0=True)  # (H1, W1)
    mask = _load_binary_png(raw_mask_path(seg), channel0=False)  # (H2, W2)

    D, H0, W0 = vol.shape
    H1, W1 = ink.shape
    H2, W2 = mask.shape
    Hc, Wc = min(H0, H1, H2), min(W0, W1, W2)

    ink_c = np.ascontiguousarray(ink[:Hc, :Wc])
    mask_c = np.ascontiguousarray(mask[:Hc, :Wc])
    prof = _depth_profile(vol)
    peak = int(np.argmax(prof))
    ink_frac = float(ink_c[mask_c > 0].mean()) if mask_c.any() else float(ink_c.mean())

    info = {
        "segment": seg, "depth": int(D),
        "raw_vol_hw": [int(H0), int(W0)], "raw_ink_hw": [int(H1), int(W1)],
        "raw_mask_hw": [int(H2), int(W2)], "cropped_hw": [int(Hc), int(Wc)],
        "depth_profile": prof, "depth_peak": peak, "ink_fraction": round(ink_frac, 5),
    }
    if dry_run:
        print(f"[{seg}] vol={vol.shape} ink=({H1},{W1}) mask=({H2},{W2}) "
              f"-> crop=({Hc},{Wc})  peak={peak}  ink_frac={ink_frac:.4f}")
        return info

    out = proc_dir(seg)
    os.makedirs(out, exist_ok=True)
    # volume: copy layer-by-layer into an output memmap (bounded RAM)
    vol_out = np.lib.format.open_memmap(
        os.path.join(out, "vol.npy"), mode="w+", dtype=np.uint8, shape=(D, Hc, Wc))
    for d in range(D):
        vol_out[d] = vol[d, :Hc, :Wc]
    vol_out.flush()
    del vol_out
    np.save(os.path.join(out, "ink.npy"), ink_c)
    np.save(os.path.join(out, "mask.npy"), mask_c)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[{seg}] wrote vol{(D, Hc, Wc)} ink/mask  peak={peak} ink_frac={ink_frac:.4f} -> {out}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="*", default=SEGMENTS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for seg in args.segments:
        prepare_segment(seg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
