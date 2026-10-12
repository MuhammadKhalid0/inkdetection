"""Prepare the inference (test) segment for sliding-window prediction.

The test segment (auto_grown 'flatboi', PHerc.0172 / Scroll 5) ships as 65 raw
depth tifs (00.tif..64.tif) plus tifxyz_mask.png, and has NO ink labels
(qualitative evaluation only).  Unlike the labelled segments it has no
pre-stacked <seg>.npy, so this tool stacks the tifs into a uint8 volume memmap
(bounded RAM: one layer at a time) and binarises the mask, writing
    proc_infer/<seg>/{vol.npy, mask.npy, meta.json}
in the same layout infer_segment() expects (with load_labels=False).

Usage:
    python -m scripts.prepare_infer                       # default TEST_SEGMENT
    python -m scripts.prepare_infer --segment <id> --dry-run
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import (TEST_SEGMENT, proc_infer_dir,            # noqa: E402
                           raw_infer_layers_dir, raw_infer_mask_path)

Image.MAX_IMAGE_PIXELS = None
N_LAYERS = 65


def _layer_path(seg, i):
    return os.path.join(raw_infer_layers_dir(seg), f"{i:02d}.tif")


def prepare_infer(seg, dry_run=False):
    # shapes from layer 0 + mask; crop all to common min (H,W)
    l0 = np.asarray(Image.open(_layer_path(seg, 0)))
    H0, W0 = l0.shape[:2]
    m = np.asarray(Image.open(raw_infer_mask_path(seg)))
    if m.ndim == 3:
        m = m[..., 0]
    H1, W1 = m.shape
    Hc, Wc = min(H0, H1), min(W0, W1)
    mask_c = (m[:Hc, :Wc] > 127).astype(np.uint8)
    print(f"[{seg}] layers=({H0},{W0}) mask=({H1},{W1}) -> crop=({Hc},{Wc}) "
          f"maskfrac={mask_c.mean():.3f} vol={N_LAYERS*Hc*Wc/1e9:.1f} GB")
    if dry_run:
        return

    out = proc_infer_dir(seg)
    os.makedirs(out, exist_ok=True)
    vol_out = np.lib.format.open_memmap(
        os.path.join(out, "vol.npy"), mode="w+", dtype=np.uint8, shape=(N_LAYERS, Hc, Wc))
    peak_sum = np.zeros(N_LAYERS, dtype=np.float64)
    for i in range(N_LAYERS):
        li = np.asarray(Image.open(_layer_path(seg, i)))
        if li.ndim == 3:
            li = li[..., 0]
        vol_out[i] = li[:Hc, :Wc]
        peak_sum[i] = float(vol_out[i, ::50, ::50].mean())
        print(f"  [{seg}] layer {i:02d}/{N_LAYERS-1} mean={peak_sum[i]:.1f}", flush=True)
    vol_out.flush(); del vol_out
    np.save(os.path.join(out, "mask.npy"), mask_c)
    info = {"segment": seg, "depth": N_LAYERS, "cropped_hw": [int(Hc), int(Wc)],
            "depth_profile": peak_sum.tolist(), "depth_peak": int(np.argmax(peak_sum)),
            "labels": False}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[{seg}] wrote vol{(N_LAYERS, Hc, Wc)} + mask  peak={info['depth_peak']} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default=TEST_SEGMENT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prepare_infer(args.segment, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
