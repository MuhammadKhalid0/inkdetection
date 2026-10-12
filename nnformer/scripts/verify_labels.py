"""Residual label-alignment check for ALL 9 segments (2026-07-20, per Omar).

For each segment: band-mean intensity image (band from reg_report), per-column
median detrend, then contrast c(dy) = mean(detrended | ink mask shifted by dy)
over dy in [-50, 50]. If labels sit correctly the peak is at dy ~ 0.
Expected residuals on the 4 identity-gated segments (registration below the
acceptance gate, deliberately unwarped; mitigated by the u1 trimap widening):
145341 ~30, 152031 ~20, 120732 ~10, 080880 ~20 (magnitudes).

Writes docs/label_verification.json + prints a table.
"""
import json
import glob
import os

import numpy as np

PROC = "/home/woody/iwi5/iwi5401h/vesuvius/data/proc"
OUT = "/home/woody/iwi5/iwi5401h/nnFormer/docs/label_verification.json"
EXPECT = {  # |residual| expected in ink_reg.npy (0 = fully registered)
    "20241025145341": 30, "20241030152031": 20, "20241108120732": 10,
    "20241113080880": 20, "20241025145701": 0, "20241108111522": 0,
    "20241108115232": 0, "20241113070770": 0, "20241113090990": 0,
}

def check_segment(seg):
    d = os.path.join(PROC, seg)
    rep = json.load(open(os.path.join(d, "reg_report.json")))
    b0, b1 = rep["band"]
    vol = np.load(os.path.join(d, "vol.npy"), mmap_mode="r")
    ink = np.load(os.path.join(d, "ink_reg.npy"), mmap_mode="r")
    mask = np.load(os.path.join(d, "mask.npy"), mmap_mode="r")
    H, W = ink.shape
    xs = 6 if W > 15000 else 2          # column subsample (memory/speed)
    sl = slice(None, None, xs)
    # band-mean image from 3 central band slices
    zc = (b0 + b1) // 2
    img = np.zeros((H, len(range(0, W, xs))), np.float32)
    for z in (zc - 2, zc, zc + 2):
        img += np.asarray(vol[z, :, sl], np.float32)
    img /= 3.0
    m = np.asarray(mask[:, sl]) > 0
    ik = np.asarray(ink[:, sl]) > 0
    med = np.median(np.where(m, img, np.nan), axis=0)
    med = np.nan_to_num(med, nan=float(np.nanmedian(med)))
    det = img - med[None, :]
    det[~m] = 0.0
    base = det[m].mean()

    def c(dy):
        s = np.roll(ik, dy, axis=0)
        v = s & m
        return abs(float(det[v].mean() - base)) if v.any() else 0.0

    coarse = {dy: c(dy) for dy in range(-50, 51, 2)}
    p = max(coarse, key=coarse.get)
    fine = {dy: coarse.get(dy) or c(dy) for dy in range(p - 3, p + 4)}
    peak = max(fine, key=fine.get)
    return {"segment": seg, "peak_dy": int(peak), "c_at_0": round(coarse[0], 3),
            "c_at_peak": round(fine[peak], 3), "expected_abs": EXPECT[seg],
            "band": [b0, b1], "x_subsample": xs}

results = []
for f in sorted(glob.glob(os.path.join(PROC, "*", "reg_report.json"))):
    seg = f.split(os.sep)[-2]
    r = check_segment(seg)
    ok = abs(abs(r["peak_dy"]) - r["expected_abs"]) <= 8
    r["verdict"] = "OK" if ok else "CHECK"
    results.append(r)
    print(f"{seg}  peak_dy={r['peak_dy']:+4d}  expected~|{r['expected_abs']}|  "
          f"c(0)={r['c_at_0']:.3f} c(peak)={r['c_at_peak']:.3f}  [{r['verdict']}]")

json.dump(results, open(OUT, "w"), indent=1)
print("->", OUT)
