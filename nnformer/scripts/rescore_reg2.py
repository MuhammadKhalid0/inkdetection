"""Rescore saved val probability maps against ink_reg vs ink_reg2 (2026-07-20).

No retraining/inference — pure CPU rescoring of existing maps, mirroring the
July registration rescore. Reports raw swept F0.5 and post (tau, min_size grid)
for both label versions. Output: rereg/rescore_reg2.md + .json
"""
import json
import os

import numpy as np
from scipy import ndimage

WORK = "/home/woody/iwi5/iwi5401h"
VAL = "20241113090990"
PROC = f"{WORK}/vesuvius/data/proc/{VAL}"
OUT = f"{WORK}/nnFormer/rereg"

RUNS = {
    "unetrpp_c1_t256_s42": f"{WORK}/vesuvius/outputs5/c1_t256_s42",
    "unetrpp_c1_t256_s43": f"{WORK}/vesuvius/outputs5/c1_t256_s43",
    "unetrpp_c1_t256_s44": f"{WORK}/vesuvius/outputs5/c1_t256_s44",
    "unetrpp_t256_lean_s42": f"{WORK}/vesuvius/outputs4/t256_lean_s42",
    "unetrpp_lean_t192_s42": f"{WORK}/vesuvius/outputs/lean_t192_s42",
    "unetrpp_a2bs_t192": f"{WORK}/vesuvius/outputs/a2bs_t192",
    "unetrpp_r4": f"{WORK}/vesuvius/outputs/r4",
    "nnformer_n_base_t192": f"{WORK}/nnFormer/outputs/n_base_t192_s42",
    "nnformer_n_embedz2_t192": f"{WORK}/nnFormer/outputs/n_embedz2_t192_s42",
}
TAUS = np.round(np.arange(0.10, 0.91, 0.05), 2)
MIN_SIZES = [0, 5000, 10000, 20000]


def f05(pred, gt, m):
    tp = float((pred & gt & m).sum())
    fp = float((pred & ~gt & m).sum())
    fn = float((~pred & gt & m).sum())
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return 1.25 * p * r / max(0.25 * p + r, 1e-9), p, r


def component_filter(pred, min_size):
    if min_size <= 0:
        return pred
    lab, n = ndimage.label(pred, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return pred
    sizes = ndimage.sum(pred, lab, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_size
    return keep[lab]


def score(prob, gt, m):
    best_raw = (-1, None)
    for t in TAUS:
        f, p, r = f05(prob >= t, gt, m)
        if f > best_raw[0]:
            best_raw = (f, {"tau": float(t), "f05": round(f, 4),
                            "P": round(p, 3), "R": round(r, 3)})
    raw = best_raw[1]
    best_post = (-1, None)
    for t in [max(raw["tau"] - 0.1, 0.1), max(raw["tau"] - 0.05, 0.1),
              raw["tau"], min(raw["tau"] + 0.05, 0.9)]:
        base = prob >= t
        for ms in MIN_SIZES:
            f, p, r = f05(component_filter(base, ms), gt, m)
            if f > best_post[0]:
                best_post = (f, {"tau": float(t), "min_size": ms,
                                 "f05": round(f, 4), "P": round(p, 3), "R": round(r, 3)})
    return raw, best_post[1]


def main():
    gt1 = np.load(os.path.join(PROC, "ink_reg.npy")) > 0
    gt2 = np.load(os.path.join(PROC, "ink_reg2.npy")) > 0
    m = np.load(os.path.join(PROC, "mask.npy")) > 0
    rows, js = [], {}
    for name, d in RUNS.items():
        p_path = os.path.join(d, "val_prob_best.npy")
        if not os.path.exists(p_path):
            print(f"skip {name} (no map)")
            continue
        prob = np.load(p_path).astype(np.float32)
        r1, p1 = score(prob, gt1, m)
        r2, p2 = score(prob, gt2, m)
        js[name] = {"reg": {"raw": r1, "post": p1}, "reg2": {"raw": r2, "post": p2}}
        rows.append((name, r1, p1, r2, p2))
        print(f"{name:<26} reg raw {r1['f05']:.4f} post {p1['f05']:.4f} | "
              f"reg2 raw {r2['f05']:.4f} post {p2['f05']:.4f} | "
              f"d_post {p2['f05'] - p1['f05']:+.4f}")
    lines = ["# Rescore: ink_reg vs ink_reg2 (same maps, no retraining)", "",
             "| run | reg raw | reg post | reg2 raw | reg2 post | delta post |",
             "|---|---|---|---|---|---|"]
    for name, r1, p1, r2, p2 in rows:
        lines.append(f"| {name} | {r1['f05']:.4f} | {p1['f05']:.4f} | "
                     f"{r2['f05']:.4f} | {p2['f05']:.4f} | {p2['f05'] - p1['f05']:+.4f} |")
    open(os.path.join(OUT, "rescore_reg2.md"), "w").write("\n".join(lines) + "\n")
    json.dump(js, open(os.path.join(OUT, "rescore_reg2.json"), "w"), indent=1)
    print("->", os.path.join(OUT, "rescore_reg2.md"))


if __name__ == "__main__":
    main()
