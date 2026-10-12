"""Re-register ink labels to the flattened volume (per-segment smooth dy(x) + scalar dx).

FINDING (2026-07-02): on all 9 labelled segments the inklabels sit 10-45 px BELOW
the physical ink signal in the volume (checked by maximizing detrended central-slab
ink-vs-papyrus contrast over label shifts; stable across depth bands, so it is a
true in-plane misregistration of the 2D label map, not sheet tilt). On the val
segment the offset varies smoothly along x (-45 mid-left .. -25 right), so a global
scalar is not enough.

This tool estimates, for one segment:
  * the best depth band (strongest aligned contrast among candidates),
  * per-x-window optimal (dy, dx) by exhaustive scan of detrended contrast,
  * a confidence-weighted linear fit dy(x) (deg<=1 keeps the warp conservative),
    and a confidence-weighted median scalar dx,
and writes:
  * ink_reg.npy   — labels warped by integer per-column dy(x) (+ global dx),
  * reg_report.json — everything needed to audit/reproduce the fit.

The ORIGINAL ink.npy is never modified. Registration uses only volume+labels
(no model predictions), so it is a preprocessing step, not test-time leakage.

Usage:
    python -m scripts.register_labels --segments 20241113090990
    python -m scripts.register_labels            # all 9 labelled segments
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import SEGMENTS, proc_dir  # noqa: E402

DY_RANGE = range(-70, 26, 5)          # coarse scan
DX_RANGE = range(-30, 31, 10)
BANDS = [(22, 32), (26, 36), (28, 38), (30, 40), (34, 44)]
WIN = 2500                            # x-window width
STEP = 1250                           # x-window step
MIN_INK = 20000                       # min ink px in a window to trust its fit


def detrended_slab(vol, mask, zlo, zhi, x0, x1, xstep=1):
    slab = np.asarray(vol[zlo:zhi, :, x0:x1:xstep], dtype=np.float32).mean(axis=0)
    mk = np.asarray(mask[:, x0:x1:xstep]) > 0
    mk &= slab > 10                    # drop air/holes
    s = slab.copy()
    s[~mk] = np.nan
    s -= np.nanmedian(s, axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        s -= np.nanmedian(s, axis=0, keepdims=True)
    return s, mk


def contrast(s, mk, ik):
    sel_in = ik & mk
    sel_out = (~ik) & mk
    if sel_in.sum() < 500 or sel_out.sum() < 500:
        return np.nan
    return float(np.nanmean(s[sel_in]) - np.nanmean(s[sel_out]))


def pick_band(vol, ink, mask):
    W = ink.shape[1]
    x0, x1 = 0, min(W, 12000)         # cap work on huge segments
    best = None
    for (zlo, zhi) in BANDS:
        s, mk = detrended_slab(vol, mask, zlo, zhi, x0, x1)
        c = contrast(s, mk, np.asarray(ink[:, x0:x1]) > 0)
        if not np.isnan(c) and (best is None or c > best[0]):
            best = (c, zlo, zhi)
    return best[1], best[2], best[0]


def scan_window(s, mk, ik):
    """Exhaustive (dy, dx) scan; returns best shift, its contrast and the aligned contrast."""
    c0 = contrast(s, mk, ik)
    best = (c0 if not np.isnan(c0) else -1e9, 0, 0)
    for dy in DY_RANGE:
        ik_y = np.roll(ik, dy, axis=0)
        for dx in DX_RANGE:
            c = contrast(s, mk, np.roll(ik_y, dx, axis=1) if dx else ik_y)
            if not np.isnan(c) and c > best[0]:
                best = (c, dy, dx)
    # refine dy at best dx with step 2
    cb, dyb, dxb = best
    ik_x = np.roll(ik, dxb, axis=1) if dxb else ik
    for dy in range(max(min(DY_RANGE), dyb - 6), min(max(DY_RANGE), dyb + 6) + 1, 2):
        c = contrast(s, mk, np.roll(ik_x, dy, axis=0))
        if not np.isnan(c) and c > cb:
            cb, dyb = c, dy
    return cb, dyb, dxb, c0


def register_segment(seg, write=True):
    proc = proc_dir(seg)
    vol = np.load(os.path.join(proc, "vol.npy"), mmap_mode="r")
    ink = np.load(os.path.join(proc, "ink.npy"))
    mask = np.load(os.path.join(proc, "mask.npy"), mmap_mode="r")
    H, W = ink.shape
    zlo, zhi, cband = pick_band(vol, ink, mask)

    # --- GLOBAL anchor scan: whole segment (subsampled in x on huge segs).
    # Integrates all ink -> immune to the per-window text-line aliasing that
    # produces spurious +dy fits (e.g. one line-spacing away).  Per-window fits
    # are only trusted where they agree with this anchor.
    xstep = max(1, W // 12000)
    s_g, mk_g = detrended_slab(vol, mask, zlo, zhi, 0, W, xstep=xstep)
    ik_g = np.asarray(ink[:, ::xstep]) > 0
    cg, dy_g, dx_g, c0_g = scan_window(s_g, mk_g, ik_g)
    print(f"  [{seg}] GLOBAL dy={dy_g:+3d} dx={dx_g:+3d} contrast {c0_g:+.2f}->{cg:+.2f}")

    wins = []
    for x0 in range(0, max(1, W - WIN + 1), STEP):
        x1 = min(W, x0 + WIN)
        ik = np.asarray(ink[:, x0:x1]) > 0
        if ik.sum() < MIN_INK:
            continue
        s, mk = detrended_slab(vol, mask, zlo, zhi, x0, x1)
        cb, dy, dx, c0 = scan_window(s, mk, ik)
        wins.append({"xc": (x0 + x1) / 2.0, "dy": dy, "dx": dx,
                     "c_best": round(cb, 3), "c_aligned": round(float(c0), 3) if not np.isnan(c0) else None,
                     "weight": max(cb, 0.0)})
        print(f"  [{seg}] x[{x0},{x1}) dy={dy:+3d} dx={dx:+3d} contrast {c0:+.2f}->{cb:+.2f}")

    if not wins:
        print(f"  [{seg}] no ink windows — skipping")
        return None

    # --- robust fit, anchored on the GLOBAL scan ---------------------------
    # A window can lock onto a spurious alignment one text-line-spacing away
    # (periodicity aliasing), sometimes with HIGH contrast, so weighted-median
    # rejection is not enough (145341: dy=-70 window was the heaviest).  The
    # global scan integrates all ink and is immune; windows are inliers only
    # if they agree with it within ANCHOR_TOL px.
    ANCHOR_TOL = 25
    xcs = np.array([w["xc"] for w in wins])
    dys = np.array([w["dy"] for w in wins], dtype=np.float64)
    dxs = np.array([w["dx"] for w in wins], dtype=np.float64)
    ws = np.maximum(np.array([w["weight"] for w in wins], dtype=np.float64), 1e-3)

    def _wmedian(v, w):
        o = np.argsort(v)
        cw = np.cumsum(w[o])
        return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])

    inl = np.abs(dys - dy_g) <= ANCHOR_TOL
    n_rej = int((~inl).sum())
    for w, ok in zip(wins, inl):
        w["inlier"] = bool(ok)
    if n_rej:
        rej = [f"xc={int(w['xc'])} dy={w['dy']:+d}" for w, ok in zip(wins, inl) if not ok]
        print(f"  [{seg}] rejected {n_rej} window(s) vs global anchor dy={dy_g:+d}: {', '.join(rej)}")

    fit_mode = "linear"
    if inl.sum() >= 3:
        coef = np.polyfit(xcs[inl], dys[inl], 1, w=ws[inl])   # conservative: linear dy(x)
        dx_glob = int(round(_wmedian(dxs[inl], ws[inl])))
    elif inl.sum() >= 1:
        fit_mode = "scalar_inlier"
        coef = np.array([0.0, float(np.average(dys[inl], weights=ws[inl]))])
        dx_glob = int(round(_wmedian(dxs[inl], ws[inl])))
    else:                                   # windows all disagree with anchor
        fit_mode = "scalar_global"
        coef = np.array([0.0, float(dy_g)])
        dx_glob = dx_g

    cols = np.arange(W, dtype=np.float64)
    dy_col = np.rint(np.polyval(coef, cols)).astype(np.int32)
    # never extrapolate beyond the dy range actually observed at inlier windows
    # (the line can run wild outside [first,last] window on ink-sparse edges)
    if inl.any():
        lo = int(min(dys[inl].min(), dy_g)) - 5
        hi = int(max(dys[inl].max(), dy_g)) + 5
    else:
        lo, hi = dy_g - 5, dy_g + 5
    dy_col = np.clip(dy_col, lo, hi)

    # --- final acceptance test: does the fitted warp actually beat identity? --
    # Measured on the global detrended slab.  Segments that are already aligned
    # (e.g. 145341: global gain +0.2) must NOT be warped by noise fits.
    ik_warp = ik_g[np.clip(np.arange(H)[:, None] - dy_col[None, ::xstep], 0, H - 1),
                   np.arange(ik_g.shape[1])[None, :]]
    if dx_glob:
        ik_warp = np.roll(ik_warp, dx_glob // xstep if xstep > 1 else dx_glob, axis=1)
    c_warp = contrast(s_g, mk_g, ik_warp)
    gain = (c_warp if not np.isnan(c_warp) else -1e9) - (c0_g if not np.isnan(c0_g) else 0.0)
    GAIN_MIN = 0.4
    if gain < GAIN_MIN:
        fit_mode = "identity_low_gain"
        coef = np.array([0.0, 0.0])
        dy_col = np.zeros(W, dtype=np.int32)
        dx_glob = 0
        c_warp = c0_g
        print(f"  [{seg}] warp gain {gain:+.2f} < {GAIN_MIN} — labels already aligned, keeping identity")

    # per-column vertical shift (dy<0 moves labels UP), then global dx
    src_rows = np.arange(H)[:, None] - dy_col[None, :]     # ink_reg[y,x] = ink[y-dy(x), x+(-dx)]? see note
    # np.roll(ik, dy) maps ik_reg[y] = ik[y-dy]; replicate that per column:
    src_rows = np.clip(src_rows, 0, H - 1)
    ink_reg = ink[src_rows, np.arange(W)[None, :]]
    if dx_glob:
        ink_reg = np.roll(ink_reg, dx_glob, axis=1)

    report = {"segment": seg, "band": [zlo, zhi], "band_contrast": round(cband, 3),
              "global_anchor": {"dy": int(dy_g), "dx": int(dx_g),
                                "c_aligned": round(float(c0_g), 3) if not np.isnan(c0_g) else None,
                                "c_best": round(float(cg), 3)},
              "windows": wins, "n_outliers_rejected": n_rej, "fit_mode": fit_mode,
              "warp_contrast": round(float(c_warp), 3) if not np.isnan(c_warp) else None,
              "dy_poly_coef": [float(c) for c in coef],
              "dy_col_minmax": [int(dy_col.min()), int(dy_col.max())],
              "dx_global": dx_glob,
              "note": "ink_reg[y,x] = ink[y - dy(x), x] then roll by dx; dy<0 = labels moved up"}
    if write:
        np.save(os.path.join(proc, "ink_reg.npy"), ink_reg.astype(ink.dtype))
        with open(os.path.join(proc, "reg_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"  [{seg}] wrote ink_reg.npy  dy(x) {dy_col.min()}..{dy_col.max()}  dx={dx_glob}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="*", default=SEGMENTS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for seg in args.segments:
        print(f"[{seg}]")
        register_segment(seg, write=not args.dry_run)


if __name__ == "__main__":
    main()
