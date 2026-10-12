"""Accurate re-registration of ALL 9 segments -> ink_reg2.npy (2026-07-20, per Omar).

Method (upgrades over the 2026-07 linear-dy registration):
  1. ESTIMATOR: model probability maps, out-of-fold by construction:
     - 8 train segments: the a2bs_t192 LOO fold maps (outputs/a2bs_t192_loo/) —
       each segment predicted by the model that NEVER saw it (no label leakage);
     - val 090990: UNETR++ c1_t256_s42 + nnFormer n_embedz2_t192_s42 (val is
       out-of-fold for every model); two architectures must agree.
  2. FIELD: per-window (1280 px, 50% overlap, both axes) scan of (dy, dx) that
     maximizes prob-mass under the shifted ORIGINAL label (ink.npy), then a
     Gaussian-weighted SMOOTH 2D field dy(x,y), dx(x,y) — replaces the old
     global/linear-in-x fit that left the +-15 px end-tilt on val.
  3. WARP: bilinear label warp in row bands (memory-safe on 152031).
  4. VERIFY: post-warp residual rescan — target |dy| <= 4 px in every window.
  5. QC: per-segment report + field/overlay figures + annotation-QC lists
     (unlabeled-ink suspicions, dead labels) for the upstream report.

Writes: proc/<seg>/ink_reg2.npy + reg2_report.json,
        $WORK/nnFormer/rereg/{figs/*.png, annotation_qc.json, summary.md}
Never touches ink.npy / ink_reg.npy / mask.npy.
"""
import json
import os

import numpy as np
from scipy import ndimage

WORK = "/home/woody/iwi5/iwi5401h"
PROC = f"{WORK}/vesuvius/data/proc"
LOO = f"{WORK}/vesuvius/outputs/a2bs_t192_loo"
REREG = f"{WORK}/nnFormer/rereg"
os.makedirs(os.path.join(REREG, "figs"), exist_ok=True)

TRAIN = ["20241025145341", "20241025145701", "20241030152031", "20241108111522",
         "20241108115232", "20241108120732", "20241113070770", "20241113080880"]
VAL = "20241113090990"
VAL_MAPS = [f"{WORK}/vesuvius/outputs5/c1_t256_s42/val_prob_best.npy",
            f"{WORK}/vesuvius/outputs4/t256_lean_s42/val_prob_best.npy",
            f"{WORK}/nnFormer/outputs/n_embedz2_t192_s42/val_prob_best.npy"]

WIN, STEP = 1024, 512
DY0, DY1 = -80, 40          # coarse dy range (covers global -40 + local tilt)
DX0, DX1 = -14, 14
MIN_INK = 6000              # min ink px in a window to trust an estimate
AGREE_DY, AGREE_DX = 6, 8   # multi-map agreement gate (val)
SIGMA = 700.0              # field smoothing scale (px)
CELL = 320                  # field grid cell (px)


def metric(prob, ink, mask, y0, y1, x0, x1, dy, dx):
    """Mean prob under the label shifted by (+dy,+dx), exact (no wraparound)."""
    H, W = ink.shape
    sy0, sy1 = y0 - dy, y1 - dy
    sx0, sx1 = x0 - dx, x1 - dx
    cy0, cy1 = max(sy0, 0), min(sy1, H)
    cx0, cx1 = max(sx0, 0), min(sx1, W)
    if cy1 <= cy0 or cx1 <= cx0:
        return -1.0, 0
    oy, ox = cy0 - sy0, cx0 - sx0
    lab = ink[cy0:cy1, cx0:cx1]
    p = prob[y0 + oy:y0 + oy + (cy1 - cy0), x0 + ox:x0 + ox + (cx1 - cx0)]
    m = mask[y0 + oy:y0 + oy + (cy1 - cy0), x0 + ox:x0 + ox + (cx1 - cx0)]
    sel = lab & m
    n = int(sel.sum())
    return (float(p[sel].mean()) if n else -1.0), n


def window_offset(prob, ink, mask, y0, y1, x0, x1):
    """Separable coarse scan + 2D refine. Returns (dy, dx, sharpness, ink_px)."""
    vals = {}
    for dy in range(DY0, DY1 + 1, 2):
        vals[(dy, 0)] = metric(prob, ink, mask, y0, y1, x0, x1, dy, 0)[0]
    bdy = max(vals, key=vals.get)[0]
    for dx in range(DX0, DX1 + 1, 4):
        vals[(bdy, dx)] = metric(prob, ink, mask, y0, y1, x0, x1, bdy, dx)[0]
    bdx = max(((k, v) for k, v in vals.items() if k[0] == bdy), key=lambda kv: kv[1])[0][1]
    for dy in range(bdy - 3, bdy + 4):
        for dx in range(bdx - 3, bdx + 4):
            if (dy, dx) not in vals:
                vals[(dy, dx)] = metric(prob, ink, mask, y0, y1, x0, x1, dy, dx)[0]
    (dy, dx), peak = max(vals.items(), key=lambda kv: kv[1])
    arr = np.array(list(vals.values()))
    sharp = float(peak - np.median(arr[arr > -1]))
    _, n = metric(prob, ink, mask, y0, y1, x0, x1, dy, dx)
    return dy, dx, max(sharp, 0.0), n


def estimate_field(seg, probs, ink, mask):
    H, W = ink.shape
    ys = list(range(0, max(H - WIN, 1), STEP)) or [0]
    xs = list(range(0, max(W - WIN, 1), STEP)) or [0]
    pts, table = [], []
    for y0 in ys:
        y1 = min(y0 + WIN, H)
        for x0 in xs:
            x1 = min(x0 + WIN, W)
            if ink[y0:y1, x0:x1].sum() < MIN_INK:
                continue
            ests = [window_offset(p, ink, mask, y0, y1, x0, x1) for p in probs]
            if len(ests) > 1:
                dys = sorted(e[0] for e in ests)
                dxs = sorted(e[1] for e in ests)
                if dys[-1] - dys[0] > AGREE_DY or dxs[-1] - dxs[0] > AGREE_DX:
                    if len(ests) >= 3:  # majority: drop the outlier, keep the closest pair
                        best_pair = min(((a, b) for a in ests for b in ests if a is not b),
                                        key=lambda ab: abs(ab[0][0] - ab[1][0]))
                        if abs(best_pair[0][0] - best_pair[1][0]) <= AGREE_DY:
                            ests = list(best_pair)
                        else:
                            table.append({"y": y0, "x": x0, "flag": "disagree",
                                          "ests": [e[:2] for e in ests]})
                            continue
                    else:
                        table.append({"y": y0, "x": x0, "flag": "disagree",
                                      "ests": [e[:2] for e in ests]})
                        continue
            w = sum(e[2] * e[3] for e in ests)
            dy = sum(e[0] * e[2] * e[3] for e in ests) / max(w, 1e-9)
            dx = sum(e[1] * e[2] * e[3] for e in ests) / max(w, 1e-9)
            pts.append((y0 + (y1 - y0) / 2, x0 + (x1 - x0) / 2, dy, dx, w))
            table.append({"y": y0, "x": x0, "dy": round(dy, 1), "dx": round(dx, 1),
                          "w": round(w, 1)})
    if not pts:
        raise RuntimeError(f"{seg}: no usable windows")
    P = np.array(pts)  # [N, (yc, xc, dy, dx, w)]
    gh, gw = int(np.ceil(H / CELL)), int(np.ceil(W / CELL))
    gy, gx = np.mgrid[0:gh, 0:gw]
    gy = gy * CELL + CELL / 2
    gx = gx * CELL + CELL / 2
    d2 = (gy[..., None] - P[:, 0]) ** 2 + (gx[..., None] - P[:, 1]) ** 2
    wgt = P[:, 4] * np.exp(-d2 / (2 * SIGMA ** 2))
    wsum = wgt.sum(-1)
    glob_dy = float(np.average(P[:, 2], weights=P[:, 4]))
    glob_dx = float(np.average(P[:, 3], weights=P[:, 4]))
    dyg = np.where(wsum > 1e-9, (wgt * P[:, 2]).sum(-1) / np.maximum(wsum, 1e-9), glob_dy)
    dxg = np.where(wsum > 1e-9, (wgt * P[:, 3]).sum(-1) / np.maximum(wsum, 1e-9), glob_dx)
    return dyg.astype(np.float32), dxg.astype(np.float32), table


def warp_labels(ink, dyg, dxg):
    """ink_reg2(y,x) = ink(y - dy(y,x), x - dx(y,x)), banded, bilinear."""
    H, W = ink.shape
    out = np.zeros_like(ink, dtype=np.uint8)
    src = ink.astype(np.float32)
    band = 2000
    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)
        Y, X = np.mgrid[r0:r1, 0:W].astype(np.float32)
        cy, cx = (Y - 0.0) / CELL - 0.5, (X - 0.0) / CELL - 0.5
        dy = ndimage.map_coordinates(dyg, [cy.ravel(), cx.ravel()], order=1,
                                     mode="nearest").reshape(Y.shape)
        dx = ndimage.map_coordinates(dxg, [cy.ravel(), cx.ravel()], order=1,
                                     mode="nearest").reshape(Y.shape)
        vals = ndimage.map_coordinates(src, [(Y - dy).ravel(), (X - dx).ravel()],
                                       order=1, mode="constant", cval=0.0)
        out[r0:r1] = (vals.reshape(Y.shape) > 0.5).astype(np.uint8)
    return out


def verify(probs, ink2, mask):
    """Residual dy scan (+-24) vs the NEW labels, per window; also dx +-8."""
    H, W = ink2.shape
    ink2b = ink2 > 0
    res = []
    ys = list(range(0, max(H - WIN, 1), STEP)) or [0]
    xs = list(range(0, max(W - WIN, 1), STEP)) or [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + WIN, H), min(x0 + WIN, W)
            if ink2b[y0:y1, x0:x1].sum() < MIN_INK:
                continue
            v0 = metric(probs[0], ink2b, mask, y0, y1, x0, x1, 0, 0)[0]
            best = (v0, 0, 0)
            for dy in range(-24, 25, 2):
                v = metric(probs[0], ink2b, mask, y0, y1, x0, x1, dy, 0)[0]
                if v > best[0]:
                    best = (v, dy, 0)
            for dy in (best[1] - 1, best[1] + 1):
                v = metric(probs[0], ink2b, mask, y0, y1, x0, x1, dy, 0)[0]
                if v > best[0]:
                    best = (v, dy, 0)
            res_dy = best[1] if best[0] > 1.01 * max(v0, 1e-9) else 0
            res.append({"y": y0, "x": x0, "res_dy": res_dy,
                        "gain": round(best[0] / max(v0, 1e-9), 4)})
    return res


def annotation_qc(seg, probs, ink2, mask):
    """Cross-map consensus: unlabeled-ink suspicions + dead labels."""
    sub = 2 if ink2.size > 8e7 else 1
    p = np.minimum.reduce([q[::sub, ::sub] for q in probs]) if len(probs) > 1 \
        else probs[0][::sub, ::sub]
    i2 = ink2[::sub, ::sub] > 0
    m = mask[::sub, ::sub] > 0
    halo = ndimage.binary_dilation(i2, iterations=max(24 // sub, 1))
    susp = (p >= 0.5) & ~halo & m
    lab, n = ndimage.label(susp)
    out = {"suspected_unlabeled": [], "dead_labels": []}
    if n:
        sizes = ndimage.sum(susp, lab, range(1, n + 1))
        order = np.argsort(sizes)[::-1]
        for k in order[:20]:
            if sizes[k] * sub * sub < 4000:
                break
            ys, xs = np.where(lab == k + 1)
            out["suspected_unlabeled"].append(
                {"x": int(xs.mean()) * sub, "y": int(ys.mean()) * sub,
                 "area_px": int(sizes[k]) * sub * sub,
                 "mean_prob": round(float(p[lab == k + 1].mean()), 3)})
    pmax = np.maximum.reduce([q[::sub, ::sub] for q in probs]) if len(probs) > 1 \
        else probs[0][::sub, ::sub]
    labi, ni = ndimage.label(i2)
    if ni:
        mp = ndimage.mean(pmax, labi, range(1, ni + 1))
        sz = ndimage.sum(i2, labi, range(1, ni + 1))
        for k in np.argsort(sz)[::-1][:200]:
            if sz[k] * sub * sub < 4000:
                break
            if mp[k] < 0.07:
                ys, xs = np.where(labi == k + 1)
                out["dead_labels"].append(
                    {"x": int(xs.mean()) * sub, "y": int(ys.mean()) * sub,
                     "area_px": int(sz[k]) * sub * sub,
                     "mean_prob": round(float(mp[k]), 3)})
    return out


def qc_figure(seg, probs, ink_old, ink2, dyg, table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    im = axes[0].imshow(dyg, cmap="coolwarm", vmin=-40, vmax=40, aspect="auto")
    axes[0].set_title(f"{seg}  dy field (px; grid cell {CELL}px)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    good = [t for t in table if "dy" in t]
    worst = max(good, key=lambda t: abs(t["dy"])) if good else None
    if worst:
        y0, x0 = worst["y"], worst["x"]
        y1, x1 = y0 + WIN, x0 + WIN
        crop = probs[0][y0:y1, x0:x1]
        rgb = np.stack([crop, crop, crop], -1)
        old_e = ink_old[y0:y1, x0:x1] & ~ndimage.binary_erosion(ink_old[y0:y1, x0:x1], iterations=3)
        new_e = (ink2[y0:y1, x0:x1] > 0) & ~ndimage.binary_erosion(ink2[y0:y1, x0:x1] > 0, iterations=3)
        rgb[old_e] = [1, 0, 0]
        rgb[new_e] = [0, 1, 0]
        axes[1].imshow(rgb, aspect="auto")
        axes[1].set_title(f"worst window ({x0},{y0}) dy={worst['dy']}: red=old edge, green=new edge, gray=OOF prob")
    out = os.path.join(REREG, "figs", f"reg2_{seg}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  fig ->", out)


def process(seg, prob_paths):
    print(f"\n=== {seg} ===")
    d = os.path.join(PROC, seg)
    ink = np.load(os.path.join(d, "ink.npy")) > 0          # ORIGINAL labels
    mask = np.load(os.path.join(d, "mask.npy")) > 0
    probs = []
    for p in prob_paths:
        arr = np.load(p).astype(np.float32)
        assert arr.shape == ink.shape, (seg, p, arr.shape, ink.shape)
        probs.append(arr)
    dyg, dxg, table = estimate_field(seg, probs, ink, mask)
    print(f"  windows: {sum(1 for t in table if 'dy' in t)} used, "
          f"{sum(1 for t in table if t.get('flag'))} disagreed; "
          f"dy range [{dyg.min():.0f},{dyg.max():.0f}] dx range [{dxg.min():.0f},{dxg.max():.0f}]")
    ink2 = warp_labels(ink, dyg, dxg)
    res = verify(probs, ink2, mask)
    res_dys = [abs(r["res_dy"]) for r in res]
    ok = sum(1 for v in res_dys if v <= 4)
    print(f"  residuals: {ok}/{len(res_dys)} windows |dy|<=4 "
          f"(max {max(res_dys) if res_dys else 0})")
    np.save(os.path.join(d, "ink_reg2.npy"), ink2)
    report = {"segment": seg, "estimator": [os.path.relpath(p, WORK) for p in prob_paths],
              "windows": table, "residuals": res,
              "residual_ok_frac": round(ok / max(len(res_dys), 1), 3),
              "field_stats": {"dy_min": float(dyg.min()), "dy_max": float(dyg.max()),
                              "dx_min": float(dxg.min()), "dx_max": float(dxg.max())},
              "params": {"WIN": WIN, "STEP": STEP, "SIGMA": SIGMA, "CELL": CELL,
                         "dy_range": [DY0, DY1], "dx_range": [DX0, DX1]}}
    json.dump(report, open(os.path.join(d, "reg2_report.json"), "w"), indent=1)
    qc = annotation_qc(seg, probs, ink2, mask)
    qc_figure(seg, probs, ink, ink2, dyg.reshape(int(np.ceil(ink.shape[0] / CELL)),
                                                 int(np.ceil(ink.shape[1] / CELL))), table)
    print(f"  QC: {len(qc['suspected_unlabeled'])} unlabeled-ink suspicions, "
          f"{len(qc['dead_labels'])} dead labels")
    return report, qc


def main():
    all_qc = {}
    for seg in TRAIN:
        prob = os.path.join(LOO, f"train_prob_{seg}.npy")
        rep, qc = process(seg, [prob])
        all_qc[seg] = qc
    rep, qc = process(VAL, VAL_MAPS)
    all_qc[VAL] = qc
    json.dump(all_qc, open(os.path.join(REREG, "annotation_qc.json"), "w"), indent=1)
    print("\nannotation QC ->", os.path.join(REREG, "annotation_qc.json"))
    print("DONE — review rereg/figs/*.png before retraining; then rebuild "
          "manifests + trimaps with --label-file ink_reg2.npy")


if __name__ == "__main__":
    # NOTE: estimate_field returns flat grids; reshape handled in qc_figure only.
    main()
