"""Run Inkalyzer input-space IBA attribution on trained ink models.

For each run, samples ink-rich tiles on the validation segment and runs depth-axis
IBA (attribute_z) to recover the per-depth-layer importance profile the model relies
on, plus 3D IBA (attribute_3d) on a few tiles for a spatial saliency volume.  Produces
per-run npy + figures and a cross-model comparison of the depth profiles.

Methodology target (vesuvius_methodology_plan §Cross-scroll / Figures):
  "Inkalyzer saliency on A0s vs A2/A2b vs the weakest model — does the depth band
   become more surface-concentrated?"  -> depth-profile concentration table below.

Usage:
  python -m scripts.inkalyzer_run --runs a0,a0s,a2b,a2bs --n-tiles 24 --n-3d 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from data.dataset import normalize_volume                              # noqa: E402
from data.segments import VAL_SEGMENT, proc_dir, OUTPUT_ROOT           # noqa: E402
from engine.losses import dice_bce                                     # noqa: E402
from models.wrapper import build_model                                 # noqa: E402
from xai.ink_iba import attribute_z, attribute_3d, get_noise           # noqa: E402


def load_model(run, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, ckpt


def select_ink_tiles(gt, mask, tile, n_tiles, ink_thr=0.06, seed=42):
    """Deterministic ink-rich, spatially-spread tile coords on a grid (stride=tile)."""
    H, W = mask.shape
    cands = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            m = mask[y:y + tile, x:x + tile]
            if m.mean() < 0.99:
                continue
            frac = (gt[y:y + tile, x:x + tile] > 0.5).mean()
            if frac >= ink_thr:
                cands.append((frac, y, x))
    cands.sort(reverse=True)
    # take the top tiles but drop near-duplicates that overlap heavily
    picked = []
    for frac, y, x in cands:
        if all(abs(y - py) >= tile or abs(x - px) >= tile for _, py, px in picked):
            picked.append((frac, y, x))
        if len(picked) >= n_tiles:
            break
    return picked


def depth_concentration(profile):
    """Summarise a per-layer importance vector: peak, mass-in-band, normalized entropy."""
    p = np.asarray(profile, dtype=np.float64)
    w = p / (p.sum() + 1e-9)
    peak = int(np.argmax(p))
    lo, hi = max(0, peak - 2), min(len(p), peak + 3)
    mass_pm2 = float(w[lo:hi].sum())
    ent = float(-(w * np.log(w + 1e-12)).sum() / np.log(len(p)))  # 0=peaky, 1=flat
    return {"peak_idx": peak, "mass_within_pm2": mass_pm2, "norm_entropy": ent}


def run_one(run, args, device, autocast_dtype, z0, vol, mask, gt, tile_cache):
    ckpt_path = os.path.join(OUTPUT_ROOT, run, "best.pt")
    model, cfg, ckpt = load_model(run, ckpt_path, device)
    tile = int(cfg["data"]["tile"])
    depth = int(cfg["data"]["depth"])
    norm = float(cfg["data"].get("normalize", 255.0))
    norm_mode = cfg["data"].get("norm_mode", "global")
    # tiles are selected once per tile size (shared across models of that size,
    # so within-tile comparisons use the identical windows).
    if tile not in tile_cache:
        tile_cache[tile] = select_ink_tiles(gt, mask, tile, args.n_tiles, ink_thr=args.ink_thr)
    tiles = tile_cache[tile]
    print(f"[{run}] tile={tile} depth={depth} epoch={ckpt.get('epoch')} "
          f"f05={ckpt.get('best_metric'):.4f}  {len(tiles)} tiles")

    out_dir = os.path.join(OUTPUT_ROOT, run, "inkalyzer")
    os.makedirs(out_dir, exist_ok=True)
    loss_func = lambda pred, y: dice_bce(pred, y)  # noqa: E731  preserve model's own map

    gates_all, meta = [], []
    for i, (frac, y, x) in enumerate(tiles):
        xv = normalize_volume(np.asarray(vol[z0:z0 + depth, y:y + tile, x:x + tile],
                                         dtype=np.float32), norm_mode, norm)
        data = torch.from_numpy(xv).unsqueeze(0).unsqueeze(0).to(device)   # [1,1,D,t,t]
        noise = get_noise(data, args.noise).repeat(1, 1, depth, 1, 1)
        g, _, _ = attribute_z(data, noise, model, loss_func, n_steps=args.n_steps,
                              lr=args.lr, beta=args.beta, size=tile, depth=depth,
                              autocast_dtype=autocast_dtype)
        prof = g[0, 0, :, 0, 0]                                            # [D]
        gates_all.append(prof)
        meta.append({"tile": i, "y": int(y), "x": int(x), "ink_frac": float(frac)})

        if i < args.n_3d:
            sal, fpred = attribute_3d(data, noise, model, loss_func, n_steps=args.n_steps_3d,
                                      lr=args.lr, beta=args.beta, size=tile,
                                      window_size=args.window_size, depth=depth,
                                      autocast_dtype=autocast_dtype)
            np.save(os.path.join(out_dir, f"attr3d_tile{i}.npy"), sal.astype(np.float32))
            _save_3d_fig(out_dir, i, xv, gt[y:y + tile, x:x + tile], sal, fpred, z0)

    gates_all = np.stack(gates_all)                                       # [n,D]
    np.save(os.path.join(out_dir, "depth_gates.npy"), gates_all.astype(np.float32))
    mean_prof = gates_all.mean(0)
    conc = depth_concentration(mean_prof)
    conc["peak_layer_abs"] = z0 + conc["peak_idx"]
    conc["tile"] = tile
    conc["f05"] = float(ckpt.get("best_metric") or 0.0)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"run": run, "n_tiles": len(tiles), "noise": args.noise,
                   "n_steps": args.n_steps, "beta": args.beta, "lr": args.lr,
                   "depth": depth, "z0": z0, "mean_profile": mean_prof.tolist(),
                   "concentration": conc, "tiles": meta}, f, indent=2)
    _save_profile_fig(out_dir, run, z0, gates_all)
    print(f"[{run}] peak layer={conc['peak_layer_abs']} mass±2={conc['mass_within_pm2']:.3f} "
          f"entropy={conc['norm_entropy']:.3f}  -> {out_dir}")
    return z0, mean_prof, conc


def _save_profile_fig(out_dir, run, z0, gates_all):
    layers = z0 + np.arange(gates_all.shape[1])
    m, s = gates_all.mean(0), gates_all.std(0)
    plt.figure(figsize=(7, 4))
    for row in gates_all:
        plt.plot(layers, row, color="0.8", lw=0.6)
    plt.plot(layers, m, color="C0", lw=2, label="mean gate")
    plt.fill_between(layers, m - s, m + s, color="C0", alpha=0.2)
    plt.xlabel("depth layer index"); plt.ylabel("IBA gate (importance)")
    plt.title(f"{run}: depth attribution profile"); plt.ylim(0, 1.02); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "depth_profile.png"), dpi=130)
    plt.close()


def _save_3d_fig(out_dir, i, xv, gt_tile, sal, fpred, z0):
    sal_v = sal[0, 0]                                    # [D,t,t]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(np.median(xv, axis=0), cmap="gray"); ax[0].set_title("input (median-z)")
    ax[1].imshow(gt_tile, cmap="gray"); ax[1].set_title("GT ink")
    ax[2].imshow(fpred[0, 0], cmap="magma"); ax[2].set_title("bottlenecked pred")
    ax[3].imshow(sal_v.max(0), cmap="magma"); ax[3].set_title("3D saliency (z-max)")
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"attr3d_tile{i}.png"), dpi=120)
    plt.close()


def _save_compare_fig(runs, z0, profiles, concs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for run, prof, c in zip(runs, profiles, concs):
        layers = z0 + np.arange(len(prof))
        plt.plot(layers, prof, lw=2, marker="o", ms=3, label=f"{run} (t{c['tile']})")
    plt.xlabel("depth layer index"); plt.ylabel("mean IBA gate (importance)")
    plt.title("Inkalyzer depth attribution — model comparison")
    plt.ylim(0, 1.02); plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "depth_profiles.png"), dpi=140)
    plt.close()
    with open(os.path.join(out_dir, "concentration.json"), "w") as f:
        json.dump({run: c for run, c in zip(runs, concs)}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names (outputs/<run>/best.pt)")
    ap.add_argument("--seg", default=VAL_SEGMENT)
    ap.add_argument("--label-file", default="ink_reg.npy")
    ap.add_argument("--n-tiles", type=int, default=24)
    ap.add_argument("--ink-thr", type=float, default=0.06)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--n-steps-3d", type=int, default=20)
    ap.add_argument("--n-3d", type=int, default=3, help="#tiles to also run 3D attribution on")
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--noise", default="median", choices=["median", "mean", "top85", "top", "bottom", "max"])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-1)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--compare-dir", default=os.path.join(OUTPUT_ROOT, "inkalyzer_compare"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = None if (args.no_amp or device == "cpu") else torch.bfloat16
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    # tile geometry is shared across the ladder (tile=64, depth=32, center=32).
    proc = proc_dir(args.seg)
    vol = np.load(os.path.join(proc, "vol.npy"), mmap_mode="r")           # [L,H,W]
    mask = np.load(os.path.join(proc, "mask.npy"))
    gt = np.load(os.path.join(proc, args.label_file)).astype(np.float32)
    if gt.max() > 1.5:
        gt = (gt > 127).astype(np.float32)

    # depth geometry is uniform across the ladder (depth=32, center=32 -> z0=16);
    # only the spatial tile size differs, so tiles are chosen per size (tile_cache).
    ref = torch.load(os.path.join(OUTPUT_ROOT, runs[0], "best.pt"),
                     map_location="cpu", weights_only=False)["cfg"]["data"]
    depth, center, n_layers = (int(ref["depth"]), int(ref.get("center_layer", 32)),
                               int(ref.get("n_layers", 65)))
    z0 = max(0, min(center - depth // 2, n_layers - depth))
    print(f"[inkalyzer] seg={args.seg} device={device} amp={autocast_dtype} "
          f"z0={z0} depth={depth} runs={runs}")

    tile_cache, profiles, concs = {}, [], []
    for run in runs:
        _, mean_prof, conc = run_one(run, args, device, autocast_dtype, z0, vol, mask, gt, tile_cache)
        profiles.append(mean_prof); concs.append(conc)

    if len(runs) > 1:
        _save_compare_fig(runs, z0, profiles, concs, args.compare_dir)
        print(f"[inkalyzer] comparison -> {args.compare_dir}/depth_profiles.png")


if __name__ == "__main__":
    main()
