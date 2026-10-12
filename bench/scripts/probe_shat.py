"""G2 (plan §9.2): s-hat fidelity diagnostic — does the learned surface
estimate track the physical surface? Inference only.

For each checkpoint: enable the DepthCollapse probe hook, run the val grid
tiles, record the tile-level mean of the per-pixel s_hat field, and compare it
against a reference s_ref = argmax of the Gaussian-smoothed (sigma=1.5 layers)
mean intensity depth profile of the same tile window (the meta.json depth_peak
construction, per tile). Both are in window coordinates [0, D).

Interpretation rule (pre-registered): Pearson r >= 0.5 and MAE <= 2 layers ->
'the estimator tracks the surface' (mechanism confirmed). r ~ 0 -> the surfrel
benefit is adaptive depth gating, not surface localization — also a finding.

Writes: <NEW_ROOT>/probe_shat/{probe_shat.json, figs/*.png}

Usage:  python -m scripts.probe_shat            # lean seeds 42/43/44
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import VAL_SEGMENT, proc_dir  # noqa: E402
from data.dataset import normalize_volume  # noqa: E402
from engine.infer import _tile_starts  # noqa: E402
from models.wrapper import build_model  # noqa: E402

OLD_ROOT = "/home/woody/iwi5/iwi5401h/vesuvius/outputs"
NEW_ROOT = os.environ.get("INK_OUTPUT_ROOT",
                          "/home/woody/iwi5/iwi5401h/vesuvius/outputs2")
DEFAULT_RUNS = ["lean_t192_s42", "lean_t192_s43", "lean_t192_s44"]


@torch.no_grad()
def probe_run(run, device, n_qual=3):
    ckpt = torch.load(os.path.join(OLD_ROOT, run, "best.pt"),
                      map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    d = cfg["data"]
    tile, depth = int(d["tile"]), int(d["depth"])
    center, n_layers = int(d.get("center_layer", 32)), int(d.get("n_layers", 65))
    z0 = max(0, min(center - depth // 2, n_layers - depth))

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.collapse.probe = True   # G2 hook (models/shared.py), eval-only stash

    proc = proc_dir(VAL_SEGMENT)
    vol = np.load(os.path.join(proc, "vol.npy"), mmap_mode="r")
    mask = np.load(os.path.join(proc, "mask.npy"))
    H, W = mask.shape
    stride = int(cfg.get("infer", {}).get("stride", tile // 2))
    coords = [(y, x) for y in _tile_starts(H, tile, stride)
              for x in _tile_starts(W, tile, stride)
              if mask[y:y + tile, x:x + tile].mean() > 0.5]

    amp_dtype = (torch.bfloat16 if cfg.get("train", {}).get("amp_dtype", "fp16") == "bf16"
                 else torch.float16)
    s_hat_l, s_ref_l, xs_l, qual = [], [], [], []
    for (y, x) in coords:
        xv = np.asarray(vol[z0:z0 + depth, y:y + tile, x:x + tile], dtype=np.float32)
        prof = xv.mean(axis=(1, 2))
        prof_s = ndimage.gaussian_filter1d(prof, sigma=1.5)
        s_ref = float(np.argmax(prof_s))
        xn = normalize_volume(xv, d.get("norm_mode", "global"), float(d.get("normalize", 255.0)))
        xb = torch.from_numpy(xn).unsqueeze(0).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=torch.cuda.is_available()):
            model(xb)
        sh = model.collapse.last_s_hat            # [1,1,H,W] (per-pixel) or [1,1]
        sh_map = sh.numpy().squeeze()
        s_hat_l.append(float(sh_map.mean()))
        s_ref_l.append(s_ref)
        xs_l.append(x)
        if len(qual) < n_qual and np.asarray(sh_map).ndim == 2:
            qual.append((y, x, sh_map.copy(), s_ref))

    s_hat = np.array(s_hat_l); s_ref = np.array(s_ref_l); xs = np.array(xs_l)
    r = float(np.corrcoef(s_hat, s_ref)[0, 1])
    mae = float(np.abs(s_hat - s_ref).mean())
    verdict = ("TRACKS the surface" if (r >= 0.5 and mae <= 2.0)
               else "does NOT track the surface (adaptive gating reading)")
    print(f"[{run}] tiles={len(s_hat)}  pearson r={r:.3f}  MAE={mae:.2f} layers  -> {verdict}")
    return {"run": run, "n_tiles": len(s_hat), "pearson_r": r, "mae_layers": mae,
            "verdict": verdict,
            "s_hat_mean": float(s_hat.mean()), "s_ref_mean": float(s_ref.mean()),
            "_arrays": (s_hat, s_ref, xs, qual)}


def main():
    global OLD_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--root", default=OLD_ROOT,
                    help="where the run folders (best.pt) live, read-only")
    args = ap.parse_args()
    OLD_ROOT = args.root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(NEW_ROOT, "probe_shat")
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    results = []
    for run in args.runs:
        res = probe_run(run, device)
        s_hat, s_ref, xs, qual = res.pop("_arrays")
        results.append(res)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(s_ref, s_hat, s=6, alpha=0.4)
        lo, hi = min(s_ref.min(), s_hat.min()) - 1, max(s_ref.max(), s_hat.max()) + 1
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[0].set_xlabel("s_ref (intensity depth peak, window coords)")
        axes[0].set_ylabel("s_hat (learned, tile mean)")
        axes[0].set_title(f"{res['run']}  r={res['pearson_r']:.3f} MAE={res['mae_layers']:.2f}")
        order = np.argsort(xs)
        axes[1].plot(xs[order], s_ref[order], ".", ms=3, alpha=0.5, label="s_ref")
        axes[1].plot(xs[order], s_hat[order], ".", ms=3, alpha=0.5, label="s_hat")
        axes[1].set_xlabel("tile x (val segment)")
        axes[1].set_ylabel("depth (window coords)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figs", f"{res['run']}_shat.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

        for i, (y, x, sh_map, sref) in enumerate(qual):
            fig2, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(sh_map, cmap="coolwarm")
            fig2.colorbar(im, ax=ax, label="s_hat(h,w)")
            ax.set_title(f"{res['run']} tile(y={y},x={x}) s_ref={sref:.0f}")
            ax.axis("off")
            fig2.savefig(os.path.join(out_dir, "figs",
                                      f"{res['run']}_spix_{i}.png"), dpi=130,
                         bbox_inches="tight")
            plt.close(fig2)

    with open(os.path.join(out_dir, "probe_shat.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {out_dir}/probe_shat.json")


if __name__ == "__main__":
    main()
