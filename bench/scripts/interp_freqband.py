"""Interpretability probe 5 (plan §14.4): frequency-band occlusion — which
in-plane texture scales carry the ink evidence (the crackle hypothesis,
quantified). Perturbation-based, model-agnostic, no training.

For each band: 2D FFT band-stop filter applied to layers [16,48) of the val
volume (per depth slice), volume re-quantized to uint8, full sliding-window
inference, F0.5 at the model's frozen tau + re-swept. A no-filter uint8
round-trip control isolates quantization from band effects.

Bands are period ranges in px/cycle. DC is always kept (removing global
brightness would trivially destroy the input).

Usage:  python -m scripts.interp_freqband --root .../outputs4 --run t256_lean_s42
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import VAL_SEGMENT, proc_dir  # noqa: E402
from engine.eval import pixel_metrics, sweep_threshold  # noqa: E402
from engine.infer import infer_segment  # noqa: E402
from models.wrapper import build_model  # noqa: E402

W = "/home/woody/iwi5/iwi5401h/vesuvius"
BANDS = [(2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, None)]


def bandstop(vol_layers, pmin, pmax):
    """Zero the radial frequency annulus with periods [pmin, pmax) px. DC kept."""
    D, H, W_ = vol_layers.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W_)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    f_hi = 1.0 / pmin
    f_lo = 0.0 if pmax is None else 1.0 / pmax
    kill = (fr >= f_lo) & (fr < f_hi)
    kill[0, 0] = False                      # never remove DC
    out = np.empty_like(vol_layers)
    for z in range(D):
        F = np.fft.fft2(vol_layers[z].astype(np.float32))
        F[kill] = 0
        out[z] = np.clip(np.fft.ifft2(F).real, 0, 255)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{W}/outputs4")
    ap.add_argument("--run", default="t256_lean_s42")
    ap.add_argument("--out", default=f"{W}/output_inter")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rd = os.path.join(args.root, args.run)
    ckpt = torch.load(os.path.join(rd, "best.pt"), map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    tau = float(json.load(open(os.path.join(rd, "summary.json")))["best_tau"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    proc = proc_dir(VAL_SEGMENT)
    gt = np.load(os.path.join(proc, "ink_reg.npy")) > 0
    mask = np.load(os.path.join(proc, "mask.npy")) > 0
    vol = np.load(os.path.join(proc, "vol.npy"))            # full load, ~1.2 GB
    z0, z1 = 16, 48

    tmp = tempfile.mkdtemp(prefix="freqband_", dir=os.environ.get("TMPDIR", "/tmp"))
    tmp_seg = os.path.join(tmp, VAL_SEGMENT)
    os.makedirs(tmp_seg, exist_ok=True)
    for f in ("mask.npy", "ink_reg.npy"):
        os.symlink(os.path.join(proc, f), os.path.join(tmp_seg, f))
    tmp_proc_dir = lambda seg: tmp_seg  # noqa: E731

    def run_once(vol_arr, name):
        np.save(os.path.join(tmp_seg, "vol.npy"), vol_arr)
        prob, m, _ = infer_segment(model, VAL_SEGMENT, cfg, device, tmp_proc_dir)
        frozen = pixel_metrics(prob, gt, mask, tau)
        swept, _ = sweep_threshold(prob, gt, mask)
        print(f"[{name}] frozen@tau{tau:.2f} F0.5={frozen['fbeta']:.4f}  "
              f"swept={swept['fbeta']:.4f}@tau{swept['tau']:.2f}")
        return {"band": name, "frozen_f05": frozen["fbeta"], "swept_f05": swept["fbeta"]}

    results = [run_once(vol, "control_uint8")]
    base = results[0]["frozen_f05"]
    for pmin, pmax in BANDS:
        v = vol.copy()
        v[z0:z1] = bandstop(vol[z0:z1].astype(np.float32), pmin, pmax).astype(np.uint8)
        name = f"stop_{pmin}-{pmax if pmax else 'inf'}px"
        results.append(run_once(v, name))

    out_dir = os.path.join(args.out, "freqband")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.run}_freqband.json"), "w") as f:
        json.dump({"run": args.run, "tau": tau, "results": results}, f, indent=2)

    labels = [r["band"] for r in results[1:]]
    drops = [base - r["frozen_f05"] for r in results[1:]]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(drops)), drops, color="tab:red")
    ax.set_xticks(range(len(drops)))
    ax.set_xticklabels([l.replace("stop_", "") for l in labels], rotation=30, fontsize=8)
    ax.set_ylabel(f"F0.5 drop vs control ({base:.3f})")
    ax.set_title(f"{args.run} — frequency-band occlusion (removed period band)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{args.run}_freqband.png"), dpi=130,
                bbox_inches="tight")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"-> {out_dir}/{args.run}_freqband.png")


if __name__ == "__main__":
    main()
