"""G1 (plan §9.1): depth-offset robustness stress test — inference only.

The geometry family was motivated by surface-offset robustness, but the offset
regime never existed in the data. Create it synthetically: evaluate existing
checkpoints on the val segment with the inference depth window shifted by
delta layers, and score at
  (a) FROZEN tau = the run's delta=0 swept tau (deployment-relevant), and
  (b) re-swept tau (adaptation upper bound).

Pre-registered predictions (write-up must quote them): surfrel-ON (lean)
degrades slower than surfrel-OFF (abl_surfrel) for |delta| >= 4; all models
hold roughly flat to |delta| = 2 (the z_shift training range). If OFF degrades
just as slowly, surfrel's +0.009 is NOT offset robustness — report that.

Writes: <NEW_ROOT>/zshift_stress/zshift_stress.json + figs/zshift_stress.png

Usage:  python -m scripts.stress_zshift
"""
import argparse
import copy
import json
import os
import sys

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

OLD_ROOT = "/home/woody/iwi5/iwi5401h/vesuvius/outputs"
NEW_ROOT = os.environ.get("INK_OUTPUT_ROOT",
                          "/home/woody/iwi5/iwi5401h/vesuvius/outputs2")

DEFAULT_RUNS = ["lean_t192_s42", "lean_t192_s43", "lean_t192_s44",
                "abl_surfrel", "a2bs_t192", "a2bs_t192_s43", "a2bs_t192_s44",
                "abl_da"]
GROUPS = {"lean_t192_s42": "surfrel-ON (lean)", "lean_t192_s43": "surfrel-ON (lean)",
          "lean_t192_s44": "surfrel-ON (lean)", "abl_surfrel": "surfrel-OFF",
          "a2bs_t192": "full (a2bs_t192)", "a2bs_t192_s43": "full (a2bs_t192)",
          "a2bs_t192_s44": "full (a2bs_t192)", "abl_da": "depth-attn OFF",
          "t256_lean_s42": "surfrel-ON t256", "t256_lean_s43": "surfrel-ON t256",
          "t256_lean_s44": "surfrel-ON t256", "t256_nosurf_s42": "surfrel-OFF t256",
          "t320_lean_s42": "t320"}
DEFAULT_DELTAS = [-8, -6, -4, -2, 0, 2, 4, 6, 8]


def main():
    global OLD_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--deltas", nargs="*", type=int, default=DEFAULT_DELTAS)
    ap.add_argument("--root", default=OLD_ROOT,
                    help="where the run folders (best.pt/summary.json) live, read-only")
    args = ap.parse_args()
    OLD_ROOT = args.root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt = np.load(os.path.join(proc_dir(VAL_SEGMENT), "ink_reg.npy")) > 0
    mask_arr = np.load(os.path.join(proc_dir(VAL_SEGMENT), "mask.npy")) > 0

    out_dir = os.path.join(NEW_ROOT, "zshift_stress")
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
    results = []
    for run in args.runs:
        ckpt_path = os.path.join(OLD_ROOT, run, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] {run}"); continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        base_cfg = ckpt["cfg"]
        frozen_tau = float(json.load(open(os.path.join(OLD_ROOT, run, "summary.json")))["best_tau"])
        model = build_model(base_cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        rows = []
        for d in args.deltas:
            cfg = copy.deepcopy(base_cfg)
            cfg["data"]["center_layer"] = int(cfg["data"].get("center_layer", 32)) + d
            prob, mask, _ = infer_segment(model, VAL_SEGMENT, cfg, device, proc_dir)
            m_frozen = pixel_metrics(prob, gt, mask_arr, frozen_tau)
            m_sweep, _ = sweep_threshold(prob, gt, mask_arr)
            rows.append({"delta": d, "frozen": m_frozen, "sweep": m_sweep})
            print(f"[{run}] d={d:+d}  F0.5 frozen@tau{frozen_tau:.2f}={m_frozen['fbeta']:.4f}  "
                  f"sweep={m_sweep['fbeta']:.4f}@tau{m_sweep['tau']:.2f}")
        results.append({"run": run, "group": GROUPS.get(run, run),
                        "frozen_tau": frozen_tau, "rows": rows})
        with open(os.path.join(out_dir, "zshift_stress.json"), "w") as f:
            json.dump(results, f, indent=2)   # save incrementally, job-safe

    # --- figure: frozen-tau F0.5 vs delta, grouped mean +- sd ------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, key, title in ((axes[0], "frozen", "frozen tau (deployment)"),
                           (axes[1], "sweep", "re-swept tau (upper bound)")):
        groups = {}
        for r in results:
            groups.setdefault(r["group"], []).append(
                [row[key]["fbeta"] for row in r["rows"]])
        for gname, curves in groups.items():
            arr = np.array(curves)
            mean = arr.mean(0)
            ax.plot(args.deltas, mean, marker="o", label=f"{gname} (n={len(curves)})")
            if len(curves) > 1:
                sd = arr.std(0, ddof=1)
                ax.fill_between(args.deltas, mean - sd, mean + sd, alpha=0.15)
        ax.set_xlabel("induced depth-window shift delta (layers)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val F0.5 (reg labels)")
    axes[0].legend(fontsize=8)
    fig.suptitle("G1 — depth-offset robustness stress test")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "zshift_stress.png"), dpi=130,
                bbox_inches="tight")
    print(f"-> {out_dir}/figs/zshift_stress.png")


if __name__ == "__main__":
    main()
