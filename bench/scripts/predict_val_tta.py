"""Phase 1 (plan §4): TTA re-scoring of existing checkpoints — no training.

For each run: load best.pt from the OLD outputs root (read-only), rebuild the
model from its stored config, run val inference with 8-fold dihedral TTA, then
sweep tau and (tau, min_size), plus IoU. Compares against the run's stored
no-TTA numbers.

Writes (new root only): <NEW_ROOT>/tta/<run>/{val_prob_tta.npy, tta_report.json}
and an aggregate <NEW_ROOT>/tta/tta_summary.{json,md}.

Decision rule (plan): adopt TTA if mean Delta(post F0.5) over the lean seeds
> +0.003 (paired comparison — the checkpoint is held fixed).

Usage:  python -m scripts.predict_val_tta
        python -m scripts.predict_val_tta --runs lean_t192_s42 a2bs_t192
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import VAL_SEGMENT, proc_dir  # noqa: E402
from engine.eval import pixel_metrics, sweep_threshold  # noqa: E402
from engine.infer import infer_segment  # noqa: E402
from models.wrapper import build_model  # noqa: E402
from scripts.postprocess import component_filter, TAUS, MIN_SIZES  # noqa: E402

OLD_ROOT = "/home/woody/iwi5/iwi5401h/vesuvius/outputs"
NEW_ROOT = os.environ.get("INK_OUTPUT_ROOT",
                          "/home/woody/iwi5/iwi5401h/vesuvius/outputs2")
DEFAULT_RUNS = ["lean_t192_s42", "lean_t192_s43", "lean_t192_s44",
                "a2bs_t192", "a2bs_t192_s43", "a2bs_t192_s44"]


def post_sweep(prob, gt, mask):
    best = None
    for tau in TAUS:
        pred0 = prob >= tau
        for ms in MIN_SIZES:
            pred = component_filter(pred0, ms)
            m = pixel_metrics(pred.astype(np.float32), gt, mask, 0.5)
            if best is None or m["fbeta"] > best["fbeta"]:
                best = m; best["tau"], best["min_size"] = float(tau), int(ms)
    return best


def main():
    global OLD_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--root", default=OLD_ROOT,
                    help="where the run folders (best.pt/summary.json) live, read-only")
    args = ap.parse_args()
    OLD_ROOT = args.root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt = np.load(os.path.join(proc_dir(VAL_SEGMENT), "ink_reg.npy")) > 0
    mask_arr = np.load(os.path.join(proc_dir(VAL_SEGMENT), "mask.npy")) > 0

    results = []
    for run in args.runs:
        ckpt_path = os.path.join(OLD_ROOT, run, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] {run}: no best.pt"); continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = copy.deepcopy(ckpt["cfg"])
        cfg.setdefault("infer", {})["tta"] = True
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        prob, mask, _ = infer_segment(model, VAL_SEGMENT, cfg, device, proc_dir)
        out_dir = os.path.join(NEW_ROOT, "tta", run)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "val_prob_tta.npy"), prob.astype(np.float32))

        best_raw, _ = sweep_threshold(prob, gt.astype(np.float64), mask_arr.astype(np.float64))
        best_post = post_sweep(prob, gt, mask_arr)

        base = {}
        sp = os.path.join(OLD_ROOT, run, "summary.json")
        pp = os.path.join(OLD_ROOT, run, "postprocess.json")
        if os.path.exists(sp):
            base["raw_f05"] = json.load(open(sp)).get("best_f05")
        if os.path.exists(pp):
            base["post_f05"] = json.load(open(pp)).get("postprocessed", {}).get("f05")

        rep = {"run": run, "tta": {"raw": best_raw, "post": best_post},
               "baseline_no_tta": base,
               "delta_raw": (best_raw["fbeta"] - base["raw_f05"]) if base.get("raw_f05") else None,
               "delta_post": (best_post["fbeta"] - base["post_f05"]) if base.get("post_f05") else None}
        with open(os.path.join(out_dir, "tta_report.json"), "w") as f:
            json.dump(rep, f, indent=2)
        results.append(rep)
        print(f"[{run}] TTA raw {best_raw['fbeta']:.4f} (D {rep['delta_raw'] if rep['delta_raw'] is None else round(rep['delta_raw'], 4)}) "
              f"post {best_post['fbeta']:.4f} (D {rep['delta_post'] if rep['delta_post'] is None else round(rep['delta_post'], 4)})")

    os.makedirs(os.path.join(NEW_ROOT, "tta"), exist_ok=True)
    with open(os.path.join(NEW_ROOT, "tta", "tta_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = ["# TTA re-scoring (8-fold dihedral, existing checkpoints)", "",
             "| run | raw F0.5 TTA | Delta raw | post F0.5 TTA | Delta post | IoU_ink post |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['run']} | {r['tta']['raw']['fbeta']:.4f} | "
                     f"{r['delta_raw'] if r['delta_raw'] is None else format(r['delta_raw'], '+.4f')} | "
                     f"{r['tta']['post']['fbeta']:.4f} | "
                     f"{r['delta_post'] if r['delta_post'] is None else format(r['delta_post'], '+.4f')} | "
                     f"{r['tta']['post']['iou_ink']:.4f} |")
    deltas = [r["delta_post"] for r in results if r["delta_post"] is not None and r["run"].startswith("lean")]
    if deltas:
        lines += ["", f"Mean Delta(post) over lean seeds: {np.mean(deltas):+.4f} "
                  f"-> {'ADOPT' if np.mean(deltas) > 0.003 else 'DO NOT ADOPT'} (gate +0.003)"]
    with open(os.path.join(NEW_ROOT, "tta", "tta_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]))


if __name__ == "__main__":
    main()
