"""Backfill IoU (ink / background / mean) for every historical run — plan §3.

Supervisor request C5: report IoU for foreground AND background. This script
retro-scores every saved val probability map (read-only on the old outputs
folder) at three operating points:
  * the reported swept tau (raw),
  * the reported (tau, min_size) post-filter point (if postprocess.json exists),
  * the IoU_ink-optimal tau from a fresh sweep (IoU prefers lower tau than F0.5).

Acceptance check: the recomputed F0.5 must match summary/postprocess.json to
4 decimals — mismatches are flagged, never silently overwritten.

Writes (new root only, old folders untouched):
    <NEW_ROOT>/iou_backfill/<run>_iou.json
    <NEW_ROOT>/iou_backfill/iou_backfill.{json,md}

Usage:  python -m scripts.rescore_iou            # default run list
        python -m scripts.rescore_iou --runs a2bs_t192 r4
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import VAL_SEGMENT, proc_dir  # noqa: E402
from engine.eval import pixel_metrics  # noqa: E402
from scripts.postprocess import component_filter  # noqa: E402

OLD_ROOT = "/home/woody/iwi5/iwi5401h/vesuvius/outputs"
NEW_ROOT = os.environ.get("INK_OUTPUT_ROOT",
                          "/home/woody/iwi5/iwi5401h/vesuvius/outputs2")

DEFAULT_RUNS = [
    "a0", "a0s", "a0b", "a1", "a1_reg", "a2", "a2b", "a2bs", "a2bs_reg",
    "a1_t128", "a2bs_t128", "a2bs_t192", "a2bs_t192_s43", "a2bs_t192_s44",
    "lean_t192_s42", "lean_t192_s43", "lean_t192_s44",
    "abl_stem", "abl_da", "abl_surfrel", "abl_perpix", "abl_warpstab",
    "abl_labels", "abl_loss_recall", "abl_loss_prec",
    "r2", "r2_self", "r3", "r4", "r4_s43", "r4_s44",
]


def score_run(run, gt, mask, root=OLD_ROOT):
    rd = os.path.join(root, run)
    prob_path = os.path.join(rd, "val_prob_best.npy")
    if not os.path.exists(prob_path):
        return None
    prob = np.load(prob_path)
    out = {"run": run}

    summ = {}
    sp = os.path.join(rd, "summary.json")
    if os.path.exists(sp):
        summ = json.load(open(sp))
    tau = float(summ.get("best_tau", 0.5))
    m_raw = pixel_metrics(prob, gt, mask, tau)
    out["raw"] = m_raw
    ref = summ.get("best_f05")
    out["raw_f05_matches_summary"] = (ref is None or abs(m_raw["fbeta"] - ref) < 5e-4)
    if not out["raw_f05_matches_summary"]:
        out["summary_f05"] = ref
        print(f"  [warn] {run}: recomputed F0.5 {m_raw['fbeta']:.4f} != summary {ref:.4f} "
              f"(different label file at selection time? reported, not overwritten)")

    pp = os.path.join(rd, "postprocess.json")
    if os.path.exists(pp):
        ppd = json.load(open(pp)).get("postprocessed", {})
        if ppd:
            pt, ms = float(ppd["tau"]), int(ppd["min_size"])
            pred = component_filter(prob >= pt, ms)
            # pixel_metrics thresholds prob itself; feed the filtered mask as prob
            m_post = pixel_metrics(pred.astype(np.float32), gt, mask, 0.5)
            m_post["tau"], m_post["min_size"] = pt, ms
            out["post"] = m_post

    best_iou, best_tau_iou = None, None
    for t in np.round(np.arange(0.20, 0.91, 0.05), 2):
        m = pixel_metrics(prob, gt, mask, float(t))
        if best_iou is None or m["iou_ink"] > best_iou["iou_ink"]:
            best_iou, best_tau_iou = m, float(t)
    out["iou_opt"] = best_iou
    out["iou_opt"]["tau"] = best_tau_iou
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--root", default=OLD_ROOT, help="where the run folders live (read-only)")
    args = ap.parse_args()
    runs = args.runs or DEFAULT_RUNS

    gt = np.load(os.path.join(proc_dir(VAL_SEGMENT), "ink_reg.npy")) > 0
    mask = np.load(os.path.join(proc_dir(VAL_SEGMENT), "mask.npy")) > 0

    out_dir = os.path.join(NEW_ROOT, "iou_backfill")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for run in runs:
        r = score_run(run, gt, mask, root=args.root)
        if r is None:
            print(f"  [skip] {run}: no val_prob_best.npy")
            continue
        results.append(r)
        with open(os.path.join(out_dir, f"{run}_iou.json"), "w") as f:
            json.dump(r, f, indent=2)
        raw = r["raw"]
        print(f"[{run}] raw@tau{raw['tau']:.2f} F0.5={raw['fbeta']:.4f} "
              f"IoU_ink={raw['iou_ink']:.4f} IoU_bg={raw['iou_bg']:.4f}")

    with open(os.path.join(out_dir, "iou_backfill.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# IoU backfill (masked, vs registered labels; tau/ms as reported; "
             "IoU reported — never selected on)", "",
             "| run | F0.5 raw | IoU_ink raw | IoU_bg raw | F0.5 post | IoU_ink post "
             "| IoU_bg post | IoU_ink @IoU-opt tau | ok |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        raw, post, opt = r["raw"], r.get("post"), r["iou_opt"]
        lines.append(
            f"| {r['run']} | {raw['fbeta']:.4f} | {raw['iou_ink']:.4f} | {raw['iou_bg']:.4f} | "
            + (f"{post['fbeta']:.4f} | {post['iou_ink']:.4f} | {post['iou_bg']:.4f} | " if post
               else "– | – | – | ")
            + f"{opt['iou_ink']:.4f} @{opt['tau']:.2f} | "
            + ("y" if r["raw_f05_matches_summary"] else "MISMATCH") + " |")
    with open(os.path.join(out_dir, "iou_backfill.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {out_dir}/iou_backfill.md ({len(results)} runs)")


if __name__ == "__main__":
    main()
