"""Resolution-transfer study (SegFormer3D-only, inference-only) — plan Phase S4.7.

Tests the source paper's own claim that a positional-encoding-free design "prevents
accuracy loss when there is a resolution mismatch during training and inference", in 3D,
on ink, across a 16x change in token count.  Only possible because this model has ZERO
tile-dependent parameters (no relative-position tables, no PE, no token-axis projections)
-- neither UNETR++ nor nnFormer can be scored this way.

BatchNorm running statistics are the model's only tile-dependent state, so a paired run
with `--no-bn-runstats` (BN in train-mode statistics) localises any degradation.

Usage: python -m scripts.stress_tile --runs sf_base_t192_s42 --tiles 128 192 256 320 384
"""
import argparse, copy, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from data.segments import VAL_SEGMENT, proc_dir
from engine.eval import pixel_metrics, sweep_threshold
from engine.infer import infer_segment
from models.wrapper import build_model

ROOT = os.environ.get("INK_OUTPUT_ROOT", "/home/woody/iwi5/iwi5401h/segFormer/outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tiles", nargs="*", type=int, default=[128, 192, 256, 320, 384])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-bn-runstats", action="store_true",
                    help="put BatchNorm in train-mode statistics (isolates the only "
                         "tile-dependent state in the model)")
    a = ap.parse_args()
    out_dir = a.out or os.path.join(a.root, "tile_transfer")
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for run in a.runs:
        d = os.path.join(a.root, run)
        ck = torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=False)
        summ = json.load(open(os.path.join(d, "summary.json")))
        tau_frozen = float(summ["best_tau"])
        cfg0 = ck["cfg"]
        train_tile = int(cfg0["data"]["tile"])
        rows = []
        for t in a.tiles:
            cfg = copy.deepcopy(cfg0)
            cfg["data"]["tile"] = t
            cfg["infer"] = dict(cfg.get("infer", {}))
            cfg["infer"]["stride"] = t // 2
            try:
                model = build_model(cfg).to(dev)
                model.load_state_dict(ck["model"])
            except Exception as e:
                rows.append({"tile": t, "error": str(e)[:200]})
                print(f"  {run} @t{t}: BUILD/LOAD FAILED — {str(e)[:120]}")
                continue
            model.eval()
            if a.no_bn_runstats:
                for m in model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.train()
            prob, mask, gt = infer_segment(model, VAL_SEGMENT, cfg, dev, proc_dir)
            frozen = pixel_metrics(prob, gt, mask, tau_frozen)
            swept = sweep_threshold(prob, gt, mask)
            rows.append({"tile": t, "frozen_tau": tau_frozen,
                         "frozen": {k: float(v) for k, v in frozen.items()},
                         "swept": {k: (float(v) if isinstance(v, (int, float)) else v)
                                   for k, v in swept.items()}})
            print(f"  {run} train@t{train_tile} -> infer@t{t}: "
                  f"frozen F0.5 {frozen['f05']:.4f} · swept {swept['f05']:.4f}")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
        results.append({"run": run, "train_tile": train_tile,
                        "bn_runstats": not a.no_bn_runstats, "rows": rows})
        tag = "_nobn" if a.no_bn_runstats else ""
        json.dump(results, open(os.path.join(out_dir, f"tile_transfer{tag}.json"), "w"),
                  indent=1)

    # pre-registered reading
    print("\n== pre-registered verdict (plan S4.7) ==")
    for r in results:
        base = next((x for x in r["rows"] if x.get("tile") == r["train_tile"]
                     and "frozen" in x), None)
        if not base:
            continue
        b = base["frozen"]["f05"]
        for x in r["rows"]:
            if "frozen" not in x or x["tile"] == r["train_tile"]:
                continue
            d = x["frozen"]["f05"] - b
            verdict = ("HOLDS (|d| <= 0.008)" if abs(d) <= 0.008 else
                       "FREE CONTEXT (>= +0.008)" if d >= 0.008 else
                       "REFUTED at this rung" if d < -0.03 else "degrades")
            print(f"  {r['run']}: t{r['train_tile']} -> t{x['tile']}: {d:+.4f}  {verdict}")


if __name__ == "__main__":
    main()
