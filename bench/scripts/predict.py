"""Sliding-window inference on a prepared segment with a trained checkpoint.

Loads a run's best.pt, rebuilds the model from the checkpoint's stored config
(so architecture flags always match the weights), runs Gaussian-blended
sliding-window inference, and saves the full-resolution probability map.

Defaults target the TEST segment (flatboi, no labels -> qualitative).  It can
also re-predict a labelled segment (pass --labelled --seg <id>).

Usage:
    python -m scripts.predict --run a2bs                     # test segment
    python -m scripts.predict --run a2bs --tau 0.6           # + save thresholded png-ready mask
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import (TEST_SEGMENT, proc_infer_dir, proc_dir,   # noqa: E402
                           OUTPUT_ROOT)
from engine.infer import infer_segment                                # noqa: E402
from models.wrapper import build_model                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output run name whose best.pt to load")
    ap.add_argument("--seg", default=TEST_SEGMENT)
    ap.add_argument("--labelled", action="store_true",
                    help="segment is a labelled one under proc/ (else proc_infer/, no labels)")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path (default outputs/<run>/best.pt)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.ckpt or os.path.join(OUTPUT_ROOT, args.run, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    print(f"[predict] run={args.run} ckpt={ckpt_path} epoch={ckpt.get('epoch')} "
          f"best_metric={ckpt.get('best_metric')}")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    proc_dir_fn = proc_dir if args.labelled else proc_infer_dir
    load_labels = args.labelled
    prob, mask, gt = infer_segment(model, args.seg, cfg, device, proc_dir_fn,
                                   load_labels=load_labels)
    print(f"[predict] prob {prob.shape} range [{prob.min():.3f},{prob.max():.3f}] "
          f"mean-in-mask {prob[mask>0].mean():.3f}")

    out = args.out or os.path.join(OUTPUT_ROOT, args.run,
                                   f"{'val' if args.labelled else 'test'}_prob_{args.seg[:16]}.npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, prob.astype(np.float32))
    with open(out.replace(".npy", "_meta.json"), "w") as f:
        json.dump({"run": args.run, "seg": args.seg, "ckpt": ckpt_path,
                   "epoch": ckpt.get("epoch"), "shape": list(prob.shape)}, f, indent=2)
    print(f"[predict] saved -> {out}")


if __name__ == "__main__":
    main()
