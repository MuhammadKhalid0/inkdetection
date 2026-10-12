"""Apply a trained refiner checkpoint to an arbitrary probability map.

Used for the test segment (qualitative) and any re-refinement:
    python -m scripts.refine_predict --refiner r2 \
        --prob outputs/a2bs_t192/test_prob_<seg>.npy \
        --out  outputs/r2/test_prob_refined.npy
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import OUTPUT_ROOT                                 # noqa: E402
from engine.refiner import RefinerUNet, refine_map                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", required=True, help="refiner run name (outputs/<run>/best.pt)")
    ap.add_argument("--prob", required=True, help="input probability map .npy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(OUTPUT_ROOT, args.refiner, "best.pt"),
                    map_location=device, weights_only=False)
    model = RefinerUNet().to(device)
    model.load_state_dict(ck["model"])
    prob = np.load(args.prob).astype(np.float32)
    ref = refine_map(model, prob, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, ref.astype(np.float32))
    print(f"[refine_predict] {args.prob} -> {args.out} "
          f"(refiner {args.refiner} e{ck.get('epoch')}, f05 {ck.get('f05'):.4f})")


if __name__ == "__main__":
    main()
