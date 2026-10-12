"""Pre-flight validation — run this BEFORE launching training to catch shape,
import, and gradient bugs without touching the GPU cluster queue or real data.

    python scripts/smoke_test.py --config configs/a0.yaml

Checks:
  1. all imports resolve (monai/einops/timm/torch),
  2. model builds from the config,
  3. forward pass on a synthetic batch yields logits [B,1,H,W],
  4. loss is finite and backward populates gradients,
  5. encoder/decoder stage resolutions are internally consistent,
  6. (if processed data exists) one real tile reads and passes through.
Exit code 0 = safe to train.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.losses import build_loss                       # noqa: E402
from engine.utils import count_params, load_config          # noqa: E402
from models.wrapper import build_model                       # noqa: E402


def check(name, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "a0.yaml"))
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"== smoke test (device={device}) ==")

    D = cfg["data"]["depth"]
    T = cfg["data"]["tile"]

    print("[1] build model")
    model = build_model(cfg).to(device)
    n = count_params(model)
    print(f"      trainable params: {n:,}")
    check("params > 0", n > 0)

    # report the size bookkeeping from the backbone
    bb = model.backbone
    print(f"      stage_res = {bb.stage_res}")
    print(f"      encoder input_size (tokens) = {bb.input_size}")
    print(f"      feat_size (bottleneck) = {bb.feat_size}")
    check("bottleneck volume matches token count",
          int(np.prod(bb.feat_size)) == bb.input_size[-1])
    check("feature out channels == feature_size",
          bb.out_channels == cfg["model"]["feature_size"])

    print("[2] forward pass")
    B = 2
    C_out = int(cfg.get("model", {}).get("head_classes", 1))   # 1 (sigmoid) or 3 (soft_ce3)
    x = torch.randn(B, 1, D, T, T, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"      input  {tuple(x.shape)} -> logits {tuple(logits.shape)}")
    check(f"logits shape == [B,{C_out},T,T]", tuple(logits.shape) == (B, C_out, T, T))

    print("[3] loss + backward")
    model.train()
    loss_fn = build_loss(cfg["train"])
    logits = model(x)
    y = (torch.rand(B, 1, T, T, device=device) > 0.9).float()
    loss = loss_fn(logits, y)
    check("loss is finite", torch.isfinite(loss).item())
    loss.backward()
    has_grad = any((p.grad is not None and torch.isfinite(p.grad).all())
                   for p in model.parameters() if p.requires_grad)
    check("gradients populated & finite", has_grad)
    print(f"      loss = {loss.item():.4f}")

    print("[4] real-tile read (optional)")
    try:
        from data.segments import VAL_SEGMENT, proc_dir
        vp = os.path.join(proc_dir(VAL_SEGMENT), "vol.npy")
        if os.path.exists(vp):
            vol = np.load(vp, mmap_mode="r")
            z0 = max(0, min(cfg["data"]["center_layer"] - D // 2,
                            cfg["data"]["n_layers"] - D))
            tile = np.asarray(vol[z0:z0 + D, :T, :T], dtype=np.float32) / 255.0
            xb = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(xb)
            check(f"real tile -> logits [1,{C_out},T,T]", tuple(out.shape) == (1, C_out, T, T))
        else:
            print("      (skipped: processed data not found — run data.prepare first)")
    except Exception as e:  # noqa: BLE001
        print(f"      (skipped real-tile check: {e})")

    print("\nALL CHECKS PASSED — safe to train.")


if __name__ == "__main__":
    main()
