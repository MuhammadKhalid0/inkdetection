"""Efficiency bench for nnFormer-Ink: measured params / GFLOPs / ms-per-tile / peak mem.

Fills the efficiency columns of the headline table (nnformer_plan.md §6.7).
CPU: params + GFLOPs only. CUDA: adds ms/tile (median of --iters) + peak memory.

    python scripts/bench.py --caps S,M,F --tiles 192,256 --device cuda \
        --out docs/bench_table.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nnformer_ink import build_nnformer_ink  # noqa: E402

D = 32


def flops_g(model, x):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return float("nan")
    model.eval()
    with FlopCounterMode(display=False) as fcm, torch.no_grad():
        model(x)
    return fcm.get_total_flops() / 1e9


def bench_one(cap, tile, device, iters):
    m = build_nnformer_ink(img_size=(D, tile, tile), capacity=cap).to(device).eval()
    params_m = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(1, 1, D, tile, tile, device=device)
    row = {"cap": cap, "tile": tile, "params_M": round(params_m, 2),
           "gflops": round(flops_g(m, x), 1), "ms": None, "peak_mem_GB": None}
    if device == "cuda":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):  # warmup
                m(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                m(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
        row["ms"] = round(sorted(times)[len(times) // 2], 1)
        row["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        del m, x
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="S,M,F")
    ap.add_argument("--tiles", default="192,256")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None, help="markdown table output path")
    args = ap.parse_args()

    rows = []
    for cap in args.caps.split(","):
        for tile in (int(t) for t in args.tiles.split(",")):
            row = bench_one(cap.strip(), tile, args.device, args.iters)
            rows.append(row)
            print(json.dumps(row))

    lines = [f"# nnFormer-Ink bench ({args.device}, bf16 inference, batch 1)", "",
             "| cap | tile | params (M) | GFLOPs | ms/tile | peak mem (GB) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| nnF-{r['cap']} | {r['tile']} | {r['params_M']} | "
                     f"{r['gflops']} | {r['ms']} | {r['peak_mem_GB']} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
