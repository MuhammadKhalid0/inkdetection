"""Measured efficiency table: params / GFLOPs / ms-per-tile / peak memory.

Fills the efficiency columns of Table S-VII and the frontier figure S-VIII.
Usage:  python scripts/bench.py [--device cuda] [--out docs/bench_table.md]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from segformer3d_ink import build_segformer3d_ink as B


def measure(cap, tile, stem, sr, device, dtype, reps=5):
    m = B(img_size=(32, tile, tile), capacity=cap, stem=stem, sr=sr).to(device).eval()
    params = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 1, 32, tile, tile, device=device)
    gf = None
    try:
        from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count
        # CPU CAVEAT (measured): F.scaled_dot_product_attention dispatches to
        # aten._scaled_dot_product_flash_attention_for_cpu, which is NOT in
        # FlopCounterMode's registry -- a naive CPU count SILENTLY OMITS every attention
        # matmul (t192: 102.8 vs the true 202.4 GFLOPs). The CUDA flash/efficient kernels
        # are registered, so this mapping only matters on CPU, but it is always applied
        # so CPU and GPU numbers are directly comparable.
        cm = {}
        op = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)
        if op is not None:
            def _sdpa(q_shape, k_shape, v_shape, *args, out_shape=None, **kw):
                return sdpa_flop_count(q_shape, k_shape, v_shape)
            cm[op] = _sdpa
        fc = FlopCounterMode(display=False, custom_mapping=cm)
        with fc, torch.no_grad():
            m(x)
        gf = fc.get_total_flops() / 1e9
    except Exception as e:                                    # pragma: no cover
        print(f"  (flop counter unavailable: {e})")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1e3
    mem = (torch.cuda.max_memory_allocated() / 2**30) if device == "cuda" else float("nan")
    del m, x
    if device == "cuda":
        torch.cuda.empty_cache()
    return params, gf, ms, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "docs", "bench_table.md"))
    ap.add_argument("--tiles", nargs="*", type=int, default=[192, 256, 320])
    a = ap.parse_args()
    dtype = torch.bfloat16
    rows = []
    grid = ([("S", "dp", "144_122"), ("S", "dp", "444_222"), ("S", "iso", "444_222"),
             ("M", "dp", "144_122"), ("L", "dp", "144_122")])
    for cap, stem, sr in grid:
        for t in a.tiles:
            if cap == "L" and t == 320:
                continue                      # anti-goal R9: SF-L never at t320
            try:
                p, gf, ms, mem = measure(cap, t, stem, sr, a.device, dtype)
                rows.append(dict(cap=cap, stem=stem, sr=sr, tile=t, params=p,
                                 gflops=gf, ms=ms, mem=mem))
                print(f"  SF-{cap} {stem} {sr} t{t}: {p:,} params, "
                      f"{gf:.1f} GFLOPs, {ms:.1f} ms, {mem:.2f} GB")
            except RuntimeError as e:
                print(f"  SF-{cap} {stem} {sr} t{t}: FAILED ({str(e)[:80]})")
    L = ["# SegFormer3D-Ink bench (%s, bf16 inference, batch 1)" % a.device, "",
         "| cap | stem | sr | tile | params | GFLOPs | ms/tile | peak mem (GB) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| SF-{r['cap']} | {r['stem']} | {r['sr']} | {r['tile']} | "
                 f"{r['params']:,} | {r['gflops']:.1f} | {r['ms']:.1f} | {r['mem']:.2f} |")
    L += ["", "Reference (chapter 2, same method): nnF-S 19.87M 171.8/305.6 GFLOPs "
          "25.8/36.5 ms at t192/t256; nnF-M 79.11M 900.9 at t256; nnF-F 177.74M 1860.3.",
          "UNETR++ deployed t256 = 79.1M."]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(rows, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print("\n->", a.out)


if __name__ == "__main__":
    main()
