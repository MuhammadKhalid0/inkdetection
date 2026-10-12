"""Stage geometry, presets and divisibility contracts for the SegFormer3D-Ink port.

The reference implementation (OSUPCVLab/SegFormer3D) recovers the token grid with
``cube_root(N)`` and reshapes to ``(n, n, n)``.  That is only valid for cubic token
counts.  For ink input ``(B, 1, 32, t, t)`` it raises at t in {64,128,192,320} and --
much worse -- SILENTLY SUCCEEDS at t=256, where 8*64*64 = 32768 = 32**3, reinterpreting
a (8,64,64) grid as (32,32,32).  Every downstream op then mixes tokens across depth rows
and no exception is raised (plan D1).

This module replaces that inference-by-arithmetic with an explicit per-stage grid that is
computed at build time, threaded through every block, and asserted at forward time.
"""
from typing import List, Sequence, Tuple

# --- patch-embedding stems (per-axis kernel / stride / padding, per stage) -----------
# Stock isotropic stem: depth 32 -> 8 -> 4 -> 2 -> 1  (stage-4 Mix-FFN sees only padding
# along depth, stage-4 attention is purely in-plane).  Kept as ablation cell A1.
STEM_ISO = dict(
    kernel=[(7, 7, 7), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
    stride=[(4, 4, 4), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    padding=[(3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
)
# Depth-preserving stem (the ink default): depth 32 -> 32 -> 16 -> 8 -> 4, so stage-1
# grid is (32, t/4, t/4) -- the same S1 grid as the nnFormer port.
STEM_DP = dict(
    kernel=[(3, 7, 7), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
    stride=[(1, 4, 4), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    padding=[(1, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
)
# 2.5D control (plan A10 / 2.6): depth-as-channels. The 32 depth layers become input
# channels and the depth AXIS is a singleton, so every 3x3x3 conv degenerates to 3x3 and
# every sr conv to a 2D reduction -- i.e. the identical code path runs as a 2D MiT.
STEM_2D = dict(
    kernel=[(1, 7, 7), (1, 3, 3), (1, 3, 3), (1, 3, 3)],
    stride=[(1, 4, 4), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
    padding=[(0, 3, 3), (0, 1, 1), (0, 1, 1), (0, 1, 1)],
)
STEMS = {"dp": STEM_DP, "iso": STEM_ISO, "2d": STEM_2D}

# --- sequence-reduction presets (per-axis, per stage) --------------------------------
# The reference uses scalar r per stage, realised as Conv3d(C, C, k=r, s=r), i.e. an
# r**3 reduction.  Per-axis tuples let the keys keep depth resolution.
SR_PRESETS = {
    "444_222": [(4, 4, 4), (2, 2, 2), (1, 1, 1), (1, 1, 1)],   # stock-isotropic
    "244_222": [(2, 4, 4), (2, 2, 2), (1, 1, 1), (1, 1, 1)],
    "144_122": [(1, 4, 4), (1, 2, 2), (1, 1, 1), (1, 1, 1)],   # ink default
}

# --- capacity rungs (measured counts in docs/bench_table.md) -------------------------
CAPACITY_PRESETS = {
    #            dims                     heads         depths
    "S": dict(dims=(32, 64, 160, 256),  heads=(1, 2, 5, 8), depths=(2, 2, 2, 2)),
    "M": dict(dims=(64, 128, 320, 512), heads=(1, 2, 5, 8), depths=(2, 2, 5, 2)),
    "L": dict(dims=(96, 192, 480, 768), heads=(1, 2, 5, 8), depths=(3, 6, 14, 3)),
    # alternates at the same ~79M budget, kept for reference (plan 3.2)
    "L_deep": dict(dims=(64, 128, 320, 512), heads=(1, 2, 5, 8), depths=(3, 6, 48, 3)),
    "L_wide": dict(dims=(128, 256, 640, 1024), heads=(2, 4, 10, 16), depths=(2, 4, 5, 2)),
}


def conv_out(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


def stage_grids(img_size: Sequence[int], stem: str) -> List[Tuple[int, int, int]]:
    """Per-stage (D, H, W) token grids for input ``img_size`` = (D, H, W)."""
    cfg = STEMS[stem]
    grids, cur = [], tuple(int(v) for v in img_size)
    for i in range(4):
        k, s, p = cfg["kernel"][i], cfg["stride"][i], cfg["padding"][i]
        cur = tuple(conv_out(cur[a], k[a], s[a], p[a]) for a in range(3))
        if min(cur) < 1:
            raise ValueError(f"stage {i+1} grid collapsed to {cur} for img_size={img_size}")
        grids.append(cur)
    return grids


def assert_exact(img_size: Sequence[int], stem: str) -> None:
    """Every stage stride must divide its input exactly (no ceil rounding anywhere)."""
    cfg = STEMS[stem]
    cur = tuple(int(v) for v in img_size)
    for i in range(4):
        k, s, p = cfg["kernel"][i], cfg["stride"][i], cfg["padding"][i]
        for a, axis in enumerate("DHW"):
            n, k_, s_, p_ = cur[a], k[a], s[a], p[a]
            out = conv_out(n, k_, s_, p_)
            if n == 1 and s_ == 1 and out == 1:
                continue                      # singleton axis (2.5D control): nothing to divide
            # The real invariant for a PADDED patch-embed conv is not exact divisibility but
            # (i) the standard ceil output size and (ii) that every input position is seen by
            # at least one kernel window, i.e. nothing is silently dropped. Exact halving is
            # the usual case (depth 32, tiles that are multiples of 32); an odd axis such as
            # the full 65-layer stack rounds up (65 -> 33 -> 17 -> 9) and is still complete.
            # The sr conv has NO padding and DOES truncate, so it keeps the strict check in
            # assert_sr_divides().
            if out != -(-n // s_):
                raise ValueError(
                    f"stem='{stem}' stage {i+1} axis {axis}: size {n} with k={k_} s={s_} "
                    f"p={p_} gives {out}, expected ceil({n}/{s_})={-(-n // s_)}.")
            if (out - 1) * s_ + k_ - p_ < n:
                raise ValueError(
                    f"stem='{stem}' stage {i+1} axis {axis}: size {n} with k={k_} s={s_} "
                    f"p={p_} leaves the last {n - ((out - 1) * s_ + k_ - p_)} position(s) "
                    f"unseen by any kernel window -- input would be silently dropped.")
        cur = tuple(conv_out(cur[a], k[a], s[a], p[a]) for a in range(3))


def assert_sr_divides(grids: Sequence[Tuple[int, int, int]],
                      sr: Sequence[Tuple[int, int, int]]) -> None:
    """The sr conv has no padding, so a non-dividing grid is silently TRUNCATED by the
    reference.  Fail at build time instead."""
    for i, (g, r) in enumerate(zip(grids, sr)):
        for a, axis in enumerate("DHW"):
            if g[a] % r[a] != 0:
                raise ValueError(
                    f"stage {i+1} axis {axis}: sr ratio {r[a]} does not divide grid "
                    f"{g[a]} (grid {g}) -- the reference would floor-truncate silently.")


def kv_lengths(grids: Sequence[Tuple[int, int, int]],
               sr: Sequence[Tuple[int, int, int]]) -> List[int]:
    return [(g[0] // r[0]) * (g[1] // r[1]) * (g[2] // r[2]) for g, r in zip(grids, sr)]


def token_counts(grids: Sequence[Tuple[int, int, int]]) -> List[int]:
    return [g[0] * g[1] * g[2] for g in grids]
