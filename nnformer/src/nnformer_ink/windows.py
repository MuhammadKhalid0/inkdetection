"""Anisotropic 3D window utilities (tuple window sizes).

Axis convention everywhere in this package: volumes are [B, C, D, H, W] and
token grids are (D, H, W) with D = depth (the surface-normal axis). Inside
blocks, tokens are [B, L, C] with L = D*H*W, row-major over (D, H, W).
"""
from typing import Sequence, Tuple

import torch


def clamp_window(window: Sequence[int], grid: Sequence[int]) -> Tuple[int, int, int]:
    """Per-axis clamp of the window to the grid (original clamps by min(grid))."""
    return tuple(min(int(w), int(g)) for w, g in zip(window, grid))


def assert_divisible(grid: Sequence[int], window: Sequence[int]) -> None:
    for ax, (g, w) in enumerate(zip(grid, window)):
        if g % w != 0:
            raise ValueError(
                f"grid {tuple(grid)} not divisible by window {tuple(window)} on axis {ax} "
                f"(no padding paths by design — see nnformer_plan.md §3.1)"
            )


def window_partition(x: torch.Tensor, window: Sequence[int]) -> torch.Tensor:
    """[B, D, H, W, C] -> [B*nW, wd*wh*ww, C]."""
    B, D, H, W, C = x.shape
    wd, wh, ww = window
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, C)


def window_reverse(win: torch.Tensor, window: Sequence[int], B: int, D: int, H: int, W: int) -> torch.Tensor:
    """[B*nW, wd*wh*ww, C] -> [B, D, H, W, C]."""
    wd, wh, ww = window
    C = win.shape[-1]
    x = win.view(B, D // wd, H // wh, W // ww, wd, wh, ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, C)


def relative_position_index(window: Sequence[int]) -> torch.Tensor:
    """Pairwise relative-position index [N, N] into a table of size
    (2wd-1)*(2wh-1)*(2ww-1).

    Uses the standard Swin-3D strides. NOTE: the official nnFormer repo uses
    `*= 3*ws-1` / `*= 2*ws-1` (cubic windows only), which collides distinct
    offsets onto the same table entry; deliberately not reproduced here
    (PORT_LOG.md, deviation D2).
    """
    wd, wh, ww = window
    coords = torch.stack(
        torch.meshgrid(torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij")
    )  # [3, wd, wh, ww]
    flat = torch.flatten(coords, 1)  # [3, N]
    rel = flat[:, :, None] - flat[:, None, :]  # [3, N, N]
    rel = rel.permute(1, 2, 0).contiguous()  # [N, N, 3]
    rel[:, :, 0] += wd - 1
    rel[:, :, 1] += wh - 1
    rel[:, :, 2] += ww - 1
    rel[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
    rel[:, :, 1] *= (2 * ww - 1)
    index = rel.sum(-1)  # [N, N]
    assert int(index.max()) < (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
    assert int(index.min()) >= 0
    return index


def shift_sizes(window: Sequence[int], grid: Sequence[int], shifted: bool) -> Tuple[int, int, int]:
    """Per-axis cyclic-shift size: w//2 where shifting is possible and requested."""
    if not shifted:
        return (0, 0, 0)
    return tuple((w // 2 if g > w else 0) for w, g in zip(window, grid))


def shifted_window_mask(grid: Sequence[int], window: Sequence[int], shift: Sequence[int]) -> torch.Tensor:
    """Attention mask [nW, N, N] for shifted-window attention (0 / -100).

    Grid sizes are static per config, so this is computed once at module init
    and registered as a buffer (the original rebuilds it every forward).
    """
    D, H, W = grid
    img = torch.zeros(1, D, H, W, 1)

    def _slices(size: int, w: int, s: int):
        if s == 0:
            return [slice(0, size)]
        return [slice(0, size - w), slice(size - w, size - s), slice(size - s, size)]

    cnt = 0
    for ds in _slices(D, window[0], shift[0]):
        for hs in _slices(H, window[1], shift[1]):
            for ws in _slices(W, window[2], shift[2]):
                img[:, ds, hs, ws, :] = cnt
                cnt += 1
    mw = window_partition(img, window).squeeze(-1)  # [nW, N]
    mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))
    return mask
