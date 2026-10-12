"""Convolutional embedding and down/up-sampling layers (faithful ports).

PatchEmbed3D generalizes the original's two `project` blocks to per-axis
stride factorization so patch_size (1,4,4) (depth-preserving, the ink default,
plan §1) and (2,4,4) (the n_embedz2 ablation) share one code path.
"""
from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _two_factor(ps: int) -> Tuple[int, int]:
    """Factor a per-axis patch stride into the two conv strides of the stem."""
    table = {1: (1, 1), 2: (2, 1), 4: (2, 2)}
    if ps not in table:
        raise ValueError(f"patch stride {ps} unsupported (use 1, 2 or 4)")
    return table[ps]


class _Project(nn.Module):
    """conv3(k3,s) -> GELU -> LN -> conv3(k3,1) [-> GELU -> LN unless last]."""

    def __init__(self, in_dim, out_dim, stride, last=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_dim, out_dim, 3, stride=1, padding=1)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = None if last else nn.LayerNorm(out_dim)

    @staticmethod
    def _ln(x, norm):
        # LayerNorm over channels of a [B,C,D,H,W] volume via token view
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = norm(x)
        return x.transpose(1, 2).contiguous().view(B, C, D, H, W)

    def forward(self, x):
        x = self._ln(self.act(self.conv1(x)), self.norm1)
        x = self.conv2(x)
        if self.norm2 is not None:
            x = self._ln(self.act(x), self.norm2)
        return x


class PatchEmbed3D(nn.Module):
    """4-conv embedding, strides factored per axis from patch_size."""

    def __init__(self, patch_size: Sequence[int], in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = tuple(patch_size)
        s1, s2 = zip(*(_two_factor(p) for p in self.patch_size))
        self.proj1 = _Project(in_chans, embed_dim // 2, tuple(s1), last=False)
        self.proj2 = _Project(embed_dim // 2, embed_dim, tuple(s2), last=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # [B, in, D, H, W] -> [B, C, D/p0, H/p1, W/p2]
        for ax, p in enumerate(self.patch_size):
            assert x.shape[2 + ax] % p == 0, \
                f"input axis {ax} size {x.shape[2 + ax]} not divisible by patch {p}"
        x = self.proj2(self.proj1(x))
        B, C, D, H, W = x.shape
        t = x.flatten(2).transpose(1, 2).contiguous()
        t = self.norm(t)
        return t.transpose(1, 2).contiguous().view(B, C, D, H, W)


class PatchMerging3D(nn.Module):
    """GELU -> LN -> conv3(k3, s2) doubling channels (faithful)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Conv3d(dim, dim * 2, 3, stride=2, padding=1)

    def forward(self, x, grid):  # x: [B, L, C]
        B, L, C = x.shape
        D, H, W = grid
        assert L == D * H * W
        x = torch.nn.functional.gelu(x)
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        x = self.reduction(x)
        return x.flatten(2).transpose(1, 2).contiguous()  # [B, L/8, 2C]


class PatchExpanding3D(nn.Module):
    """LN -> transposed conv (k2, s2) halving channels (faithful)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.ConvTranspose3d(dim, dim // 2, 2, 2)

    def forward(self, x, grid):  # x: [B, L, C] at coarse grid
        B, L, C = x.shape
        D, H, W = grid
        assert L == D * H * W
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        x = self.up(x)
        return x.flatten(2).transpose(1, 2).contiguous()  # [B, 8L, C/2]


class FinalExpand3D(nn.Module):
    """Stage-1 grid -> full resolution feature volume (out_ch channels).

    Replaces the original per-class final_patch_expanding: DS is off and the
    2D head lives outside the backbone (bench wrapper contract)."""

    def __init__(self, dim, out_ch, patch_size: Sequence[int]):
        super().__init__()
        ps = tuple(patch_size)
        self.up = nn.ConvTranspose3d(dim, out_ch, ps, ps)

    def forward(self, x):  # [B, C, D/p0, t/p1, t/p2] -> [B, out_ch, D, t, t]
        return self.up(x)
