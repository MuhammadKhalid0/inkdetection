"""Transformer blocks for nnFormer-Ink (LV/GV self-attention + skip attention).

Deviations from the original (all documented in docs/PORT_LOG.md):
  - anisotropic tuple windows + per-axis shift sizes;
  - static grids per config => no padding paths, shift masks precomputed as
    buffers at init instead of per forward;
  - identical residual/MLP structure otherwise.
"""
import torch
import torch.nn as nn

from timm.models.layers import DropPath

from .attention import WindowAttention3D
from .contrib import CrossScaleKV, SurfaceDepthBias
from .windows import (assert_divisible, clamp_window, shift_sizes,
                      shifted_window_mask, window_partition, window_reverse)


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwinBlock3D(nn.Module):
    """Standard (shifted-)window self-attention block on a fixed grid."""

    def __init__(self, dim, grid, num_heads, window, shifted,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0, sr_pb=False):
        super().__init__()
        self.grid = tuple(grid)
        self.window = clamp_window(window, grid)
        assert_divisible(self.grid, self.window)
        self.shift = shift_sizes(self.window, self.grid, shifted)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, self.window, num_heads,
                                      attn_drop=attn_drop, proj_drop=drop, sr_pb=sr_pb)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        # C1 / SR-PB (plan §2.1): zero-init -> exact stock reduction
        self.sr_pb = SurfaceDepthBias(dim, self.grid[0], num_heads) if sr_pb else None
        if any(self.shift):
            self.register_buffer("attn_mask",
                                 shifted_window_mask(self.grid, self.window, self.shift),
                                 persistent=False)
        else:
            self.attn_mask = None

    def forward(self, x):  # x: [B, L, C]
        B, L, C = x.shape
        D, H, W = self.grid
        assert L == D * H * W, f"token count {L} != grid {self.grid}"
        shortcut = x
        x = self.norm1(x).view(B, D, H, W, C)
        kb = None
        if self.sr_pb is not None:
            # [B,D,heads] -> per-token key bias, partitioned exactly like x
            b = self.sr_pb(x)[:, :, None, None, :].expand(B, D, H, W, -1)
            if any(self.shift):
                b = torch.roll(b, shifts=tuple(-s for s in self.shift), dims=(1, 2, 3))
            kb = window_partition(b, self.window)
        if any(self.shift):
            x = torch.roll(x, shifts=tuple(-s for s in self.shift), dims=(1, 2, 3))
        win = window_partition(x, self.window)
        win = self.attn(win, mask=self.attn_mask, key_bias=kb)
        x = window_reverse(win, self.window, B, D, H, W)
        if any(self.shift):
            x = torch.roll(x, shifts=tuple(self.shift), dims=(1, 2, 3))
        x = x.view(B, L, C)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SwinBlockKV3D(nn.Module):
    """Skip-attention block: Q = upsampled decoder tokens, K/V = encoder skip.

    Faithful to the original SwinTransformerBlock_kv: shift=0, one shared norm1
    applied to both skip and x_up, residual anchored on x (= x_up + skip).
    """

    def __init__(self, dim, grid, num_heads, window,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0, sr_pb=False,
                 cssa_dims=None):
        super().__init__()
        self.grid = tuple(grid)
        self.window = clamp_window(window, grid)
        assert_divisible(self.grid, self.window)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, self.window, num_heads, kv=True,
                                      attn_drop=attn_drop, proj_drop=drop, sr_pb=sr_pb)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.sr_pb = SurfaceDepthBias(dim, self.grid[0], num_heads) if sr_pb else None
        # C2 / CSSA (plan §2.2): zero-init gamma -> exact stock reduction
        self.cssa = CrossScaleKV(dim, cssa_dims, num_heads) if cssa_dims else None

    def forward(self, x, skip, x_up, coarse_tokens=None):  # all [B, L, C]
        B, L, C = x.shape
        D, H, W = self.grid
        assert L == D * H * W, f"token count {L} != grid {self.grid}"
        shortcut = x
        skip = self.norm1(skip).view(B, D, H, W, C)
        x_up = self.norm1(x_up).view(B, D, H, W, C)
        kb = None
        if self.sr_pb is not None:
            b = self.sr_pb(skip)[:, :, None, None, :].expand(B, D, H, W, -1)
            kb = window_partition(b, self.window)
        coarse = None
        if self.cssa is not None and coarse_tokens:
            coarse = self.cssa(coarse_tokens)
        skip_w = window_partition(skip, self.window)
        xup_w = window_partition(x_up, self.window)
        win = self.attn(skip_w, x_up=xup_w, key_bias=kb, coarse=coarse)
        x = window_reverse(win, self.window, B, D, H, W).view(B, L, C)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))
