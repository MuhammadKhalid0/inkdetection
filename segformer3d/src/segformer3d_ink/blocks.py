"""Encoder blocks: overlapping patch embedding, sequence-reduced attention, Mix-FFN.

Faithful to OSUPCVLab/SegFormer3D ``architectures/segformer3d.py`` (main @ e314242f14)
with the deviations recorded in docs/PORT_LOG.md.  The load-bearing one: every
``cube_root(N)`` reshape is replaced by the explicit (D,H,W) grid threaded in.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrib import ContextStem, DepthCoordinate, PeripheralGate, widen_sr_conv


class PatchEmbed3D(nn.Module):
    """Overlapping convolutional patch embedding (per-axis kernel/stride/padding)."""

    def __init__(self, in_ch: int, dim: int, kernel, stride, padding):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=tuple(kernel),
                              stride=tuple(stride), padding=tuple(padding))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)                                  # [B,C,D,H,W]
        grid = tuple(x.shape[2:])
        x = x.flatten(2).transpose(1, 2)                  # [B,N,C]
        return self.norm(x), grid


class SelfAttention3D(nn.Module):
    """Sequence-reduced ("efficient") self-attention.

    Q is computed on the full stage grid; K/V come from a strided conv on the token grid
    -- the key set is already a low-resolution learned summary whose resolution is
    DECOUPLED from the query resolution.  That is the property C1/PDK extends.
    """

    def __init__(self, dim: int, num_heads: int, sr: Tuple[int, int, int],
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 scale_mode: str = "head_dim", sr_overlap: bool = False,
                 pdk: bool = False, pdk_grid: Tuple[int, int, int] = (8, 8, 8),
                 use_sdpa: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "embed dim must be divisible by num_heads"
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5) if scale_mode == "head_dim" \
            else (num_heads ** -0.5)
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_value = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.sr = tuple(sr)
        self.reduces = any(r > 1 for r in self.sr)
        if self.reduces:
            # NOTE: always build the STOCK conv here.  C3/OSR widening happens AFTER
            # _init_weights in the backbone -- otherwise `self.apply(self._init_weights)`
            # re-initialises the widened kernel with the WIDE fan-out and overwrites the
            # zero taps, silently destroying the exact-reduction property (caught by
            # tests/test_smoke.py [3] on the first run of this port).
            self.sr_conv = nn.Conv3d(dim, dim, kernel_size=self.sr, stride=self.sr)
            self.sr_norm = nn.LayerNorm(dim)
        self.sr_overlap = sr_overlap
        # --- C1 PDK -------------------------------------------------------------
        self.pdk = pdk
        if pdk:
            self.ctx_stem = ContextStem(dim, pdk_grid)
            self.ctx_gate = PeripheralGate(num_heads)

    def _heads(self, t, B, n):
        return t.reshape(B, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _attend(self, q, k, v):
        if self.use_sdpa:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                is_causal=False, scale=self.scale)
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = a.softmax(dim=-1)
        if self.training and self.attn_drop_p > 0:
            a = F.dropout(a, p=self.attn_drop_p, training=True)
        return a @ v

    def forward(self, x, grid, halo: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        D, H, W = grid
        assert D * H * W == N, f"grid {grid} does not match N={N}"
        q = self._heads(self.query(x), B, N)
        if self.reduces:
            x_ = x.transpose(1, 2).reshape(B, C, D, H, W)      # explicit grid, no cube_root
            x_ = self.sr_conv(x_).flatten(2).transpose(1, 2)
            x_ = self.sr_norm(x_)
        else:
            x_ = x
        kv = self.key_value(x_).reshape(B, -1, 2, self.num_heads,
                                        self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0].contiguous(), kv[1].contiguous()
        out = self._attend(q, k, v)
        # --- C1: peripheral branch, OWN softmax, zero-gated ----------------------
        if self.pdk and halo is not None:
            ctx = self.ctx_stem(halo)                          # [B,M_p,C]
            kvp = self.key_value(ctx).reshape(B, -1, 2, self.num_heads,
                                              self.head_dim).permute(2, 0, 3, 1, 4)
            outp = self._attend(q, kvp[0].contiguous(), kvp[1].contiguous())
            out = out + self.ctx_gate(outp)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class DWConv3D(nn.Module):
    """Mix-FFN depthwise conv -- the architecture's ENTIRE positional apparatus.

    Position is leaked only by this conv's zero padding (Xie et al. 3.1); there is no
    positional encoding anywhere in the model.  C2/DCM injects a depth coordinate here.
    """

    def __init__(self, dim: int, norm: str = "bn"):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim) if norm == "bn" else nn.Identity()

    def forward(self, x, grid):
        B, N, C = x.shape
        D, H, W = grid
        assert D * H * W == N, f"grid {grid} does not match N={N}"
        x = x.transpose(1, 2).reshape(B, C, D, H, W)          # explicit grid
        x = self.bn(self.dwconv(x))
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, drop: float = 0.0,
                 dw_norm: str = "bn", dcm: bool = False, dcm_frame: str = "absolute",
                 dcm_depth: int = 32):
        super().__init__()
        hidden = mlp_ratio * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv3D(hidden, norm=dw_norm)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.dcm = DepthCoordinate(hidden, dcm_depth, dcm_frame) if dcm else None

    def forward(self, x, grid):
        x = self.fc1(x)
        x = self.dwconv(x, grid)
        if self.dcm is not None:                              # C2, immediately after DW
            x = self.dcm(x, grid)
        x = self.drop(self.act(x))
        return self.drop(self.fc2(x))


class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, sr, mlp_ratio=4, qkv_bias=True, attn_drop=0.0,
                 proj_drop=0.0, drop_path=0.0, scale_mode="head_dim", dw_norm="bn",
                 sr_overlap=False, pdk=False, pdk_grid=(8, 8, 8), dcm=False,
                 dcm_frame="absolute", dcm_depth=32, use_sdpa=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SelfAttention3D(dim, num_heads, sr, qkv_bias, attn_drop,
                                         proj_drop, scale_mode, sr_overlap, pdk,
                                         pdk_grid, use_sdpa)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, mlp_ratio, proj_drop, dw_norm, dcm, dcm_frame, dcm_depth)
        if drop_path > 0.0:
            from timm.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, grid, halo=None):
        x = x + self.drop_path(self.attention(self.norm1(x), grid, halo))
        x = x + self.drop_path(self.mlp(self.norm2(x), grid))
        return x
