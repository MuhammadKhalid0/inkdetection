"""All-MLP decoder + the full-resolution path to the wrapper's [B,16,D,t,t] contract.

The decoder is faithful to the reference (per-stage 1x1x1 projection, trilinear resize to
the stage-1 grid, concat, 1x1x1 fuse + BN + ReLU).  What is NOT faithful, deliberately:
the reference's final ``nn.Upsample(scale_factor=4.0)`` is ISOTROPIC and would return
depth 128 under the depth-preserving stem (plan 3.4 item 5), so the final path is
explicit and configurable.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrib import DecoderSpatialMixing


class StageProj(nn.Module):
    """Reference ``MLP_``: Linear(C_i -> dec) + LayerNorm (the repo's added norm)."""

    def __init__(self, in_dim: int, dim: int, use_ln: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.bn = nn.LayerNorm(dim) if use_ln else nn.Identity()

    def forward(self, x):                       # x: [B,C,D,H,W]
        x = x.flatten(2).transpose(1, 2)
        return self.bn(self.proj(x))


class SkipStem(nn.Module):
    """Full-resolution conv skip -- parity with the UNETR++ / nnFormer ports."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, padding=1), nn.GELU(),
            nn.Conv3d(8, out_ch, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * out_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 1))

    def forward(self, vol, feat):
        return self.fuse(torch.cat([self.stem(vol), feat], dim=1))


class SegFormerDecoder3D(nn.Module):
    def __init__(self, dims, dec_dim: int = 256, out_ch: int = 16, dropout: float = 0.0,
                 dec_proj_ln: bool = True, final: str = "expand",
                 stem_stride: Tuple[int, int, int] = (1, 4, 4), skip_stem: bool = True,
                 in_ch: int = 1, dsm: bool = False, dsm_dilation: int = 1):
        super().__init__()
        # reference order: decoder receives the reversed dim list (c4 first)
        rev = list(dims)[::-1]
        self.linear_c4 = StageProj(rev[0], dec_dim, dec_proj_ln)
        self.linear_c3 = StageProj(rev[1], dec_dim, dec_proj_ln)
        self.linear_c2 = StageProj(rev[2], dec_dim, dec_proj_ln)
        self.linear_c1 = StageProj(rev[3], dec_dim, dec_proj_ln)
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(4 * dec_dim, dec_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(dec_dim), nn.ReLU())
        self.dsm = DecoderSpatialMixing(dec_dim, dsm_dilation) if dsm else None   # C4
        self.dropout = nn.Dropout(dropout)
        self.final_mode = final
        if final == "expand":
            self.up = nn.ConvTranspose3d(dec_dim, out_ch, kernel_size=tuple(stem_stride),
                                         stride=tuple(stem_stride))
        elif final in ("trilinear", "trilinear_lowres_pred"):
            self.pred = nn.Conv3d(dec_dim, out_ch, kernel_size=1)
        else:
            raise ValueError(f"unknown final path '{final}'")
        self.skip = SkipStem(in_ch, out_ch) if skip_stem else None

    def forward(self, feats, vol):
        """feats: [c1,c2,c3,c4] each [B,C_i,D_i,H_i,W_i]; vol: raw input [B,in_ch,D,H,W]."""
        c1, c2, c3, c4 = feats
        n = c1.shape[0]
        g1 = c1.shape[2:]

        def up(proj, c):
            t = proj(c).permute(0, 2, 1).reshape(n, -1, *c.shape[2:])
            return F.interpolate(t, size=g1, mode="trilinear", align_corners=False)

        _c4, _c3, _c2 = up(self.linear_c4, c4), up(self.linear_c3, c3), up(self.linear_c2, c2)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, *g1)
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        if self.dsm is not None:
            x = self.dsm(x)
        x = self.dropout(x)

        full = vol.shape[2:]
        if self.final_mode == "expand":
            x = self.up(x)
        elif self.final_mode == "trilinear":
            # allocates a dec_dim-channel FULL-RESOLUTION intermediate (4.0 GB at t256
            # batch 4 bf16) -- ablation cell only, batch <= 2 (plan 3.1).
            x = F.interpolate(x, size=full, mode="trilinear", align_corners=False)
            x = self.pred(x)
        else:                                   # trilinear_lowres_pred (reference order)
            x = self.pred(x)
            x = F.interpolate(x, size=full, mode="trilinear", align_corners=False)
        if x.shape[2:] != tuple(full):
            x = F.interpolate(x, size=full, mode="trilinear", align_corners=False)
        if self.skip is not None:
            x = self.skip(vol, x)
        return x
