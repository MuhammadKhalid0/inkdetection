"""SegFormer3D-Ink backbone: MiT encoder + all-MLP decoder -> [B,16,D,t,t].

Wrapper contract (identical to the UNETR++ and nnFormer ports):
    forward(x: [B,in_ch,D,H,W]) -> [B,out_channels,D,H,W]
    backbone.out_channels / stage_res / input_size / feat_size
"""
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .blocks import PatchEmbed3D, TransformerBlock3D
from .decoder import SegFormerDecoder3D
from .geometry import (CAPACITY_PRESETS, SR_PRESETS, STEMS, assert_exact,
                       assert_sr_divides, kv_lengths, stage_grids, token_counts)


class SegFormer3DInk(nn.Module):
    def __init__(self, in_ch: int = 1, img_size: Sequence[int] = (32, 192, 192),
                 capacity: str = "S", stem: str = "dp", sr: str = "144_122",
                 final: str = "expand", skip_stem: bool = True, drop_path: float = 0.0,
                 out_channels: int = 16, dec_dim: int = 256, dw_norm: str = "bn",
                 dec_proj_ln: bool = True, scale_mode: str = "head_dim",
                 mlp_ratio: int = 4, qkv_bias: bool = True, dropout: float = 0.0,
                 pdk: bool = False, pdk_stages: Sequence[int] = (2, 3, 4),
                 pdk_grid: Sequence[int] = (8, 8, 8),
                 dcm: bool = False, dcm_frame: str = "absolute",
                 dcm_stages: Sequence[int] = (1, 2, 3, 4),
                 sr_overlap: bool = False, dsm: bool = False, dsm_dilation: int = 1,
                 use_sdpa: bool = True, mode: str = "3d",
                 dims=None, heads=None, depths=None):
        super().__init__()
        if capacity not in CAPACITY_PRESETS:
            raise ValueError(f"unknown capacity '{capacity}'")
        if stem not in STEMS:
            raise ValueError(f"unknown stem '{stem}'")
        if sr not in SR_PRESETS:
            raise ValueError(f"unknown sr preset '{sr}'")
        # 2.5D control: the depth layers become input channels and the depth axis is a
        # singleton, so the shared DepthCollapse/Head stay byte-identical shared code
        # (the output is broadcast back over depth, and every depth slice is equal, so any
        # depth weighting returns the same 2D map). Reported as a CONTROL, never a
        # benchmark row (plan 2.6).
        self.mode = mode
        if mode == "25d":
            if stem != "2d":
                stem = "2d"
            in_ch, img_size = int(img_size[0]), (1, img_size[1], img_size[2])
        pre = CAPACITY_PRESETS[capacity]
        self.dims = tuple(dims or pre["dims"])
        self.heads = tuple(heads or pre["heads"])
        self.depths = tuple(depths or pre["depths"])
        self.sr_name, self.stem_name = sr, stem
        self.sr = SR_PRESETS[sr]
        self.img_size = tuple(int(v) for v in img_size)
        self.in_ch, self.out_channels = in_ch, out_channels
        self.orig_depth = in_ch if mode == "25d" else self.img_size[0]

        # --- geometry contract (build-time, no padding paths, no cube_root) ---------
        assert_exact(self.img_size, stem)
        self.grids = stage_grids(self.img_size, stem)
        assert_sr_divides(self.grids, self.sr)
        self.stage_res = [tuple(g) for g in self.grids]
        self.input_size = token_counts(self.grids)
        self.feat_size = self.stage_res[-1]
        self.kv_len = kv_lengths(self.grids, self.sr)

        cfg = STEMS[stem]
        self.pdk, self.pdk_stages = pdk, tuple(pdk_stages)
        self.dcm_stages = tuple(dcm_stages)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(self.depths))] \
            if drop_path > 0 else [0.0] * sum(self.depths)

        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev, b = in_ch, 0
        for i in range(4):
            self.embeds.append(PatchEmbed3D(prev, self.dims[i], cfg["kernel"][i],
                                            cfg["stride"][i], cfg["padding"][i]))
            blocks = nn.ModuleList()
            for _ in range(self.depths[i]):
                blocks.append(TransformerBlock3D(
                    self.dims[i], self.heads[i], self.sr[i], mlp_ratio, qkv_bias,
                    0.0, dropout, dpr[b], scale_mode, dw_norm, sr_overlap,
                    pdk and (i + 1) in self.pdk_stages, tuple(pdk_grid),
                    dcm and (i + 1) in self.dcm_stages, dcm_frame,
                    self.grids[i][0], use_sdpa))
                b += 1
            self.stages.append(blocks)
            self.norms.append(nn.LayerNorm(self.dims[i]))
            prev = self.dims[i]

        self.decoder = SegFormerDecoder3D(
            self.dims, dec_dim, out_channels, dropout, dec_proj_ln, final,
            cfg["stride"][0], skip_stem, in_ch, dsm, dsm_dilation)
        self.apply(self._init_weights)
        # C3/OSR must be applied AFTER _init_weights: the widened kernel is built from
        # the already-initialised stock conv, so the core block carries the stock draw
        # (stock fan-out) and every extra tap is exactly 0.0 -> exact reduction.
        if sr_overlap:
            from .contrib import widen_sr_conv
            for blocks, r in zip(self.stages, self.sr):
                if all(v == 1 for v in r):
                    continue
                for blk in blocks:
                    a = blk.attention
                    a.sr_conv = widen_sr_conv(a.sr_conv, a.sr)

    # reference _init_weights, verbatim in behaviour
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = (m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
                       * m.out_channels) // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C1 data carrier: the halo rides as channel 1 (same rank, same voxel count).
        # MISSING-CHANNEL RULE: a 1-channel input skips the peripheral branch entirely,
        # which keeps the SHARED smoke test / ERF / Inkalyzer probes working unmodified
        # and gives a free "halo ablated at inference" cell (plan 2.1).
        if self.mode == "25d":
            x = x.squeeze(1).unsqueeze(2)            # [B,1,D,t,t] -> [B,D,1,t,t]
        halo = None
        if x.shape[1] > self.in_ch:
            halo = x[:, self.in_ch:self.in_ch + 1]
            x = x[:, :self.in_ch]
        vol = x
        feats = []
        for i in range(4):
            x, grid = self.embeds[i](x)
            if tuple(grid) != tuple(self.grids[i]):
                raise RuntimeError(f"stage {i+1} grid {grid} != built grid "
                                   f"{self.grids[i]} (img_size mismatch)")
            for blk in self.stages[i]:
                x = blk(x, grid, halo)
            x = self.norms[i](x)
            x = x.transpose(1, 2).reshape(x.shape[0], self.dims[i], *grid)
            feats.append(x)
        out = self.decoder(feats, vol)
        if self.mode == "25d":
            out = out.expand(-1, -1, self.orig_depth, -1, -1)   # broadcast over depth
        return out


def build_segformer3d_ink(in_ch: int = 1, img_size=(32, 192, 192), capacity: str = "S",
                          stem: str = "dp", sr: str = "144_122", final: str = "expand",
                          skip_stem: bool = True, drop_path: float = 0.0,
                          out_channels: int = 16, dec_dim: int = 256,
                          dw_norm: str = "bn", dec_proj_ln: bool = True,
                          scale_mode: str = "head_dim", pdk: bool = False,
                          pdk_stages=(2, 3, 4), pdk_grid=(8, 8, 8), dcm: bool = False,
                          dcm_frame: str = "absolute", dcm_stages=(1, 2, 3, 4),
                          sr_overlap: bool = False, dsm: bool = False,
                          dsm_dilation: int = 1, mode: str = "3d",
                          **kw) -> SegFormer3DInk:
    return SegFormer3DInk(
        in_ch=in_ch, img_size=img_size, capacity=capacity, stem=stem, sr=sr, final=final,
        skip_stem=skip_stem, drop_path=drop_path, out_channels=out_channels,
        dec_dim=dec_dim, dw_norm=dw_norm, dec_proj_ln=dec_proj_ln,
        scale_mode=scale_mode, pdk=pdk, pdk_stages=pdk_stages, pdk_grid=pdk_grid,
        dcm=dcm, dcm_frame=dcm_frame, dcm_stages=dcm_stages, sr_overlap=sr_overlap,
        dsm=dsm, dsm_dilation=dsm_dilation, mode=mode, **kw)
