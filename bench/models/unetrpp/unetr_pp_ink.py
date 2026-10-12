"""Adapted UNETR++ for the Vesuvius ink task (A0 baseline).

Differences from the published `synapse/unetr_pp_synapse.py`:
  * Inherits ``nn.Module`` (not ``SegmentationNetwork``) so it does not drag in
    nnU-Net / batchgenerators inference machinery — we run our own sliding window.
  * ``input_size`` (encoder) and ``out_size`` (decoders) are COMPUTED from the
    actual ``img_size`` instead of being hardcoded to the medical (64,128,128)
    volume.  The published class silently uses medical token counts, which are
    wrong for the (32,64,64) ink patch and break the EPA E/F projections and the
    learnable positional embeddings.
  * ``forward`` returns the last decoder FEATURE volume ``[B, feature_size, D, H, W]``
    (the 3D segmentation head is bypassed); the unified wrapper does the
    depth->2D collapse and 2D head.

For A0 the patch embed is the medical default (2,4,4) == paper "(4,4,2)"
(depth /2, H /4, W /4) and depths/dims/heads are the published values.
"""
from typing import Sequence, Tuple, Union
import math

import torch
from torch import nn

from .dynunet_block import UnetResBlock
from .model_components import UnetrPPEncoder, UnetrUpBlock


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


class UNETR_PP_Ink(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        norm_name: Union[Tuple, str] = "instance",
        depths: Sequence[int] = (3, 3, 3, 3),
        dims: Sequence[int] = (32, 64, 128, 256),
        img_size: Sequence[int] = (32, 64, 64),
        patch_size: Sequence[int] = (2, 4, 4),
        proj_size: Sequence[int] = (64, 64, 64, 32),
        transformer_dropout_rate: float = 0.15,
        pe_mode: str = "3d",
        depth_attn: bool = False,
        da_gamma_init: float = 0.1,
        surface_relative: bool = False,
        per_pixel: bool = False,
        warp_stabilize: bool = False,
    ) -> None:
        super().__init__()
        assert len(img_size) == 3 and len(patch_size) == 3
        assert hidden_size == dims[-1], "hidden_size must equal dims[-1]"

        self.in_channels = in_channels
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.pe_mode = pe_mode

        # --- resolution bookkeeping (the crux) --------------------------------
        # res after the stem (downsampled by patch_size), then 3x (2,2,2).
        res0 = tuple(img_size[i] // patch_size[i] for i in range(3))
        for i in range(3):
            if img_size[i] % patch_size[i] != 0:
                raise ValueError(f"img_size[{i}]={img_size[i]} not divisible by patch_size[{i}]={patch_size[i]}")
        res = [res0]
        for _ in range(3):
            prev = res[-1]
            nxt = tuple(r // 2 for r in prev)
            if any(r == 0 for r in nxt):
                raise ValueError(f"Stage resolution collapsed to 0: {prev} -> {nxt}. img_size too small.")
            res.append(nxt)
        # res[0..3] = stage input resolutions; res[3] = bottleneck (feat_size)
        self.stage_res = res
        self.feat_size = res[3]
        input_size = [_prod(r) for r in res]          # encoder EPA token counts
        self.input_size = input_size
        stage_depths = [r[0] for r in res]            # z-slices per stage (for 2D PE)

        # --- encoder ----------------------------------------------------------
        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=input_size,
            dims=list(dims),
            proj_size=list(proj_size),
            depths=list(depths),
            num_heads=num_heads,
            in_channels=in_channels,
            transformer_dropout_rate=transformer_dropout_rate,
            pe_mode=pe_mode,
            stage_depths=stage_depths,
            patch_size=self.patch_size,
            depth_attn=depth_attn,
            da_gamma_init=da_gamma_init,
            surface_relative=surface_relative,
            per_pixel=per_pixel,
            warp_stabilize=warp_stabilize,
        )

        # full-resolution conv skip (fused in the last decoder)
        self.encoder1 = UnetResBlock(
            spatial_dims=3, in_channels=in_channels, out_channels=feature_size,
            kernel_size=3, stride=1, norm_name=norm_name,
        )

        # --- decoders (out_size computed from res, NOT hardcoded) -------------
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3, in_channels=feature_size * 16, out_channels=feature_size * 8,
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name,
            proj_size=proj_size[2], num_heads=num_heads, out_size=_prod(res[2]),
            pe_mode=pe_mode, pe_depth=res[2][0],
            depth_attn=depth_attn, da_gamma_init=da_gamma_init, surface_relative=surface_relative,
            per_pixel=per_pixel, warp_stabilize=warp_stabilize,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3, in_channels=feature_size * 8, out_channels=feature_size * 4,
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name,
            proj_size=proj_size[1], num_heads=num_heads, out_size=_prod(res[1]),
            pe_mode=pe_mode, pe_depth=res[1][0],
            depth_attn=depth_attn, da_gamma_init=da_gamma_init, surface_relative=surface_relative,
            per_pixel=per_pixel, warp_stabilize=warp_stabilize,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3, in_channels=feature_size * 4, out_channels=feature_size * 2,
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name,
            proj_size=proj_size[0], num_heads=num_heads, out_size=_prod(res[0]),
            pe_mode=pe_mode, pe_depth=res[0][0],
            depth_attn=depth_attn, da_gamma_init=da_gamma_init, surface_relative=surface_relative,
            per_pixel=per_pixel, warp_stabilize=warp_stabilize,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3, in_channels=feature_size * 2, out_channels=feature_size,
            kernel_size=3, upsample_kernel_size=tuple(patch_size), norm_name=norm_name,
            out_size=_prod(img_size), conv_decoder=True,
        )

    @property
    def out_channels(self) -> int:
        return self.feature_size

    def proj_feat(self, x):
        # x: [B, N, hidden] -> [B, hidden, d0, d1, d2]
        f = self.feat_size
        x = x.view(x.size(0), f[0], f[1], f[2], self.hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        # x_in: [B, in_channels, D, H, W]
        _, hidden_states = self.unetr_pp_encoder(x_in)
        conv_block = self.encoder1(x_in)

        enc1, enc2, enc3, enc4 = hidden_states  # res0,res1,res2 volumes; enc4 tokens

        dec4 = self.proj_feat(enc4)             # bottleneck volume
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, conv_block)   # [B, feature_size, D, H, W]
        return out


def build_unetrpp_ink(in_ch: int = 1, img_size=(32, 64, 64), feature_size: int = 16,
                      depths=(3, 3, 3, 3), dims=(32, 64, 128, 256), num_heads: int = 4,
                      patch_size=(2, 4, 4), pe_mode: str = "3d",
                      depth_attn: bool = False, da_gamma_init: float = 0.1,
                      surface_relative: bool = False, per_pixel: bool = False,
                      warp_stabilize: bool = False) -> UNETR_PP_Ink:
    return UNETR_PP_Ink(
        in_channels=in_ch, img_size=img_size, feature_size=feature_size,
        depths=depths, dims=dims, hidden_size=dims[-1], num_heads=num_heads,
        patch_size=patch_size, pe_mode=pe_mode,
        depth_attn=depth_attn, da_gamma_init=da_gamma_init,
        surface_relative=surface_relative, per_pixel=per_pixel,
        warp_stabilize=warp_stabilize,
    )
