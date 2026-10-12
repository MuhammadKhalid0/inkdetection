"""nnFormer-Ink backbone: encoder / decoder / full-res feature head.

Wrapper contract (vesuvius-ink-bench models/wrapper.py):
    forward(x: [B, in_ch, D, H, W]) -> [B, out_channels=16, D, H, W]
The shared DepthCollapse + Head produce the 2D ink logits outside this module.

Config-derived geometry (nnformer_plan.md §3.1), all asserted at build time:
    embed (1,4,4) -> S1 (D, t/4, t/4) -> S2 -> S3 -> S4 = bottleneck (GV).
Ablation toggles: gv_bottleneck (n_lvonly), skip_mode attn|conv (n_skipconv),
patch_size (2,4,4) (n_embedz2). cssa / sr_pb are reserved for port step 2 and
assert off. deep_supervision must stay False (benchmark protocol, plan §2.3).
"""
from typing import List, Sequence

import torch
import torch.nn as nn

from .blocks import SwinBlock3D, SwinBlockKV3D
from .contrib import VFFCResBlock
from .sampling import FinalExpand3D, PatchEmbed3D, PatchExpanding3D, PatchMerging3D

# dims per stage, head_dim 32 throughout. Calibrated against MEASURED counts
# (tests/test_smoke.py, 2026-07-19): S ~19.9M (UNETR++ t192-class), M ~79M
# (capacity-matched to deployed UNETR++ t256 = 79.1M), F ~178M (published-class;
# the paper's 150.5M uses embed_dim 192 with a different head/DS layout).
CAPACITY_PRESETS = {
    "S": {"dims": [64, 128, 256, 512], "num_heads": [2, 4, 8, 16]},
    "M": {"dims": [128, 256, 512, 1024], "num_heads": [4, 8, 16, 32]},
    "F": {"dims": [192, 384, 768, 1536], "num_heads": [6, 12, 24, 48]},
}


def _stage_grids(img_size: Sequence[int], patch_size: Sequence[int], n_stages: int):
    D, H, W = img_size
    p0, p1, p2 = patch_size
    assert D % p0 == 0 and H % p1 == 0 and W % p2 == 0, \
        f"img_size {tuple(img_size)} not divisible by patch {tuple(patch_size)}"
    grids = []
    g = (D // p0, H // p1, W // p2)
    for i in range(n_stages):
        for ax, v in enumerate(g):
            assert v >= 1, f"stage {i} grid {g} collapsed on axis {ax}"
        grids.append(g)
        if i < n_stages - 1:
            assert all(v % 2 == 0 for v in g), \
                f"stage {i} grid {g} not divisible by 2 for merging"
            g = tuple(v // 2 for v in g)
    return grids


def _lv_bottleneck_window(grid, cap):
    """Window for the n_lvonly ablation (gv_bottleneck=False): per axis the
    largest divisor of the grid that is <= the S3 window cap; in-plane it must
    also be strictly smaller than the grid (otherwise it would be GV again).
    Depth may equal the grid (bottleneck depth is only 4 tokens).
    E.g. t192 grid (4,6,6), cap (4,4,4) -> (4,3,3); t256 (4,8,8) -> (4,4,4)."""
    def best(g, c, strict):
        cands = [d for d in range(1, g + 1) if g % d == 0 and d <= c and (d < g or not strict)]
        return max(cands) if cands else 1
    return (best(grid[0], cap[0], strict=False),
            best(grid[1], cap[1], strict=True),
            best(grid[2], cap[2], strict=True))


class EncoderStage(nn.Module):
    def __init__(self, dim, grid, depth, num_heads, window, drop_path, downsample,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, sr_pb=False):
        super().__init__()
        self.grid = tuple(grid)
        self.blocks = nn.ModuleList([
            SwinBlock3D(dim, grid, num_heads, window, shifted=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i], sr_pb=sr_pb)
            for i in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim)
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x):  # [B, L, C] -> (skip tokens, downsampled tokens or None)
        for blk in self.blocks:
            x = blk(x)
        skip = self.norm_out(x)
        down = self.downsample(x, self.grid) if self.downsample is not None else None
        return skip, down


class VFFCStage(nn.Module):
    """Bottleneck stage using vFFC residual blocks instead of GV-MSA (plan §2.4).
    Same (skip, down) interface as EncoderStage so the encoder loop is unchanged.
    Tokens [B,L,C] <-> volume [B,C,D,H,W] around the conv/FFT blocks."""

    def __init__(self, dim, grid, n_blocks=3, ratio=0.5):
        super().__init__()
        self.grid = tuple(grid)
        self.blocks = nn.ModuleList([VFFCResBlock(dim, ratio) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(dim)
        self.downsample = None

    def forward(self, x):
        B, L, C = x.shape
        D, H, W = self.grid
        assert L == D * H * W, f"token count {L} != grid {self.grid}"
        v = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        for blk in self.blocks:
            v = blk(v)
        x = v.flatten(2).transpose(1, 2).contiguous()
        return self.norm_out(x), None


class DecoderStage(nn.Module):
    """PatchExpanding from the coarser grid, then skip fusion + blocks at `grid`."""

    def __init__(self, dim, grid, depth, num_heads, window, drop_path, skip_mode,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, sr_pb=False, cssa_dims=None):
        super().__init__()
        assert skip_mode in ("attn", "conv")
        self.grid = tuple(grid)
        self.skip_mode = skip_mode
        self.coarse_grid = tuple(v // 2 for v in grid)
        self.upsample = PatchExpanding3D(dim * 2)
        blocks: List[nn.Module] = []
        if skip_mode == "attn":
            blocks.append(SwinBlockKV3D(dim, grid, num_heads, window,
                                        mlp_ratio=mlp_ratio, drop=drop,
                                        attn_drop=attn_drop, drop_path=drop_path[0],
                                        sr_pb=sr_pb, cssa_dims=cssa_dims))
            rest = range(1, depth)
        else:
            self.skip_fuse = nn.Linear(2 * dim, dim)
            blocks.append(SwinBlock3D(dim, grid, num_heads, window, shifted=False,
                                      mlp_ratio=mlp_ratio, drop=drop,
                                      attn_drop=attn_drop, drop_path=drop_path[0],
                                      sr_pb=sr_pb))
            rest = range(1, depth)
        for i in rest:
            blocks.append(SwinBlock3D(dim, grid, num_heads, window, shifted=True,
                                      mlp_ratio=mlp_ratio, drop=drop,
                                      attn_drop=attn_drop, drop_path=drop_path[i],
                                      sr_pb=sr_pb))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skip, coarse_tokens=None):
        # x: coarse tokens [B, L/8, 2C]; skip: [B, L, C]
        x_up = self.upsample(x, self.coarse_grid)  # [B, L, C]
        if self.skip_mode == "attn":
            x = x_up + skip
            x = self.blocks[0](x, skip=skip, x_up=x_up, coarse_tokens=coarse_tokens)
            for blk in self.blocks[1:]:
                x = blk(x)
        else:
            x = self.skip_fuse(torch.cat([x_up, skip], dim=-1))
            for blk in self.blocks:
                x = blk(x)
        return x


class NnFormerInk(nn.Module):
    def __init__(self, in_ch=1, img_size=(32, 192, 192), capacity="S",
                 dims=None, num_heads=None, patch_size=(1, 4, 4),
                 depths=(2, 2, 2, 3), decoder_depths=(2, 2, 2),
                 window_sizes=((4, 8, 8), (4, 8, 8), (4, 4, 4)),
                 gv_bottleneck=True, skip_attention=True,
                 drop_path=0.1, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 out_channels=16, cssa=False, cssa_scales=None, sr_pb=False,
                 vffc_bottleneck=False, vffc_ratio=0.5, deep_supervision=False):
        super().__init__()
        assert not deep_supervision, "DS is OFF for all benchmark models (plan §2.3)"
        preset = CAPACITY_PRESETS[capacity]
        dims = list(dims) if dims is not None else list(preset["dims"])
        num_heads = list(num_heads) if num_heads is not None else list(preset["num_heads"])
        n_stages = len(dims)
        assert n_stages == len(depths) == len(num_heads)
        assert len(decoder_depths) == n_stages - 1
        assert len(window_sizes) == n_stages - 1, "bottleneck window is derived (GV/LV)"

        self.img_size = tuple(img_size)
        self.out_channels = out_channels
        grids = _stage_grids(img_size, patch_size, n_stages)
        self.grids = grids
        # bookkeeping attributes read by vesuvius-ink-bench scripts/smoke_test.py
        self.stage_res = [tuple(g) for g in grids]
        self.feat_size = tuple(grids[-1])
        self.input_size = [g[0] * g[1] * g[2] for g in grids]

        # bottleneck window: GV = full grid (matches original behaviour where
        # window >= resolution disables partitioning); LV ablation = largest
        # grid-dividing local window capped by the S3 window (see helper)
        bottleneck_window = grids[-1] if gv_bottleneck \
            else _lv_bottleneck_window(grids[-1], window_sizes[-1])
        enc_windows = list(window_sizes) + [bottleneck_window]

        self.patch_embed = PatchEmbed3D(patch_size, in_ch, dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.encoder = nn.ModuleList()
        for i in range(n_stages):
            dp = dpr[sum(depths[:i]):sum(depths[:i + 1])]
            if vffc_bottleneck and i == n_stages - 1:
                # prior-art cell (Quattrini et al.): spectral conv instead of
                # global attention for the same "global context" job
                self.encoder.append(VFFCStage(dims[i], grids[i],
                                              n_blocks=depths[i], ratio=vffc_ratio))
            else:
                self.encoder.append(EncoderStage(
                    dims[i], grids[i], depths[i], num_heads[i], enc_windows[i],
                    drop_path=dp, downsample=(i < n_stages - 1),
                    mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, sr_pb=sr_pb))

        dpr_d = [x.item() for x in torch.linspace(0, drop_path, sum(decoder_depths))]
        self.decoder = nn.ModuleList()
        skip_mode = "attn" if skip_attention else "conv"
        # CSSA (plan §2.2): decoder stages additionally attend into the COARSE
        # encoder scales. Default = the bottleneck (few tokens: 144 at t192,
        # 256 at t256) so the added cost stays small.
        self.cssa = bool(cssa)
        self.cssa_scales = list(cssa_scales) if cssa_scales is not None else [n_stages - 1]
        cdims = [dims[s] for s in self.cssa_scales] if self.cssa else None
        for j, i in enumerate(reversed(range(n_stages - 1))):  # coarse -> fine
            dp = dpr_d[sum(decoder_depths[:j]):sum(decoder_depths[:j + 1])]
            # a decoder stage only queries scales strictly coarser than itself
            stage_cdims = ([dims[s] for s in self.cssa_scales if s > i]
                           if self.cssa and skip_mode == "attn" else None)
            self.decoder.append(DecoderStage(
                dims[i], grids[i], decoder_depths[j], num_heads[i], window_sizes[i],
                drop_path=dp, skip_mode=skip_mode,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                sr_pb=sr_pb, cssa_dims=stage_cdims))
            self.decoder[-1].cssa_src = [s for s in self.cssa_scales if s > i] \
                if (self.cssa and skip_mode == "attn") else []

        # full-resolution feature head: expand + conv-skip stem (UNETR++ parity,
        # substitutes for the removed deep supervision — plan §1)
        self.final_expand = FinalExpand3D(dims[0], out_channels, patch_size)
        self.full_res_stem = nn.Sequential(
            nn.Conv3d(in_ch, out_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, 3, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 1),
        )

    def forward(self, x):  # [B, in, D, H, W]
        assert tuple(x.shape[2:]) == self.img_size, \
            f"input {tuple(x.shape[2:])} != configured img_size {self.img_size}"
        stem = self.full_res_stem(x)

        t = self.patch_embed(x)  # [B, C0, D0, H0, W0]
        B, C0 = t.shape[0], t.shape[1]
        t = t.flatten(2).transpose(1, 2).contiguous()

        skips = []
        for stage in self.encoder:
            skip, down = stage(t)
            skips.append(skip)
            t = down if down is not None else skip

        t = skips[-1]  # bottleneck tokens (normed)
        for j, dec in enumerate(self.decoder):
            skip = skips[len(self.encoder) - 2 - j]
            coarse = [skips[s] for s in getattr(dec, "cssa_src", [])] or None
            t = dec(t, skip, coarse_tokens=coarse)

        D0, H0, W0 = self.grids[0]
        vol = t.transpose(1, 2).contiguous().view(B, C0, D0, H0, W0)
        vol = self.final_expand(vol)  # [B, out, D, H, W]
        return self.fuse(torch.cat([vol, stem], dim=1))


def build_nnformer_ink(in_ch=1, img_size=(32, 192, 192), capacity="S",
                       patch_size=(1, 4, 4), gv_bottleneck=True, skip_attention=True,
                       drop_path=0.1, out_channels=16, cssa=False, cssa_scales=None,
                       sr_pb=False, vffc_bottleneck=False, deep_supervision=False,
                       **kw) -> NnFormerInk:
    return NnFormerInk(in_ch=in_ch, img_size=img_size, capacity=capacity,
                       patch_size=patch_size, gv_bottleneck=gv_bottleneck,
                       skip_attention=skip_attention, drop_path=drop_path,
                       out_channels=out_channels, cssa=cssa, cssa_scales=cssa_scales,
                       sr_pb=sr_pb, vffc_bottleneck=vffc_bottleneck,
                       deep_supervision=deep_supervision, **kw)
