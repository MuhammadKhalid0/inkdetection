"""Unified model builder: stem -> backbone -> depth-collapse -> 2D head.

forward(x:[B,1,D,H,W]) -> logits:[B,1,H,W], identical interface for every model.
For A0: stem OFF, backbone = adapted UNETR++, collapse = mean.
"""
import torch.nn as nn

from .shared import TextureStem, DepthCollapse, Head
from .unetrpp import build_unetrpp_ink


class UnifiedModel(nn.Module):
    def __init__(self, stem, backbone, collapse, head):
        super().__init__()
        self.stem = stem
        self.backbone = backbone
        self.collapse = collapse
        self.head = head

    def forward(self, x):  # x: [B,1,D,H,W]
        if self.stem is not None:
            x = self.stem(x)
        feat = self.backbone(x)                 # [B,C,D,H,W]
        s_hat = getattr(self.backbone, "last_s_hat", None)
        x2d = self.collapse(feat, s_hat)        # [B,C,H,W]
        return self.head(x2d)                   # [B,1,H,W]


def build_model(cfg) -> UnifiedModel:
    m = cfg["model"]
    d = cfg["data"]
    shared = cfg.get("shared", {})

    # --- input stem -------------------------------------------------------
    if shared.get("texture_stem", False):
        stem = TextureStem(out_extra=shared.get("stem_out_extra", 8))
        in_ch = stem.out_ch
    else:
        stem = None
        in_ch = 1

    D, H, W = d["depth"], d["tile"], d["tile"]

    name = m["name"]
    if name == "unetrpp":
        backbone = build_unetrpp_ink(
            in_ch=in_ch,
            img_size=(D, H, W),
            feature_size=m.get("feature_size", 16),
            depths=tuple(m.get("depths", [3, 3, 3, 3])),
            dims=tuple(m.get("dims", [32, 64, 128, 256])),
            num_heads=m.get("num_heads", 4),
            patch_size=tuple(m.get("patch_size", [2, 4, 4])),
            pe_mode=m.get("pe_mode", "3d"),
            depth_attn=m.get("depth_attn", False),
            da_gamma_init=m.get("da_gamma_init", 0.1),
            surface_relative=m.get("surface_relative", False),
            per_pixel=m.get("per_pixel", False),
            warp_stabilize=m.get("warp_stabilize", False),
        )
        feat_ch = backbone.out_channels
    elif name == "nnformer":
        # nnFormer-Ink port (2026-07-19): models/nnformer -> $WORK/nnFormer/src/nnformer_ink
        # (symlink; single source of truth in the nnFormer workspace).
        from .nnformer import build_nnformer_ink
        backbone = build_nnformer_ink(
            in_ch=in_ch,
            img_size=(D, H, W),
            capacity=m.get("capacity", "S"),
            patch_size=tuple(m.get("patch_size", [1, 4, 4])),
            gv_bottleneck=m.get("gv_bottleneck", True),
            skip_attention=m.get("skip_attention", True),
            drop_path=m.get("drop_path", 0.1),
            out_channels=m.get("feature_size", 16),
            cssa=m.get("cssa", False),
            sr_pb=m.get("sr_pb", False),
            vffc_bottleneck=m.get("vffc_bottleneck", False),
        )
        feat_ch = backbone.out_channels
    elif name == "segformer3d":
        # SegFormer3D-Ink port (2026-09-10): models/segformer3d -> $WORK/segFormer/src/segformer3d_ink
        # (symlink; single source of truth in the segFormer workspace).
        # NOTE: every model.* key MUST be forwarded here. A key present in the YAML but
        # not forwarded is SILENTLY IGNORED and the run trains a different model than the
        # config describes, with no error (segformer3d_plan.md 3.3). The unit test
        # `test_config_keys_reach_the_model` guards this list.
        from .segformer3d import build_segformer3d_ink
        backbone = build_segformer3d_ink(
            in_ch=in_ch,
            img_size=(D, H, W),
            capacity=m.get("capacity", "S"),
            stem=m.get("stem", "dp"),
            sr=m.get("sr", "144_122"),
            final=m.get("final", "expand"),
            skip_stem=m.get("skip_stem", True),
            drop_path=m.get("drop_path", 0.0),
            out_channels=m.get("feature_size", 16),
            dec_dim=m.get("dec_dim", 256),
            dw_norm=m.get("dw_norm", "bn"),
            dec_proj_ln=m.get("dec_proj_ln", True),
            scale_mode=m.get("scale_mode", "head_dim"),
            pdk=m.get("pdk", False),
            pdk_stages=tuple(m.get("pdk_stages", [2, 3, 4])),
            pdk_grid=tuple(m.get("pdk_grid", [8, 8, 8])),
            dcm=m.get("dcm", False),
            dcm_frame=m.get("dcm_frame", "absolute"),
            dcm_stages=tuple(m.get("dcm_stages", [1, 2, 3, 4])),
            sr_overlap=m.get("sr_overlap", False),
            dsm=m.get("dsm", False),
            dsm_dilation=m.get("dsm_dilation", 1),
            mode=m.get("mode", "3d"),
        )
        feat_ch = backbone.out_channels
    else:
        raise ValueError(f"model '{name}' not implemented (unetrpp | nnformer | segformer3d)")

    collapse = DepthCollapse(depth=D, mode=shared.get("collapse", "mean"), channels=feat_ch,
                             per_pixel=m.get("per_pixel", False),
                             warp_stabilize=m.get("warp_stabilize", False))
    # head_classes (2026-07-17): 1 (default, sigmoid — every pre-existing run)
    # or 3 (u2_soft3 uncertainty variant — softmax ink/bg/uncertain).
    head = Head(feat_ch, out_ch=int(m.get("head_classes", 1)))
    return UnifiedModel(stem, backbone, collapse, head)
