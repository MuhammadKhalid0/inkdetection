"""nnFormer-Ink: adapted nnFormer backbone for Vesuvius ink detection.

Port of Zhou et al., "nnFormer" (IEEE TIP 2023, arXiv:2109.03201), adapted per
`plan(md_files)/nnformer_plan.md`:
  - input (B, in_ch, D=32, t, t), output full-resolution feature volume (B, 16, D, t, t)
    for the shared DepthCollapse + Head of vesuvius-ink-bench (wrapper contract);
  - depth-preserving embedding (patch_size (1,4,4) default);
  - anisotropic tuple windows, correct relative-position index strides (the
    original repo's cubic-window index math collides — see docs/PORT_LOG.md);
  - GV-MSA = full-grid window at the bottleneck stage (matches original behaviour);
  - deep supervision OFF (benchmark protocol); full-res conv skip for UNETR++ parity;
  - every grid/window/token count derived from config, asserted at build time.
"""
from .backbone import build_nnformer_ink, NnFormerInk, CAPACITY_PRESETS

__all__ = ["build_nnformer_ink", "NnFormerInk", "CAPACITY_PRESETS"]
