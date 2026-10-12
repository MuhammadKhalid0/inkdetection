"""The four SegFormer3D-Ink contributions.

House rule (inherited from both previous chapters): every component is an EXACT no-op
at init behind a default-off flag, so each ablation is a genuine single toggle and the
parameter count verifies each rung.

  C1 PDK  -- Peripheral Downsampled Keys       (contrib to attention; plan 2.1)
  C2 DCM  -- Depth-Coordinate Mix-FFN          (contrib to Mix-FFN;   plan 2.2)
  C3 OSR  -- Overlapping Sequence Reduction    (built in blocks.py;   plan 2.3)
  C4 DSM  -- Decoder Spatial Mixing            (contrib to decoder;   plan 2.4)
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- C1 PDK
class ContextStem(nn.Module):
    """Encodes the halo channel DIRECTLY at (or below) a stage's reduced K/V grid.

    The halo is never materialised at full resolution: a 2x-downsampled 2t halo has
    exactly the tile's voxel count, so encoding it through the normal path would double
    the encoder and the 'constant full-resolution cost' claim would be false (plan 2.1).
    """

    def __init__(self, dim: int, kernel: Tuple[int, int, int]):
        super().__init__()
        self.proj = nn.Conv3d(1, dim, kernel_size=kernel, stride=kernel)
        self.norm = nn.LayerNorm(dim)
        self.kernel = tuple(kernel)

    def forward(self, halo: torch.Tensor) -> torch.Tensor:
        """halo: [B,1,D,H,W] (full tile-resolution halo) -> [B, M_p, C] tokens."""
        x = self.proj(halo)                       # [B,C,d,h,w]
        x = x.flatten(2).transpose(1, 2)          # [B,M_p,C]
        return self.norm(x)


class PeripheralGate(nn.Module):
    """Per-head zero-initialised gate for the peripheral attention branch.

    EXACTNESS: out = SDPA(q,k,v) + gamma * SDPA(q,k_p,v_p).  With gamma == 0 the second
    term is multiplied by exactly 0.0 and the sum is bitwise identical to stock.
    NOTE the branch must have its OWN softmax: concatenating the peripheral keys into the
    stock softmax would enter the denominator and would NOT be a no-op (plan 3.4 item 3).
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_heads))

    def forward(self, attn_out: torch.Tensor) -> torch.Tensor:
        # attn_out: [B, heads, N, head_dim]
        return attn_out * self.gamma.view(1, -1, 1, 1)


# --------------------------------------------------------------------------- C2 DCM
class DepthCoordinate(nn.Module):
    """Depth-indexed bias added to the Mix-FFN hidden state, right after the depthwise
    conv -- the op that constitutes the architecture's entire positional apparatus.

    frame='absolute': table P[D_i, 4C] indexed by the token's depth index d.
    frame='surface' : table P[2*D_i-1, 4C] sampled at (d - s_hat) by linear interpolation,
                      s_hat = soft-argmax over a mean depth descriptor (fp32 island).

    EXACTNESS: gamma is zero-init so h is unchanged.  P is trunc_normal (NOT zero) --
    zero-initialising both would leave the branch dead at both ends with no gradient,
    which is the DA-EPA safe-init failure mode (gamma_da ended at 0.1010 from init 0.1).
    """

    def __init__(self, hidden: int, depth: int, frame: str = "absolute"):
        super().__init__()
        assert frame in ("absolute", "surface")
        self.frame, self.depth, self.hidden = frame, depth, hidden
        L = depth if frame == "absolute" else 2 * depth - 1
        self.table = nn.Parameter(torch.empty(L, hidden))
        nn.init.trunc_normal_(self.table, std=0.02)
        self.gamma = nn.Parameter(torch.zeros(hidden))
        if frame == "surface":
            # zero-init estimator -> s_hat starts at the window centre
            self.est = nn.Linear(hidden, 1)
            nn.init.zeros_(self.est.weight)
            nn.init.zeros_(self.est.bias)
        self.probe = False
        self.last_s_hat = None

    def forward(self, h: torch.Tensor, grid) -> torch.Tensor:
        """h: [B, N, hidden] with N = D*H*W of ``grid``."""
        B, N, C = h.shape
        D, H, W = grid
        if self.frame == "absolute":
            bias = self.table[:D]                                   # [D, C]
            bias = bias.view(1, D, 1, C).expand(B, D, H * W, C).reshape(B, N, C)
        else:
            hv = h.view(B, D, H * W, C)
            prof = hv.mean(dim=2).float()                           # [B,D,C] fp32 island
            sal = self.est(prof).squeeze(-1)                        # [B,D]
            w = torch.softmax(sal, dim=-1)
            idx = torch.arange(D, device=h.device, dtype=torch.float32)
            s_hat = (w * idx).sum(dim=-1, keepdim=True)             # [B,1]
            if self.probe:
                self.last_s_hat = s_hat.detach().float().cpu()
            L = self.table.shape[0]
            pos = (idx.unsqueeze(0) - s_hat) + (self.depth - 1)     # [B,D]
            pos = torch.nan_to_num(pos, nan=(L - 1) / 2.0).clamp(0, L - 1)
            lo = pos.floor().long()
            hi = (lo + 1).clamp(max=L - 1)
            frac = (pos - lo.to(pos.dtype)).unsqueeze(-1)
            tab = self.table.to(pos.dtype)
            bias = tab[lo] * (1 - frac) + tab[hi] * frac            # [B,D,C]
            bias = bias.unsqueeze(2).expand(B, D, H * W, C).reshape(B, N, C)
        return h + self.gamma * bias.to(h.dtype)


# --------------------------------------------------------------------------- C4 DSM
class DecoderSpatialMixing(nn.Module):
    """One zero-gated depthwise conv on the fused decoder feature map.

    The all-MLP decoder is purely pointwise; this gives it exactly one neighbourhood
    term.  Primary readout is the post-filter SUBSTITUTION, not F0.5 (plan 2.4).
    """

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, kernel_size=3, padding=dilation,
                            dilation=dilation, groups=dim)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma.view(1, -1, 1, 1, 1) * self.dw(x)


# --------------------------------------------------------- C3 OSR (kernel surgery)
def widen_sr_conv(conv: nn.Conv3d, sr: Tuple[int, int, int]) -> nn.Conv3d:
    """Return an OVERLAPPING replacement for a stock sr conv Conv3d(C,C,k=r,s=r).

    New conv: kernel 2r, stride r, padding r/2 on every axis whose r is EVEN (axes with
    r == 1 are left untouched).  Output grid is unchanged:
        (n + 2*(r/2) - 2r)/r + 1 = n/r
    EXACT REDUCTION: the wide kernel is zero everywhere except the tap block at offsets
    [r/2, 3r/2), which receives the stock weights, so in exact arithmetic the output is
    identical.  Bitwise on CPU; on CUDA a different kernel size may select a different
    algorithm and reorder the summation, so the test asserts <= 1e-6 there (plan 2.3).

    INIT TRAP (also unit-tested): the stock init is normal_(0, sqrt(2/fan_out)) with
    fan_out = prod(kernel)*C_out; the core block must keep the STOCK fan-out or the
    initialisation distribution silently changes and the ablation becomes an init change.
    """
    C = conv.out_channels
    new_k, new_p, off = [], [], []
    for a in range(3):
        r = sr[a]
        if r > 1 and r % 2 == 0:
            new_k.append(2 * r); new_p.append(r // 2); off.append(r // 2)
        else:
            new_k.append(conv.kernel_size[a]); new_p.append(0); off.append(0)
    wide = nn.Conv3d(C, C, kernel_size=tuple(new_k), stride=conv.stride,
                     padding=tuple(new_p), bias=conv.bias is not None)
    with torch.no_grad():
        wide.weight.zero_()
        kd, kh, kw = conv.kernel_size
        wide.weight[:, :, off[0]:off[0] + kd, off[1]:off[1] + kh,
                    off[2]:off[2] + kw].copy_(conv.weight)
        if conv.bias is not None:
            wide.bias.copy_(conv.bias)
    return wide
