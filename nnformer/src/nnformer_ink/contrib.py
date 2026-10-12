"""The two nnFormer-native contributions (nnformer_plan.md §2.1-2.2).

Both obey the project's house design rule: **exact no-op at initialization**
behind a default-off flag, so every ablation is a genuine single toggle and the
worst case is "no improvement", never "broken model".

C1 SR-PB (SurfaceDepthBias)
    An additive, per-token, surface-relative depth bias on the attention logits:
    each KEY contributes b(z - s_hat), a learned table over binned
    surface-relative depth, shared per stage/head.  s_hat is a soft-argmax over
    a mean+std depth descriptor (std because the ink cue is textural energy, not
    mean brightness -- the DA-EPA lesson).
    NOTE the plan's §2.1 correction: re-anchoring the *pairwise* relative-position
    index on s_hat is a NO-OP (dz = z_i - z_j is shift-invariant, so a common
    s_hat cancels exactly).  Hence the key-side additive form, which does not
    cancel.  Zero-init estimator -> s_hat = window centre; zero-init table ->
    b == 0 -> bitwise-identical to stock attention.

C2 CSSA (CrossScaleKV)
    Decoder queries additionally attend into the COARSE encoder stages (letterform
    -scale context) alongside their own-resolution skip.  Formulated as a gated
    cross-attention with its own softmax:
        out = softmax(q k_l^T) v_l  +  gamma * softmax(q k_c^T) v_c
    with gamma zero-init per head -> at step 0 CSSA is EXACTLY stock skip
    attention.  Coarse stages hold few tokens (S4 = 144 at t192, 256 at t256), so
    the added cost is small -- the point being letterform context at a fraction of
    the cost of enlarging the tile.
"""
import torch
import torch.nn as nn


class SurfaceDepthBias(nn.Module):
    """C1 / SR-PB.  Returns a per-(batch, depth-slice, head) additive logit bias."""

    def __init__(self, dim: int, depth: int, num_heads: int):
        super().__init__()
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        # surface estimator over a mean+std depth descriptor; zero-init => the
        # softmax is uniform => s_hat = (D-1)/2 = the window centre.
        self.est = nn.Linear(2 * dim, 1)
        nn.init.zeros_(self.est.weight)
        nn.init.zeros_(self.est.bias)
        # learned bias table over relative depth offsets r in [-(D-1), +(D-1)]
        self.table = nn.Parameter(torch.zeros(2 * self.depth - 1, self.num_heads))
        self.register_buffer("_idx", torch.arange(self.depth, dtype=torch.float32),
                             persistent=False)

    def forward(self, x):  # x: [B, D, H, W, C] -> [B, D, heads]
        B, D, H, W, C = x.shape
        assert D == self.depth, f"SR-PB configured for depth {self.depth}, got {D}"
        xf = x.float()
        desc = torch.cat([xf.mean(dim=(2, 3)), xf.std(dim=(2, 3))], dim=-1)  # [B,D,2C]
        logits = self.est(desc).squeeze(-1)                                   # [B,D]
        w = torch.softmax(logits, dim=1)
        s_hat = (w * self._idx).sum(dim=1, keepdim=True)                      # [B,1]
        # relative depth of every slice w.r.t. the estimated surface
        pos = (self._idx.unsqueeze(0) - s_hat) + (self.depth - 1)             # [B,D] in [0,2D-2]
        pos = pos.clamp(0, 2 * self.depth - 2)
        lo = pos.floor()
        hi = (lo + 1).clamp(max=2 * self.depth - 2)
        frac = (pos - lo).unsqueeze(-1)
        tl = self.table[lo.long()]                                            # [B,D,heads]
        th = self.table[hi.long()]
        bias = (1.0 - frac) * tl + frac * th
        return torch.nan_to_num(bias).to(x.dtype)

    @torch.no_grad()
    def probe_s_hat(self, x):
        """Diagnostic only (G2-style fidelity probe): the estimated surface index."""
        xf = x.float()
        desc = torch.cat([xf.mean(dim=(2, 3)), xf.std(dim=(2, 3))], dim=-1)
        w = torch.softmax(self.est(desc).squeeze(-1), dim=1)
        return (w * self._idx).sum(dim=1)


class CrossScaleKV(nn.Module):
    """C2 / CSSA.  Projects coarse encoder tokens to this stage's width and
    produces gated K/V.  gamma is zero-init per head => exact stock reduction."""

    def __init__(self, dim: int, coarse_dims, num_heads: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.proj = nn.ModuleList([nn.Linear(int(cd), dim * 2) for cd in coarse_dims])
        self.norm = nn.ModuleList([nn.LayerNorm(int(cd)) for cd in coarse_dims])
        # per-head gate, zero-init: the trained value is a reportable quantity
        # (the gamma_da lesson -- a gate that stays at init is how "nothing to
        # add" shows itself, so it is plotted either way).
        self.gamma = nn.Parameter(torch.zeros(len(coarse_dims), self.num_heads))

    def forward(self, coarse_tokens):
        """coarse_tokens: list of [B, M_i, C_i] -> (k, v, gamma)
        k, v: [B, sum(M_i), heads, head_dim]; gamma: [sum(M_i), heads]"""
        ks, vs, gs = [], [], []
        for i, t in enumerate(coarse_tokens):
            B, M, _ = t.shape
            kv = self.proj[i](self.norm[i](t))            # [B, M, 2*dim]
            dim = kv.shape[-1] // 2
            k, v = kv[..., :dim], kv[..., dim:]
            hd = dim // self.num_heads
            ks.append(k.view(B, M, self.num_heads, hd))
            vs.append(v.view(B, M, self.num_heads, hd))
            gs.append(self.gamma[i].unsqueeze(0).expand(M, -1))
        return torch.cat(ks, 1), torch.cat(vs, 1), torch.cat(gs, 0)


class SpectralTransform3D(nn.Module):
    """ST-3D of vFFC (Quattrini et al. 2023, arXiv:2308.05070), plan §2.4.

    rFFT3 -> stack(Re,Im) on channels -> 1x1x1 conv + BN + ReLU -> irFFT3.
    Runs as an **fp32 island**: torch.fft does not support bf16, so the block
    casts in/out around the transform and nan_to_num's on exit (plan §2.4 port
    note / §3.4.5 checklist item).
    """

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch * 2, ch * 2, 1, bias=False)
        self.bn = nn.BatchNorm3d(ch * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            f = torch.fft.rfftn(xf, dim=(-3, -2, -1), norm="ortho")
            y = torch.cat([f.real, f.imag], dim=1)
            y = self.act(self.bn(self.conv(y)))
            re, im = torch.chunk(y, 2, dim=1)
            out = torch.fft.irfftn(torch.complex(re, im), s=xf.shape[-3:],
                                   dim=(-3, -2, -1), norm="ortho")
            out = torch.nan_to_num(out)
        return out.to(dtype)


class VFFC3D(nn.Module):
    """Volumetric Fast Fourier Convolution layer: channels split into a LOCAL
    branch (two 3x3x3 convs) and a GLOBAL branch (3x3x3 conv + ST-3D); the two
    are cross-summed, normed and re-stacked (faithful to the paper's Fig. 2)."""

    def __init__(self, ch, ratio=0.5):
        super().__init__()
        self.cg = max(1, int(ch * ratio))     # global channels
        self.cl = ch - self.cg                # local channels
        self.local1 = nn.Conv3d(self.cl, self.cl, 3, padding=1, bias=False)
        self.local2 = nn.Conv3d(self.cl, self.cl, 3, padding=1, bias=False)
        self.l2g = nn.Conv3d(self.cl, self.cg, 3, padding=1, bias=False)
        self.g_conv = nn.Conv3d(self.cg, self.cg, 3, padding=1, bias=False)
        self.g2l = nn.Conv3d(self.cg, self.cl, 3, padding=1, bias=False)
        self.st = SpectralTransform3D(self.cg)
        self.bn_l, self.bn_g = nn.BatchNorm3d(self.cl), nn.BatchNorm3d(self.cg)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        xl, xg = torch.split(x, [self.cl, self.cg], dim=1)
        yl = self.local2(self.local1(xl)) + self.g2l(xg)        # local + global->local
        yg = self.st(self.g_conv(xg)) + self.l2g(xl)            # spectral + local->global
        return torch.cat([self.act(self.bn_l(yl)), self.act(self.bn_g(yg))], dim=1)


class VFFCResBlock(nn.Module):
    """Two vFFC layers + residual (the paper's bottleneck unit)."""

    def __init__(self, ch, ratio=0.5):
        super().__init__()
        self.f1, self.f2 = VFFC3D(ch, ratio), VFFC3D(ch, ratio)

    def forward(self, x):
        return x + self.f2(self.f1(x))
