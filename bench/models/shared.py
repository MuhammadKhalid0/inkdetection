"""Class-A shared, model-external modules (applied identically to every model).

For A0 only ``DepthCollapse(mode="mean")`` + ``Head`` are active; the texture
stem is OFF.  ``TextureStem`` and the learned collapse are included so later
ablations (A0s onward) are pure config flags, not new code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureStem(nn.Module):
    """Enrich the single intensity channel with fixed high-pass + local-variance
    features plus a few learnable 3D filters.  OFF for A0."""

    def __init__(self, out_extra: int = 8, var_k: int = 5):
        super().__init__()
        k = torch.tensor(
            [[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
             [[0, -1, 0], [-1, 6, -1], [0, -1, 0]],
             [[0, 0, 0], [0, -1, 0], [0, 0, 0]]], dtype=torch.float32)
        self.register_buffer("hp", k.view(1, 1, 3, 3, 3))
        self.learn = nn.Conv3d(1, out_extra, kernel_size=3, padding=1)
        self.var_k = var_k
        self._out_ch = 3 + out_extra  # intensity + highpass + variance + learnable

    @property
    def out_ch(self) -> int:
        return self._out_ch

    def forward(self, x):  # x: [B,1,D,H,W]
        hp = F.conv3d(x, self.hp, padding=1)
        mean = F.avg_pool3d(x, self.var_k, 1, self.var_k // 2)
        var = F.avg_pool3d(x * x, self.var_k, 1, self.var_k // 2) - mean * mean
        var = var.clamp_min(0.0)
        learn = self.learn(x)
        return torch.cat([x, hp, var, learn], dim=1)


class DepthCollapse(nn.Module):
    """Collapse the depth axis of a feature volume [B,C,D,H,W] -> [B,C,H,W].

    mode: 'mean' | 'max' | 'learned' (learned absolute-depth softmax weighting)
          | 'surface_relative' (A2: learned weighting in the surface-relative frame
          r_d = d - s_hat, where s_hat is a per-tile soft surface centre).
    A0 uses 'mean'; A0s/A0b/A1 use 'learned'.
    """

    def __init__(self, depth: int, mode: str = "mean", channels: int = None,
                 per_pixel: bool = False, warp_stabilize: bool = False):
        super().__init__()
        self.mode = mode
        self.depth = depth
        self.per_pixel = per_pixel
        self.warp_stabilize = warp_stabilize
        # probe hook (2026-07-17, G2 s-hat fidelity diagnostic): when True, the
        # surface_relative branch stashes its s_hat estimate on the module.
        # Default False -> zero behaviour change for training/inference.
        self.probe = False
        self.last_s_hat = None
        # aux hook (2026-07-18, aux_shat closing experiment): when True, stash
        # s_hat WITH the autograd graph so an auxiliary supervision loss can
        # reach it. Default False -> zero behaviour change.
        self.aux = False
        self.aux_s_hat = None
        if mode == "learned":
            self.phi = nn.Parameter(torch.zeros(depth))
        elif mode == "surface_relative":
            if channels is None:
                raise ValueError("surface_relative collapse needs channels")
            # soft surface centre estimator (depth saliency), anchored at centre by zero-init
            self.surf = nn.Linear(channels, 1)
            # learnable weighting over the RELATIVE offset grid r in [-(D-1), +(D-1)],
            # Gaussian-init at r=0 so the surface-weighted readout starts centred.
            L = 2 * depth - 1
            grid = torch.arange(L, dtype=torch.float32) - (depth - 1)
            sigma0 = max(1.0, depth / 4.0)
            self.phi_rel = nn.Parameter(-0.5 * (grid / sigma0) ** 2)
            if per_pixel and warp_stabilize:
                # A2bs: learnable softmax temperature on the per-pixel s_hat(h,w).
                self.tau_pix = nn.Parameter(torch.ones(1))

    def forward(self, feat, s_hat=None):  # feat: [B,C,D,H,W]
        if self.mode == "mean":
            return feat.mean(dim=2)
        if self.mode == "max":
            return feat.amax(dim=2)
        if self.mode == "learned":
            D = feat.shape[2]
            if D != self.depth:
                raise ValueError(f"DepthCollapse expected D={self.depth}, got {D}")
            w = torch.softmax(self.phi, dim=0).view(1, 1, D, 1, 1)
            return (feat * w).sum(dim=2)
        if self.mode == "surface_relative":
            B, C, D, H, W = feat.shape
            if D != self.depth:
                raise ValueError(f"DepthCollapse expected D={self.depth}, got {D}")
            L = self.phi_rel.shape[0]
            idx = torch.arange(D, device=feat.device, dtype=feat.dtype)
            if self.per_pixel:
                # A2b: per-pixel surface centre s_hat(h,w) -> per-pixel relative collapse.
                # fp32 chain: fp16 saliency overflow under AMP -> NaN indices (t192 crash).
                sal = self.surf(feat.permute(0, 2, 3, 4, 1)).squeeze(-1).float()  # [B,D,H,W]
                if self.warp_stabilize:
                    sal = sal / self.tau_pix.float().clamp(min=0.1)  # A2bs: temp-stabilized
                wsurf = torch.softmax(sal, dim=1)                         # over depth D
                idxf = torch.arange(D, device=feat.device, dtype=torch.float32)
                s_hat = (wsurf * idxf.view(1, D, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,H,W]
                if self.probe:
                    self.last_s_hat = s_hat.detach().float().cpu()
                if self.aux:
                    self.aux_s_hat = s_hat
                pos = (idxf.view(1, D, 1, 1) - s_hat) + (D - 1)           # [B,D,H,W]
                pos = torch.nan_to_num(pos, nan=(L - 1) / 2.0).clamp(0, L - 1)
                lo = pos.floor().long()
                hi = (lo + 1).clamp(max=L - 1)
                frac = pos - lo.to(pos.dtype)
                phi = self.phi_rel
                logits = phi[lo] * (1 - frac) + phi[hi] * frac           # [B,D,H,W]
                w = torch.softmax(logits, dim=1).unsqueeze(1)            # [B,1,D,H,W]
                return (feat * w.to(feat.dtype)).sum(dim=2)
            prof = feat.mean(dim=(3, 4))                       # [B,C,D]
            sal = self.surf(prof.transpose(1, 2)).squeeze(-1).float()  # [B,D] fp32
            wsurf = torch.softmax(sal, dim=-1)
            idxf = torch.arange(D, device=feat.device, dtype=torch.float32)
            s_hat = (wsurf * idxf).sum(dim=-1, keepdim=True)   # [B,1] anchored at centre
            if self.probe:
                self.last_s_hat = s_hat.detach().float().cpu()
            if self.aux:
                self.aux_s_hat = s_hat
            # sample phi_rel at relative positions r_d = d - s_hat (linear interp)
            pos = (idxf.unsqueeze(0) - s_hat) + (D - 1)        # [B,D] in [0, 2D-2]
            pos = torch.nan_to_num(pos, nan=(L - 1) / 2.0).clamp(0, L - 1)
            lo = pos.floor().long()
            hi = (lo + 1).clamp(max=L - 1)
            frac = pos - lo.to(pos.dtype)
            phi = self.phi_rel
            logits = phi[lo] * (1 - frac) + phi[hi] * frac     # [B,D]
            w = torch.softmax(logits, dim=-1).view(B, 1, D, 1, 1)
            return (feat * w.to(feat.dtype)).sum(dim=2)
        raise ValueError(f"unknown collapse mode {self.mode}")


class Head(nn.Module):
    """3x3 -> 1x1 conv head producing logits (after depth collapse).

    ``out_ch=1`` (default, all pre-2026-07-17 runs): binary sigmoid logits.
    ``out_ch=3`` (u2_soft3 variant): 3-class softmax logits (ink/bg/uncertain).
    """

    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 1),
        )

    def forward(self, x):  # [B,C,H,W] -> [B,out_ch,H,W]
        return self.net(x)
