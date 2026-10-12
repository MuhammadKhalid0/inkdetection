"""Windowed 3D attention for nnFormer-Ink.

Two modes, mirroring the original:
  - WindowAttention3D            : self-attention (LV-MSA / GV-MSA blocks)
  - WindowAttention3D(kv=True)   : skip attention — Q from the decoder (x_up),
                                   K/V from the encoder skip. Faithful to the
                                   original: Q gets NO linear projection.

Relative-position bias uses the corrected anisotropic index (windows.py).
SR-PB (surface-relative additive bias, plan §2.1) will plug in here in a later
port step; the constructor already reserves the flag and asserts it off.
"""
import torch
import torch.nn as nn

from .windows import relative_position_index

try:  # trunc_normal_ lives in timm (available in the training venv)
    from timm.models.layers import trunc_normal_
except Exception:  # pragma: no cover - fallback keeps CPU tooling importable
    def trunc_normal_(t, std=0.02):
        nn.init.normal_(t, std=std)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window, num_heads, kv=False,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0, sr_pb=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by heads {num_heads}"
        self.dim = dim
        self.window = tuple(window)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.kv_mode = kv

        wd, wh, ww = self.window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        self.register_buffer("rel_index", relative_position_index(self.window), persistent=False)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        if kv:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Q = x_up, unprojected (original)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _bias(self, N):
        bias = self.relative_position_bias_table[self.rel_index.view(-1)]
        return bias.view(N, N, -1).permute(2, 0, 1).contiguous()  # [heads, N, N]

    def forward(self, x, mask=None, x_up=None, key_bias=None, coarse=None):
        """x: [B_, N, C] windows. Self mode: attend within x.
        KV mode: x = encoder skip (K/V source), x_up = decoder query.
        key_bias: [B_, N, heads] additive per-key logit bias (SR-PB, plan §2.1).
        coarse:   (k_c, v_c, gamma) from CrossScaleKV (CSSA, plan §2.2); the
                  coarse branch carries its OWN softmax and a zero-init gate, so
                  gamma=0 reduces this call exactly to stock attention."""
        B_, N, C = x.shape
        h = self.num_heads
        if self.kv_mode:
            assert x_up is not None
            kv = self.kv(x).reshape(B_, N, 2, h, C // h).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]
            q = x_up.reshape(B_, N, h, C // h).permute(0, 2, 1, 3).contiguous()
        else:
            qkv = self.qkv(x).reshape(B_, N, 3, h, C // h).permute(2, 0, 3, 1, 4).contiguous()
            q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # [B_, h, N, N]
        attn = attn + self._bias(N).unsqueeze(0)
        if key_bias is not None:                     # SR-PB: broadcast over queries
            attn = attn + key_bias.permute(0, 2, 1).unsqueeze(2)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, h, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, h, N, N)
        attn = self.attn_drop(self.softmax(attn))
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()

        if coarse is not None:                       # CSSA: gated cross-scale branch
            k_c, v_c, gamma = coarse                 # [B, M, h, hd] x2, [M, h]
            B = k_c.shape[0]
            nW = B_ // B
            # every window of a sample sees that sample's coarse tokens
            k_c = k_c.repeat_interleave(nW, dim=0).permute(0, 2, 1, 3).contiguous()
            v_c = v_c.repeat_interleave(nW, dim=0).permute(0, 2, 1, 3).contiguous()
            attn_c = self.softmax(q @ k_c.transpose(-2, -1).contiguous())     # own softmax
            out_c = attn_c @ v_c                                             # [B_, h, N, hd]
            g = gamma.mean(0).view(1, h, 1, 1)       # per-head gate (zero-init)
            out = out + (g * out_c).transpose(1, 2).reshape(B_, N, C).contiguous()

        return self.proj_drop(self.proj(out))
