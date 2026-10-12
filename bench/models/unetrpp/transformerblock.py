import torch.nn as nn
import torch
from .dynunet_block import UnetResBlock


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
            pe_mode: str = "3d",
            pe_depth: int = 1,
            depth_attn: bool = False,
            da_gamma_init: float = 0.1,
            surface_relative: bool = False,
            per_pixel: bool = False,
            warp_stabilize: bool = False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.
            pe_mode: "3d" (default) = full learnable PE over all H*W*D tokens
                     (published behaviour). "2d" = spatial-only PE over H*W tokens,
                     broadcast across depth (A0b: depth-preserving embed keeps real
                     depth tokens, so absolute-depth PE is dropped to stay coherent
                     with the later surface-relative frame).
            pe_depth: depth (number of z-slices) of this stage's token volume; only
                     used when pe_mode == "2d" to size/broadcast the spatial PE.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.pe_mode = pe_mode
        self.pe_depth = pe_depth
        self.depth_attn = depth_attn
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate,
                             depth_attn=depth_attn, depth_size=pe_depth, da_gamma_init=da_gamma_init, surface_relative=surface_relative,
                             per_pixel=per_pixel, warp_stabilize=warp_stabilize)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            if pe_mode == "2d":
                if input_size % pe_depth != 0:
                    raise ValueError(
                        f"pe_mode='2d' needs input_size ({input_size}) divisible by "
                        f"pe_depth ({pe_depth})."
                    )
                spatial_size = input_size // pe_depth
                self.pos_embed = nn.Parameter(torch.zeros(1, spatial_size, hidden_size))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            if self.pe_mode == "2d":
                # Volume axes are (depth, y, x); the block names them (H, W, D), so the
                # anatomical depth is the OUTER token axis H and the within-layer plane
                # is (W, D).  Tokens flatten H-outer .. D-inner (t = h*(W*D) + w*D + d),
                # so the spatial PE (1, W*D, C) is TILED across the depth (H) axis.
                pe = self.pos_embed.repeat(1, H, 1)     # (1, H*W*D, C)
                x = x + pe
            else:
                x = x + self.pos_embed
        dims = (H, W, D) if self.depth_attn else None
        attn = x + self.gamma * self.epa_block(self.norm(x), dims)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1,
                 depth_attn=False, depth_size=1, da_gamma_init=0.1, surface_relative=False,
                 per_pixel=False, warp_stabilize=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

        # --- DA-EPA depth-attention branch (A1) -------------------------------
        # Adds a depth-axis gate over the shared EPA output.  With shared Q/K and
        # a per-channel gate ``gamma_da`` init 0.1; at gamma_da=0 the multiplier is
        # 1 -> DA-EPA == EPA (graceful degradation, clean ablation baseline).
        self.depth_attn = depth_attn
        self.surface_relative = surface_relative
        self.per_pixel = per_pixel
        self.warp_stabilize = warp_stabilize
        if depth_attn:
            self.Wvd = nn.Linear(hidden_size, hidden_size)          # depth value
            self.proj_d = nn.Linear(2 * hidden_size, hidden_size)   # mean+std desc -> C
            self.gamma_da = nn.Parameter(da_gamma_init * torch.ones(hidden_size))
            if surface_relative:
                # A2: surface-relative depth PE. Soft surface centre s_hat is estimated
                # per tile from the depth profile (anchored at centre by zero-init), and
                # the depth PE is a small MLP of the RELATIVE offset r_d = d - s_hat.
                self.surf_g = nn.Linear(hidden_size, 1)             # depth saliency -> s_hat
                self.pe_mlp = nn.Sequential(
                    nn.Linear(1, hidden_size), nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                if per_pixel:
                    # A2b: per-pixel surface field s_hat(w,d).  A small head predicts a
                    # surface depth for EACH within-layer pixel; the canonical depth gate
                    # is then warped (deformable z-sampling) along the depth axis by each
                    # pixel's deviation from the tile centre.  Zero-init -> starts as A2.
                    self.surf_pix = nn.Linear(hidden_size, 1)
                    nn.init.zeros_(self.surf_pix.weight)
                    nn.init.zeros_(self.surf_pix.bias)
                    if warp_stabilize:
                        # A2bs: stabilize the per-pixel warp (methodology risk fixes):
                        #  * learnable softmax temperature on the s_pix soft-argmax
                        #  * bounded shift (|shift| < H/4 via tanh) so the warp cannot
                        #    run away and destabilize from-scratch training.
                        self.tau_pix = nn.Parameter(torch.ones(1))
            else:
                # A1: absolute learnable depth PE.
                self.pe_abs = nn.Parameter(torch.zeros(1, depth_size, hidden_size))

    @staticmethod
    def _pool_meanstd(t, H, W, D):
        # t: [B, N=H*W*D, C] with H (=depth) outer -> per-depth-slice mean+std over
        # the within-layer plane (W, D).  Returns [B, H, 2C].
        B, N, C = t.shape
        t = t.view(B, H, W * D, C)
        mean = t.mean(dim=2)
        std = t.std(dim=2, unbiased=False)
        return torch.cat([mean, std], dim=-1)

    def _depth_pe(self, base_desc, H):
        # base_desc: [B,H,C] content descriptor; H = depth-slice count (Dz).
        # Returns (pe, s_hat).  For A1 (absolute) pe is a learnable table; for A2
        # (surface-relative) s_hat is a soft surface centre and pe = MLP(d - s_hat).
        if not self.surface_relative:
            return self.pe_abs, None
        B = base_desc.shape[0]
        # fp32: under AMP the fp16 saliency can overflow -> NaN s_hat -> NaN warp
        # indices downstream (t192 crash, IndexKernel out-of-bounds at e13).
        idx = torch.arange(H, device=base_desc.device, dtype=torch.float32)  # [H]
        sal = self.surf_g(base_desc).squeeze(-1).float()        # [B,H]
        w = torch.softmax(sal, dim=-1)
        s_hat = (w * idx).sum(dim=-1, keepdim=True)             # [B,1] anchored at centre (zero-init)
        r = (idx.unsqueeze(0) - s_hat).unsqueeze(-1)            # [B,H,1] relative offset
        pe = self.pe_mlp(r.to(base_desc.dtype))                 # [B,H,C]
        return pe, s_hat

    def _depth_gate(self, x_in, q_tok, k_tok, dims):
        # Returns gate g in (0,1)^{[B,N,C]} broadcast from the [B,Dz,C] depth verdict.
        H, W, D = dims
        B, N, C = x_in.shape
        q_desc = self.proj_d(self._pool_meanstd(q_tok, H, W, D))   # [B,H,C]
        k_desc = self.proj_d(self._pool_meanstd(k_tok, H, W, D))
        Vd = self.proj_d(self._pool_meanstd(self.Wvd(x_in), H, W, D))
        pe, s_hat = self._depth_pe(q_desc, H)
        Qd = q_desc + pe
        Kd = k_desc + pe
        Ad = torch.softmax((Qd @ Kd.transpose(-1, -2)) / (C ** 0.5), dim=-1)  # [B,H,H]
        verdict = Ad @ Vd                                                    # [B,H,C] canonical
        if self.per_pixel and self.surface_relative:
            # A2b: warp the canonical depth verdict per within-layer pixel by that
            # pixel's surface deviation (soft-argmax of per-pixel depth saliency).
            xv = x_in.view(B, H, W * D, C)
            # fp32 chain: fp16 sal (esp. after /tau_pix, x10) overflowed at tile192
            # -> softmax NaN -> NaN pos -> floor().long() garbage -> device assert.
            sal = self.surf_pix(xv).squeeze(-1).float()            # [B,H,WD]
            if self.warp_stabilize:
                sal = sal / self.tau_pix.float().clamp(min=0.1)   # A2bs: temp-stabilized
            wpix = torch.softmax(sal, dim=1)                       # over depth (H)
            idx = torch.arange(H, device=x_in.device, dtype=torch.float32)
            s_pix = (wpix * idx.view(1, H, 1)).sum(dim=1)          # [B,WD] per-pixel surface
            shift = s_pix - s_hat.float()                          # [B,WD] deviation vs tile centre
            if self.warp_stabilize:
                m = max(1.0, H / 4.0)                              # A2bs: bound |shift| < H/4
                shift = m * torch.tanh(shift / m)
            pos = idx.view(1, H, 1) - shift.unsqueeze(1)          # [B,H,WD]
            pos = torch.nan_to_num(pos, nan=(H - 1) / 2.0).clamp(0, H - 1)
            lo = pos.floor().long()
            hi = (lo + 1).clamp(max=H - 1)
            frac = (pos - lo.to(pos.dtype)).unsqueeze(-1)        # [B,H,WD,1]
            vexp = verdict.unsqueeze(2).expand(B, H, W * D, C)   # [B,H,WD,C]
            v_lo = torch.gather(vexp, 1, lo.unsqueeze(-1).expand(B, H, W * D, C))
            v_hi = torch.gather(vexp, 1, hi.unsqueeze(-1).expand(B, H, W * D, C))
            g = (v_lo * (1 - frac) + v_hi * frac).reshape(B, N, C).to(verdict.dtype)
        else:
            g = verdict.unsqueeze(2).expand(B, H, W * D, C).reshape(B, N, C)
        return torch.sigmoid(g)

    def forward(self, x, dims=None):
        B, N, C = x.shape

        qkvv0 = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        if self.depth_attn and dims is not None:
            x_in = x                                   # EPA input, for depth value Wvd
            q_tok = qkvv0[:, :, 0].reshape(B, N, C)
            k_tok = qkvv0[:, :, 1].reshape(B, N, C)

        qkvv = qkvv0.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)

        # DA-EPA gate (A1): amplify ink-bearing depths, suppress the rest.
        if self.depth_attn and dims is not None:
            g = self._depth_gate(x_in, q_tok, k_tok, dims)   # gate from EPA input + shared Q/K
            x = x * (1 + self.gamma_da * (2 * g - 1))
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
