"""Input-space Information-Bottleneck attribution (IBA), adapted from Inkalyzer.

The bottleneck lives in input space: a per-depth-layer gate g in [0,1] blends the
real volume with a depth-collapsed `noise` baseline,
    x_tilde = g * data + (1 - g) * noise,
and g is optimised to keep the model's prediction unchanged (loss1) while passing
as little information as possible (loss2 = g.mean(), weighted by beta).  The
converged gate profile over depth = which depth layers the model relies on.

Upstream (younader/Inkalyzer) hardcodes 30 depth layers and fp16 autocast; here the
depth `D` and the autocast dtype are parameters so this works with our 32-layer
UNETR++ ink models (and bf16 for the t192 warp chains).

model: forward(x:[1,1,D,H,W]) -> logits:[1,1,H,W]  (our UnifiedModel interface).
loss_func(y_preds_logits, y_target_prob) -> scalar.
"""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from .distributions import BinaryConcrete, RectifiedStreched


def _autocast(device, dtype):
    if device == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def get_noise(data, noise_type):
    """Depth-collapsed baseline from a [1,1,D,H,W] tensor -> [1,1,H,W] (caller repeats over D).

    data may be a torch.Tensor or np.ndarray of shape (1,1,D,size,size).
    """
    is_t = torch.is_tensor(data)
    arr = data.detach().cpu().numpy() if is_t else np.asarray(data)
    if noise_type == "top":
        out = arr[:, :, 0]
    elif noise_type == "bottom":
        out = arr[:, :, -1]
    elif noise_type == "median":
        out = np.median(arr, axis=2)
    elif noise_type == "mean":
        out = np.mean(arr, axis=2)
    elif noise_type == "top85":
        out = np.percentile(arr, 85, axis=2)
    elif noise_type == "max":
        out = np.max(arr, axis=2)
    else:
        raise ValueError("noise_type in {top,bottom,median,mean,top85,max}")
    out = out[:, :, None]  # (1,1,1,H,W)
    return torch.from_numpy(out).to(data.device).to(data.dtype) if is_t else out


def attribute_z(data, noise, model, loss_func, n_steps=100, lr=1e-1, beta=0.5,
                size=64, depth=32, autocast_dtype=torch.bfloat16):
    """Base IBA along the depth (z) axis.

    Returns (gates[1,1,depth,1,1] in [0,1] as np, alpha_history list, final_pred cpu tensor).
    """
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    alpha = torch.full((1, 1, depth, 1, 1), 5.0, requires_grad=True, device=data.device)
    alpha_history = [alpha.clone().detach()]
    optimizer = torch.optim.Adam([alpha], lr=lr)
    with _autocast(device, autocast_dtype):
        y = torch.sigmoid(model(data).detach())
        for _ in range(n_steps):
            optimizer.zero_grad()
            gates = alpha.sigmoid().repeat(1, 1, 1, size, size)
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss = loss_func(y_preds, y) + beta * gates.mean()
            loss.backward()
            optimizer.step()
            alpha_history.append(alpha.clone().detach())
        gates = alpha.sigmoid().repeat(1, 1, 1, size, size)
        final_pred = model(gates * data + (1 - gates) * noise)
    return alpha.sigmoid().detach().cpu().numpy(), alpha_history, final_pred.detach().float().cpu()


def attribute_z_lowres(data, noise, model, loss_func, n_steps=100, lr=1e-1, beta=0.5,
                       size=64, depth=32, autocast_dtype=torch.bfloat16):
    """Downsampled (depth//2) IBA, upsampled x2 -> smoother depth profile. depth must be even."""
    assert depth % 2 == 0, "attribute_z_lowres needs even depth"
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    alpha = torch.full((1, 1, depth // 2, 1, 1), 5.0, requires_grad=True, device=data.device)
    alpha_history = [alpha.clone().detach()]
    optimizer = torch.optim.Adam([alpha], lr=lr)

    def up(a):
        return F.interpolate(a.sigmoid().squeeze(-1).squeeze(-1), scale_factor=2,
                             mode="linear").unsqueeze(-1).unsqueeze(-1)

    with _autocast(device, autocast_dtype):
        y = torch.sigmoid(model(data).detach())
        for _ in range(n_steps):
            optimizer.zero_grad()
            gates = up(alpha).repeat(1, 1, 1, size, size)
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss = loss_func(y_preds, y) + beta * gates.mean()
            loss.backward()
            optimizer.step()
            alpha_history.append(alpha.clone().detach())
        gates = up(alpha).repeat(1, 1, 1, size, size)
        final_pred = model(gates * data + (1 - gates) * noise)
    return up(alpha)[:, :, :, 0, 0].detach().cpu().numpy(), alpha_history, final_pred.detach().float().cpu()


def attribute_3d(data, noise, model, loss_func, pred_label=None, n_steps=20, lr=2e-1,
                 beta=0.05, size=64, window_size=16, depth=32, autocast_dtype=torch.bfloat16):
    """z-attribution first, then per-(x,y)-window attribution -> [1,1,depth,size,size] saliency."""
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    alpha_prelim, _, _ = attribute_z(data, noise, model, loss_func, n_steps=n_steps,
                                     lr=lr, beta=beta, size=size, depth=depth,
                                     autocast_dtype=autocast_dtype)
    alpha = (torch.tensor(alpha_prelim)
             .repeat(1, 1, 1, size // window_size, size // window_size)
             .clone()).to(data.device)
    if pred_label is not None:
        alpha = alpha * pred_label.to(data.device)
    alpha.requires_grad = True
    optimizer = torch.optim.RMSprop([alpha], lr=2e-2, centered=True)

    def up(a):
        return F.interpolate(a.sigmoid().squeeze(0), scale_factor=window_size,
                             mode="bilinear").unsqueeze(0)

    with _autocast(device, autocast_dtype):
        y = torch.sigmoid(model(data).detach())
        for _ in range(n_steps):
            optimizer.zero_grad()
            gates = up(alpha)
            model_input = gates * data + (1 - gates) * noise
            y_preds = model(model_input)
            loss = loss_func(y_preds, y) + beta * gates.mean()
            loss.backward()
            optimizer.step()
        gates = up(alpha)
        final_pred = model(gates * data + (1 - gates) * noise)
    return up(alpha).detach().float().cpu().numpy(), final_pred.detach().float().cpu().numpy()
