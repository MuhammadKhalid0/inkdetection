"""Sliding-window inference over a full processed segment with Gaussian blending.

Returns a full-resolution probability map for the segment.  Used for validation
(F0.5 on the reconstructed segment) and for qualitative test prediction.
"""
import os

import numpy as np
import torch

from data.dataset import normalize_volume


def _gaussian_weight(tile, sigma_scale=1.0 / 6):
    ax = np.arange(tile) - (tile - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    sigma = tile * sigma_scale
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (g / g.max()).astype(np.float32)


def _tile_starts(size, tile, stride):
    if size < tile:
        return []
    s = list(range(0, size - tile + 1, stride))
    if s[-1] != size - tile:
        s.append(size - tile)
    return s


def _dihedral(x, k, flip):
    """Apply rot90^k then optional hflip on the last two axes (square tiles)."""
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if flip:
        x = torch.flip(x, dims=(-1,))
    return x


def _dihedral_inv(x, k, flip):
    if flip:
        x = torch.flip(x, dims=(-1,))
    if k:
        x = torch.rot90(x, -k, dims=(-2, -1))
    return x


@torch.no_grad()
def infer_segment(model, seg, cfg, device, proc_dir_fn, load_labels=True, also_unc=False):
    d = cfg["data"]
    tile = int(d["tile"])
    depth = int(d["depth"])
    center = int(d.get("center_layer", 32))
    n_layers = int(d.get("n_layers", 65))
    # center_mode 'per_segment' (2026-07-17, G4): centre the depth window on this
    # segment's own intensity depth peak (meta.json), clamped to [24,40]. Default
    # 'fixed' -> historical behaviour. Missing meta (test segment) -> fixed centre.
    if d.get("center_mode", "fixed") == "per_segment":
        import json as _json
        meta_path = os.path.join(proc_dir_fn(seg), "meta.json")
        if os.path.exists(meta_path):
            meta = _json.load(open(meta_path))
            if "depth_peak" in meta:
                center = int(np.clip(int(meta["depth_peak"]), 24, 40))
        else:
            print(f"[infer] center_mode=per_segment but no meta.json for {seg}; "
                  f"falling back to fixed center {center}")
    norm = float(d.get("normalize", 255.0))
    norm_mode = d.get("norm_mode", "global")
    stride = int(cfg.get("infer", {}).get("stride", 32))
    bs = int(cfg.get("infer", {}).get("batch", 32))

    proc = proc_dir_fn(seg)
    vol = np.load(os.path.join(proc, "vol.npy"), mmap_mode="r")     # [L,H,W]
    mask = np.load(os.path.join(proc, "mask.npy"))                  # [H,W]
    label_file = d.get("label_file", "ink.npy")
    gt_path = os.path.join(proc, label_file)
    if load_labels and not os.path.exists(gt_path):
        raise FileNotFoundError(f"{gt_path} missing — run scripts/register_labels.py "
                                f"or set data.label_file back to ink.npy")
    gt = np.load(gt_path) if load_labels else None
    H, W = mask.shape

    z0 = max(0, min(center - depth // 2, n_layers - depth))
    z1 = z0 + depth
    gw = _gaussian_weight(tile)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)
    unc_sum = np.zeros((H, W), dtype=np.float32) if also_unc else None

    coords = [(y, x) for y in _tile_starts(H, tile, stride)
              for x in _tile_starts(W, tile, stride)
              if mask[y:y + tile, x:x + tile].mean() > 0.0]

    model.eval()
    amp = cfg.get("train", {}).get("amp", True)
    amp_dtype = (torch.bfloat16 if cfg.get("train", {}).get("amp_dtype", "fp16") == "bf16"
                 else torch.float16)
    # tta (2026-07-17, plan §4): 8-fold dihedral test-time augmentation on the
    # in-plane axes (depth untouched); probabilities are inverse-transformed and
    # averaged. Default off -> byte-identical prior behaviour.
    tta = bool(cfg.get("infer", {}).get("tta", False))
    tta_ops = ([(k, f) for k in range(4) for f in (False, True)] if tta
               else [(0, False)])

    def _probs(logits):
        """logits [b,c,t,t] -> (prob_ink [b,t,t], prob_unc [b,t,t] or None)."""
        if logits.shape[1] == 3:                       # u2_soft3: softmax head
            sm = torch.softmax(logits.float(), 1)
            return sm[:, 0], (sm[:, 2] if also_unc else None)
        return torch.sigmoid(logits.float()).squeeze(1), None

    # --- SegFormer3D C1/PDK halo carrier (2026-09-10). DEFAULT OFF -> byte-identical
    # behaviour for every pre-existing run. The halo is a DETERMINISTIC function of the
    # tile coordinate at inference exactly as in training, so there is no train/deployment
    # distribution mismatch (the mechanism that killed word-box sampling).
    _halo_cfg = d.get("halo", {}) or {}
    halo_on = bool(_halo_cfg.get("enabled", False))
    halo_f = int(_halo_cfg.get("factor", 2))
    halo_mode = _halo_cfg.get("mode", "peripheral")

    def _halo_for(y, x):
        big = tile * halo_f
        y0, x0 = y + tile // 2 - big // 2, x + tile // 2 - big // 2
        arr = np.zeros((z1 - z0, big, big), dtype=np.float32)
        ys, xs_, ye, xe = max(0, y0), max(0, x0), min(H, y0 + big), min(W, x0 + big)
        if ye > ys and xe > xs_:
            arr[:, ys - y0:ye - y0, xs_ - x0:xe - x0] = np.asarray(
                vol[z0:z1, ys:ye, xs_:xe], dtype=np.float32)
        arr = normalize_volume(arr, norm_mode, norm)
        dd = arr.shape[0]
        return arr.reshape(dd, tile, halo_f, tile, halo_f).mean(axis=(2, 4))

    for i in range(0, len(coords), bs):
        batch = coords[i:i + bs]
        xs = np.stack([normalize_volume(np.asarray(vol[z0:z1, y:y + tile, x:x + tile], dtype=np.float32), norm_mode, norm)
                       for (y, x) in batch])            # [b,D,t,t]
        xb = torch.from_numpy(xs).unsqueeze(1).to(device)  # [b,1,D,t,t]
        if halo_on:
            if halo_mode == "self":
                dd = xs.shape[1]
                hs = xs.reshape(len(batch), dd, tile // halo_f, halo_f,
                                tile // halo_f, halo_f).mean(axis=(3, 5))
                hs = np.repeat(np.repeat(hs, halo_f, axis=2), halo_f, axis=3)
            else:
                hs = np.stack([_halo_for(y, x) for (y, x) in batch])
            hb = torch.from_numpy(np.ascontiguousarray(hs, dtype=np.float32)).unsqueeze(1)
            xb = torch.cat([xb, hb.to(device)], dim=1)     # [b,2,D,t,t]
        pb_acc, ub_acc = None, None
        for k, f in tta_ops:
            xt = _dihedral(xb, k, f)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                logits = model(xt)                       # [b,c,t,t]
            p, u = _probs(logits)
            p = _dihedral_inv(p, k, f)
            pb_acc = p if pb_acc is None else pb_acc + p
            if u is not None:
                u = _dihedral_inv(u, k, f)
                ub_acc = u if ub_acc is None else ub_acc + u
        pb = (pb_acc / len(tta_ops)).cpu().numpy()       # [b,t,t]
        ub = (ub_acc / len(tta_ops)).cpu().numpy() if ub_acc is not None else None
        for j, (y, x) in enumerate(batch):
            prob_sum[y:y + tile, x:x + tile] += pb[j] * gw
            w_sum[y:y + tile, x:x + tile] += gw
            if ub is not None:
                unc_sum[y:y + tile, x:x + tile] += ub[j] * gw

    prob = prob_sum / np.maximum(w_sum, 1e-6)
    prob = prob * (mask > 0)
    if also_unc:
        unc = unc_sum / np.maximum(w_sum, 1e-6)
        unc = unc * (mask > 0)
        return prob, mask, gt, unc
    return prob, mask, gt
