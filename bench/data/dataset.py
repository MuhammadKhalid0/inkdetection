"""Torch dataset + ink-aware sampler for tile training.

Reads (segment, y, x) rows from a manifest, memmaps each segment's processed
arrays, and returns:
    x : float32 [1, D, tile, tile]   intensity in [0,1]
    y : float32 [1, tile, tile]      binary ink label
plus meta (segment, y, x) for inference reassembly.

Depth window: D layers centred on ``center_layer`` of the 65-layer stack
(default center 32 -> layers [16,48) for D=32).  Fixed for A0.
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def normalize_volume(xv, mode="global", norm=255.0):
    """Intensity-normalize a tile/window ``xv`` (float32, raw 0-255 values).

    mode:
      'global'          : xv / norm  (published behaviour; A0..A2bs).
      'pertile_zscore'  : per-tile standardization (xv-mu)/(sd), clipped to
                          +/-3 sigma and rescaled to [0,1].  Removes the
                          absolute-brightness domain shift between bright
                          mid-sheet tiles and dark/thin edge tiles (the
                          left-word failure), which global /255 could not.
      'pertile_pct'     : per-tile percentile window [p1,p99] -> [0,1]
                          (CLAHE-like dynamic-range stretch).
    The transform is identical for training tiles and sliding-window
    inference tiles, so train/eval stay consistent.
    """
    if mode in ("global", None):
        return xv / norm
    if mode == "pertile_zscore":
        mu = xv.mean()
        sd = xv.std()
        if sd < 1e-6:
            return np.zeros_like(xv) + 0.5
        z = (xv - mu) / sd
        z = np.clip(z, -3.0, 3.0)
        return (z / 6.0 + 0.5).astype(np.float32)
    if mode == "pertile_pct":
        lo, hi = np.percentile(xv, 1.0), np.percentile(xv, 99.0)
        if hi - lo < 1e-6:
            return np.zeros_like(xv) + 0.5
        return np.clip((xv - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if mode == "spatial_zscore":
        # Per-tile SPATIAL z-score: mu/sd computed over the H*W spatial plane only
        # (averaged across depth first), then applied to all depth slices.
        # This removes tile-to-tile global brightness variation (the dark left-word
        # domain shift) WITHOUT destroying the depth-axis contrast that the DA-EPA
        # depth branch and texture stem rely on to locate the ink sheet.
        # xv: [D, H, W] float32 raw 0-255.
        spatial_mean = xv.mean(axis=0, keepdims=True)        # [1,H,W] average over depth
        mu = spatial_mean.mean()                             # scalar per-tile mean
        sd = spatial_mean.std()                              # scalar per-tile std
        if sd < 1e-6:
            return (xv / norm).astype(np.float32)
        xv_norm = (xv - mu) / sd                            # preserves depth profile shape
        xv_norm = np.clip(xv_norm, -3.0, 3.0)
        return (xv_norm / 6.0 + 0.5).astype(np.float32)
    raise ValueError(f"unknown norm_mode {mode}")


class _SegCache:
    """Lazy per-worker memmap cache so arrays are opened once per segment."""

    def __init__(self, label_file="ink.npy", unc_file="unc.npy"):
        self._vol, self._ink, self._mask = {}, {}, {}
        self._unc = {}
        self.label_file = label_file
        # unc_file (2026-07-20, reg2): trimap filename is config-selectable so the
        # ink_reg2 rebuild (unc_reg2.npy) coexists with the historical unc.npy.
        self.unc_file = unc_file

    def unc(self, proc_dir_fn, seg):
        # uncertainty trimap (2026-07-17, u1/u2 variants): 2=ink core, 0=bg core,
        # 1=uncertain band. Built by scripts/make_uncertainty.py.
        if seg not in self._unc:
            self._unc[seg] = np.load(os.path.join(proc_dir_fn(seg), self.unc_file), mmap_mode="r")
        return self._unc[seg]

    def vol(self, proc_dir_fn, seg):
        if seg not in self._vol:
            self._vol[seg] = np.load(os.path.join(proc_dir_fn(seg), "vol.npy"), mmap_mode="r")
        return self._vol[seg]

    def ink(self, proc_dir_fn, seg):
        if seg not in self._ink:
            self._ink[seg] = np.load(os.path.join(proc_dir_fn(seg), self.label_file), mmap_mode="r")
        return self._ink[seg]

    def mask(self, proc_dir_fn, seg):
        if seg not in self._mask:
            self._mask[seg] = np.load(os.path.join(proc_dir_fn(seg), "mask.npy"), mmap_mode="r")
        return self._mask[seg]


class InkDataset(Dataset):
    def __init__(self, manifest_csv, cfg_data, train, proc_dir_fn):
        self.tile = int(cfg_data["tile"])
        self.depth = int(cfg_data["depth"])
        self.center = int(cfg_data.get("center_layer", 32))
        self.n_layers = int(cfg_data.get("n_layers", 65))
        self.norm = float(cfg_data.get("normalize", 255.0))
        self.norm_mode = cfg_data.get("norm_mode", "global")
        self.label_file = cfg_data.get("label_file", "ink.npy")
        self.train = train
        self.aug = cfg_data.get("aug", {}) if train else {}
        self.proc_dir_fn = proc_dir_fn
        self.rows, self.kinds = self._read_manifest(manifest_csv)
        self._cache = _SegCache(self.label_file, cfg_data.get("unc_file", "unc.npy"))
        # use_uncertainty (2026-07-17): ship the unc.npy trimap with each tile
        # so masked / 3-class losses can down-weight untrustworthy labels.
        # Default False -> byte-identical prior behaviour.
        self.use_uncertainty = bool(cfg_data.get("use_uncertainty", False))
        # --- SegFormer3D C1/PDK halo carrier (2026-09-10). DEFAULT OFF -> byte-identical
        # behaviour for every pre-existing run. When on, a 2x-wider, 2x-downsampled crop
        # centred on the tile rides as input CHANNEL 1, so tensor rank and voxel count are
        # unchanged and engine/train.py, the sampler, the loss, the collapse and the head
        # are untouched (segformer3d_plan.md 2.1).
        _halo = cfg_data.get("halo", {}) or {}
        self.halo_on = bool(_halo.get("enabled", False))
        self.halo_factor = int(_halo.get("factor", 2))
        self.halo_mode = _halo.get("mode", "peripheral")
        if self.halo_on:
            if self.halo_mode not in ("peripheral", "self"):
                raise ValueError(f"unknown data.halo.mode '{self.halo_mode}' "
                                 "(expected 'peripheral' or 'self')")
            if self.aug.get("z_flip"):
                raise ValueError("data.halo + aug.z_flip is unsupported: the halo is "
                                 "augmented jointly along the depth axis. z_flip is OFF "
                                 "in the benchmark protocol anyway.")

        z0 = self.center - self.depth // 2
        self.z0_base = max(0, min(z0, self.n_layers - self.depth))
        # center_mode 'per_segment' (2026-07-17, G4): centre the depth window on
        # each segment's own intensity depth peak (meta.json), clamped to [24,40].
        # Default 'fixed' -> the historical [16,48) window for every segment.
        self.center_mode = cfg_data.get("center_mode", "fixed")
        self._seg_z0 = {}
        if self.center_mode == "per_segment":
            import json
            for seg in sorted({r[0] for r in self.rows}):
                meta = json.load(open(os.path.join(proc_dir_fn(seg), "meta.json")))
                c = int(np.clip(int(meta.get("depth_peak", self.center)), 24, 40))
                self._seg_z0[seg] = max(0, min(c - self.depth // 2,
                                               self.n_layers - self.depth))

    @staticmethod
    def _read_manifest(path):
        rows, kinds = [], []
        with open(path) as f:
            r = csv.DictReader(f)
            for d in r:
                rows.append((d["segment"], int(d["y"]), int(d["x"]),
                             float(d["ink_frac"]), float(d["mask_frac"])))
                kinds.append(d.get("kind", ""))
        return rows, kinds

    def __len__(self):
        return len(self.rows)

    @property
    def ink_fracs(self):
        return np.array([r[3] for r in self.rows], dtype=np.float32)

    def _depth_window(self, seg=None):
        z0 = self._seg_z0.get(seg, self.z0_base) if seg is not None else self.z0_base
        if self.train and self.aug.get("z_shift", 0):
            d = int(self.aug["z_shift"])
            if d > 0 and np.random.rand() < self.aug.get("z_shift_p", 0.3):
                z0 = z0 + np.random.randint(-d, d + 1)
        z0 = max(0, min(z0, self.n_layers - self.depth))
        return z0, z0 + self.depth

    def _halo_crop(self, vol, z0, z1, y, x):
        """2x-wider, factor-downsampled crop centred on the tile -> [D,t,t].

        Clamped to the segment and zero-filled outside; `halo_mask_frac` records how much
        of the halo actually exists (border tiles). Returns (halo, mask_frac)."""
        t, f = self.tile, self.halo_factor
        big = t * f
        y0, x0 = y + t // 2 - big // 2, x + t // 2 - big // 2
        H, W = vol.shape[1], vol.shape[2]
        big_arr = np.zeros((z1 - z0, big, big), dtype=np.float32)
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(H, y0 + big), min(W, x0 + big)
        frac = 0.0
        if ye > ys and xe > xs:
            big_arr[:, ys - y0:ye - y0, xs - x0:xe - x0] = np.asarray(
                vol[z0:z1, ys:ye, xs:xe], dtype=np.float32)
            frac = float((ye - ys) * (xe - xs)) / float(big * big)
        big_arr = normalize_volume(big_arr, self.norm_mode, self.norm)
        # average-pool f x f in-plane -> [D, t, t] (same voxel count as the tile)
        d = big_arr.shape[0]
        halo = big_arr.reshape(d, t, f, t, f).mean(axis=(2, 4))
        return np.ascontiguousarray(halo, dtype=np.float32), frac

    def __getitem__(self, idx):
        seg, y, x, _, _ = self.rows[idx]
        t = self.tile
        z0, z1 = self._depth_window(seg)
        vol = self._cache.vol(self.proc_dir_fn, seg)
        ink = self._cache.ink(self.proc_dir_fn, seg)

        xv = np.asarray(vol[z0:z1, y:y + t, x:x + t], dtype=np.float32)              # [D,t,t] raw
        xv = normalize_volume(xv, self.norm_mode, self.norm)
        yv = np.asarray(ink[y:y + t, x:x + t], dtype=np.float32)                     # [t,t]
        uv = None
        if self.use_uncertainty:
            unc = self._cache.unc(self.proc_dir_fn, seg)
            uv = np.asarray(unc[y:y + t, x:x + t], dtype=np.float32)                 # [t,t]

        hv, halo_frac = None, 1.0
        if self.halo_on:
            if self.halo_mode == "self":
                # same-cost control: halo = downsampled copy of the TILE (no new info)
                f, d = self.halo_factor, xv.shape[0]
                hv = xv.reshape(d, t // f, f, t // f, f).mean(axis=(2, 4))
                hv = np.repeat(np.repeat(hv, f, axis=1), f, axis=2).astype(np.float32)
            else:
                hv, halo_frac = self._halo_crop(vol, z0, z1, y, x)

        if self.train and self.aug:
            if hv is None:
                xv, yv, uv = self._augment(xv, yv, uv)
            else:
                # augment tile and halo JOINTLY (identical geometric transform) by
                # stacking along depth; in-plane ops act on axes (1,2) either way.
                d = xv.shape[0]
                cat = np.concatenate([xv, hv], axis=0)
                cat, yv, uv = self._augment(cat, yv, uv)
                xv, hv = cat[:d], cat[d:]

        xv = torch.from_numpy(np.ascontiguousarray(xv)).unsqueeze(0)  # [1,D,t,t]
        if hv is not None:
            xv = torch.cat([xv, torch.from_numpy(
                np.ascontiguousarray(hv)).unsqueeze(0)], dim=0)       # [2,D,t,t]
        yv = torch.from_numpy(np.ascontiguousarray(yv)).unsqueeze(0)  # [1,t,t]
        out = {"x": xv, "y": yv, "seg": seg, "yx": (y, x)}
        if hv is not None:
            out["halo_mask_frac"] = float(halo_frac)
        if uv is not None:
            out["u"] = torch.from_numpy(np.ascontiguousarray(uv)).unsqueeze(0)  # [1,t,t]
        return out

    def _augment(self, xv, yv, uv=None):
        a = self.aug
        if a.get("hflip") and np.random.rand() < 0.5:
            xv = xv[:, :, ::-1]; yv = yv[:, ::-1]
            if uv is not None: uv = uv[:, ::-1]
        if a.get("vflip") and np.random.rand() < 0.5:
            xv = xv[:, ::-1, :]; yv = yv[::-1, :]
            if uv is not None: uv = uv[::-1, :]
        if a.get("rot90") and np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            xv = np.rot90(xv, k, axes=(1, 2)); yv = np.rot90(yv, k, axes=(0, 1))
            if uv is not None: uv = np.rot90(uv, k, axes=(0, 1))
        if a.get("z_flip") and np.random.rand() < a.get("z_flip_p", 0.5):
            xv = xv[::-1, :, :]
        xv = self._photometric(xv)
        return xv, yv, uv

    def _photometric(self, xv):
        """Photometric augmentation (methodology \u00a72.5), applied AFTER normalization.

        Operates on the normalized tile (values ~[0,1]); labels untouched.  All
        transforms are gated by config keys under ``data.aug`` and default OFF,
        so prior runs are unaffected.  Motivation: the val-segment left word is
        a low-contrast / brightness-shifted regime that global /255 training
        never covers; gain/bias/gamma/contrast jitter teaches invariance at
        train time without moving the inference operating point (which is what
        made the pertile/spatial z-score experiments net-negative).
        """
        a = self.aug
        if a.get("brightness") and np.random.rand() < a.get("brightness_p", 0.5):
            xv = xv + np.random.uniform(-a["brightness"], a["brightness"])
        if a.get("contrast") and np.random.rand() < a.get("contrast_p", 0.5):
            f = 1.0 + np.random.uniform(-a["contrast"], a["contrast"])
            xv = (xv - xv.mean()) * f + xv.mean()
        if a.get("gamma") and np.random.rand() < a.get("gamma_p", 0.3):
            g = np.exp(np.random.uniform(-a["gamma"], a["gamma"]))
            xv = np.clip(xv, 0.0, None) ** g
        if a.get("noise_std") and np.random.rand() < a.get("noise_p", 0.3):
            xv = xv + np.random.normal(0.0, a["noise_std"], size=xv.shape).astype(np.float32)
        return np.clip(xv, 0.0, 1.0).astype(np.float32)


def build_sampler(dataset: InkDataset, cfg_data):
    """Weighted sampler oversampling ink-positive tiles.

    sampler_mode 'frac' (default — every pre-2026-07-17 run): positive tile
    (ink_frac >= pos_thresh) vs negative, ~1:1 in expectation.

    sampler_mode 'kind' (word-box manifests, plan §5): positives = rows with
    kind=='word' (annotation-anchored tiles), negatives = kind=='neg_clean'
    (>= neg-margin px from any labelled ink). kind=='neg_dirty' rows get weight
    0 unless ``use_dirty_neg: true``. ``pos_frac`` sets the positive share of
    each epoch's draws (default 0.5 = the historical 1:1).
    """
    fr = dataset.ink_fracs
    num_samples = cfg_data.get("samples_per_epoch", len(dataset))
    mode = cfg_data.get("sampler_mode", "frac")

    if mode == "kind":
        kinds = np.asarray(dataset.kinds)
        is_pos = kinds == "word"
        neg_ok = kinds == "neg_clean"
        if cfg_data.get("use_dirty_neg", False):
            neg_ok = neg_ok | (kinds == "neg_dirty")
        if not is_pos.any() or not neg_ok.any():
            raise ValueError("sampler_mode 'kind' needs a manifest with 'word' and "
                             "'neg_clean' rows — build it with scripts/make_word_manifests.py")
        pos_frac = float(cfg_data.get("pos_frac", 0.5))
        weights = np.zeros(len(fr), dtype=np.float64)
        weights[is_pos] = pos_frac / is_pos.sum()
        weights[neg_ok] = (1.0 - pos_frac) / neg_ok.sum()
        return WeightedRandomSampler(weights, num_samples=int(num_samples), replacement=True)

    pos_thresh = cfg_data.get("pos_thresh", 0.05)
    is_pos = fr >= pos_thresh
    n_pos = max(1, int(is_pos.sum()))
    n_neg = max(1, int((~is_pos).sum()))
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    weights = np.where(is_pos, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights, num_samples=int(num_samples), replacement=True)
