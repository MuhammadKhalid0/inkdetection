"""Stage-2 letterform refiner (2026-07-04).

Motivation: the first-stage model leaves letters INCOMPLETE (per-letter recall
0.3-0.5 on the val left word even at tile 128); the missing parts are
100-500 px chunks, so classical morphology cannot bridge them without
destroying precision (measured: closing/hole-fill <= +0.000 F0.5).  This module
trains a small 2D U-Net that maps the first-stage PROBABILITY MAP to the GT ink
mask on the 8 TRAINING segments, i.e. it learns the Greek letterform prior from
data.  Patch-dropout augmentation (attenuating letter-sized regions of the
input) explicitly teaches completion of weak/partial strokes — the deployment
condition on the val left word.  The val segment is never seen in training;
it is used (as designated) for model selection only.

Usage (after first-stage train-segment prob maps exist):
    python -m engine.refiner --first-stage a2bs_t128 --run-name r1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.segments import (TRAIN_SEGMENTS, VAL_SEGMENT, proc_dir,   # noqa: E402
                           OUTPUT_ROOT)
from engine.losses import dice_bce                                   # noqa: E402
from engine.utils import seed_all, warmup_cosine_lr                  # noqa: E402

TILE = 512
STRIDE = 256


# --------------------------------------------------------------------------- model
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = max(1, cout // 8)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class RefinerUNet(nn.Module):
    """Small 2D U-Net: 1-ch prob map -> 1-ch refined logits."""

    def __init__(self, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc0 = ConvBlock(1, c[0])
        self.enc1 = ConvBlock(c[0], c[1])
        self.enc2 = ConvBlock(c[1], c[2])
        self.enc3 = ConvBlock(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.dec2 = ConvBlock(c[2] * 2, c[2])
        self.up1 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.dec1 = ConvBlock(c[1] * 2, c[1])
        self.up0 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.dec0 = ConvBlock(c[0] * 2, c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], 1))
        return self.head(d0)


# --------------------------------------------------------------------------- data
def prob_path(first_stage: str, seg: str) -> str:
    return os.path.join(OUTPUT_ROOT, first_stage, f"train_prob_{seg[:16]}.npy")


class ProbTileDataset(Dataset):
    """Tiles of (first-stage prob map, GT ink) over the train segments.

    synth=True (r3 phase-A pretraining): the input is NOT the first-stage map
    but the GT itself, synthetically degraded into a prob-map-like image
    (thinned strokes, deleted letter chunks, soft edges, FP clutter).  The
    task is then PURE Greek-letterform completion, giving the refiner an
    explicit letter prior before fine-tuning on real maps.
    """

    def __init__(self, first_stage, label_file="ink_reg.npy", train=True, synth=False):
        self.train = train
        self.synth = synth
        self.maps = {}
        self.rows = []
        for seg in TRAIN_SEGMENTS:
            if synth:
                prob = None
            else:
                pp = prob_path(first_stage, seg)
                if not os.path.exists(pp):
                    raise FileNotFoundError(f"missing first-stage map {pp}")
                prob = np.load(pp, mmap_mode="r")
            ink = np.load(os.path.join(proc_dir(seg), label_file), mmap_mode="r")
            mask = np.load(os.path.join(proc_dir(seg), "mask.npy"), mmap_mode="r")
            self.maps[seg] = (prob, ink, mask)
            H, W = mask.shape
            ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) or [0]
            xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) or [0]
            for y in ys:
                for x in xs:
                    m = np.asarray(mask[y:y + TILE:8, x:x + TILE:8])
                    if m.mean() < 0.3:
                        continue
                    ifr = float(np.asarray(ink[y:y + TILE:8, x:x + TILE:8]).mean())
                    self.rows.append((seg, y, x, ifr))
        print(f"[refiner] {len(self.rows)} tiles from {len(TRAIN_SEGMENTS)} segments")

    def __len__(self):
        return len(self.rows)

    @property
    def ink_fracs(self):
        return np.array([r[3] for r in self.rows], dtype=np.float32)

    def _degrade(self, p):
        """Simulate weak/partial first-stage evidence (the deployment gap)."""
        # global under-confidence: gamma + scale
        if np.random.rand() < 0.7:
            p = np.clip(p, 0, 1) ** np.exp(np.random.uniform(-0.35, 0.6))
            p = p * np.random.uniform(0.6, 1.0)
        # letter-sized patch dropout / attenuation (teaches completion)
        for _ in range(np.random.randint(0, 5)):
            h = np.random.randint(80, 400)
            w = np.random.randint(80, 400)
            y = np.random.randint(0, max(TILE - h, 1))
            x = np.random.randint(0, max(TILE - w, 1))
            p[y:y + h, x:x + w] *= np.random.uniform(0.0, 0.5)
        # blur + noise
        if np.random.rand() < 0.5:
            p = ndimage.gaussian_filter(p, np.random.uniform(0.5, 3.0))
        if np.random.rand() < 0.5:
            p = p + np.random.normal(0, 0.03, p.shape).astype(np.float32)
        # np.float64 scalars (uniform/exp) promote float32 arrays to float64 under
        # numpy 2 -> double tensors -> AMP conv crash. Force float32 on exit.
        return np.clip(p, 0.0, 1.0).astype(np.float32)

    def _synth_from_gt(self, g):
        """r3 phase-A input: degraded GT letterforms rendered prob-map-like."""
        p = (g > 0.5).astype(np.float32)
        # stroke thinning / under-segmentation
        if np.random.rand() < 0.7 and p.any():
            p = ndimage.binary_erosion(
                p > 0.5, iterations=int(np.random.randint(1, 13))).astype(np.float32)
        # letter-sized chunk deletion (the completion signal)
        for _ in range(np.random.randint(1, 6)):
            h = np.random.randint(60, 400)
            w = np.random.randint(60, 400)
            y = np.random.randint(0, max(TILE - h, 1))
            x = np.random.randint(0, max(TILE - w, 1))
            if np.random.rand() < 0.7:
                p[y:y + h, x:x + w] = 0.0
            else:
                p[y:y + h, x:x + w] *= np.random.uniform(0.1, 0.5)
        # prob-map appearance: soft edges + confidence warp + under-confidence
        p = ndimage.gaussian_filter(p, np.random.uniform(1.0, 6.0))
        m = float(p.max())
        if m > 1e-6:
            p = p / m
        p = np.clip(p, 0.0, 1.0) ** np.exp(np.random.uniform(-0.5, 0.7))
        p = p * np.random.uniform(0.5, 1.0)
        # FP-like clutter so precision is also learned
        if np.random.rand() < 0.5:
            yy, xx = np.ogrid[:p.shape[0], :p.shape[1]]
            for _ in range(np.random.randint(1, 8)):
                by = np.random.randint(0, p.shape[0])
                bx = np.random.randint(0, p.shape[1])
                r = np.random.randint(5, 40)
                blob = ((yy - by) ** 2 + (xx - bx) ** 2) <= r * r
                p[blob] = np.maximum(p[blob], np.random.uniform(0.3, 0.8))
        p = p + np.random.normal(0, 0.02, p.shape)
        return np.clip(p, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, i):
        seg, y, x, _ = self.rows[i]
        prob, ink, mask = self.maps[seg]
        g = np.asarray(ink[y:y + TILE, x:x + TILE], dtype=np.float32)
        if self.synth:
            p = np.zeros_like(g)               # built from g after geometric aug
        else:
            p = np.asarray(prob[y:y + TILE, x:x + TILE], dtype=np.float32)
        if p.shape != (TILE, TILE):        # edge pad
            ph, pw = TILE - p.shape[0], TILE - p.shape[1]
            p = np.pad(p, ((0, ph), (0, pw)))
            g = np.pad(g, ((0, ph), (0, pw)))
        if self.train:
            if np.random.rand() < 0.5:
                p = p[:, ::-1]; g = g[:, ::-1]
            if np.random.rand() < 0.5:
                p = p[::-1]; g = g[::-1]
            k = np.random.randint(0, 4)
            if k:
                p = np.rot90(p, k); g = np.rot90(g, k)
            if self.synth:
                p = self._synth_from_gt(np.ascontiguousarray(g))
            else:
                p = self._degrade(np.ascontiguousarray(p))
        return (torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32)).unsqueeze(0))


# --------------------------------------------------------------------------- eval
def refine_map(model, prob, device, batch=16):
    """Sliding-window refinement of a full probability map (Gaussian blend)."""
    H, W = prob.shape
    ax = np.arange(TILE) - (TILE - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    gw = np.exp(-(xx ** 2 + yy ** 2) / (2 * (TILE / 6) ** 2)).astype(np.float32)
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE))
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE))
    if ys[-1] != H - TILE and H > TILE:
        ys.append(H - TILE)
    if xs[-1] != W - TILE and W > TILE:
        xs.append(W - TILE)
    coords = [(y, x) for y in ys for x in xs]
    out = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            bc = coords[i:i + batch]
            xb = np.stack([prob[y:y + TILE, x:x + TILE] for y, x in bc])
            xb = torch.from_numpy(xb).unsqueeze(1).float().to(device)
            with torch.autocast(device_type="cuda", enabled=device != "cpu"):
                pb = torch.sigmoid(model(xb).float()).squeeze(1).cpu().numpy()
            for (y, x), pr in zip(bc, pb):
                out[y:y + TILE, x:x + TILE] += pr * gw
                wsum[y:y + TILE, x:x + TILE] += gw
    return out / np.maximum(wsum, 1e-6)


def f05(pred, gt, m):
    tp = float((pred & gt & m).sum())
    fp = float((pred & ~gt & m).sum())
    fn = float((~pred & gt & m).sum())
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    return 1.25 * p * r / max(0.25 * p + r, 1e-9), p, r


def sweep(prob, gt, m):
    best = (-1, 0, 0, 0)
    for t in np.round(np.arange(0.30, 0.91, 0.05), 2):
        f, p, r = f05(prob >= t, gt, m)
        if f > best[0]:
            best = (f, p, r, t)
    return best


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-stage", default="a2bs_t128",
                    help="run dir holding train_prob_<seg>.npy inputs (may be an "
                         "OOF collection dir, e.g. a2bs_t192_loo)")
    ap.add_argument("--val-from", default=None,
                    help="run whose val_prob_best.npy is the selection input "
                         "(default: --first-stage). For OOF training this must be "
                         "the FULL first-stage model = the deployment condition.")
    ap.add_argument("--run-name", default="r1")
    ap.add_argument("--label-file", default="ink_reg.npy")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--samples-per-epoch", type=int, default=2000)
    ap.add_argument("--pretrain-epochs", type=int, default=0,
                    help="r3 phase-A: epochs on synthetic degraded-GT letterform "
                         "tiles (explicit Greek-letter completion prior) before "
                         "fine-tuning on real first-stage maps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(OUTPUT_ROOT, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    ds = ProbTileDataset(args.first_stage, args.label_file, train=True)
    fr = ds.ink_fracs
    is_pos = fr >= 0.02
    w = np.where(is_pos, 0.5 / max(is_pos.sum(), 1), 0.5 / max((~is_pos).sum(), 1))
    sampler = WeightedRandomSampler(w.astype(np.float64), args.samples_per_epoch, True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)

    # val inputs: first-stage map of the val segment + GT
    val_from = args.val_from or args.first_stage
    vprob = np.load(os.path.join(OUTPUT_ROOT, val_from, "val_prob_best.npy"))
    vproc = proc_dir(VAL_SEGMENT)
    vgt = np.load(os.path.join(vproc, args.label_file)) > 0
    vmask = np.load(os.path.join(vproc, "mask.npy")) > 0
    b0 = sweep(vprob, vgt, vmask)
    print(f"[refiner] first-stage val baseline: F0.5={b0[0]:.4f} P={b0[1]:.3f} "
          f"R={b0[2]:.3f} @tau{b0[3]:.2f}")

    model = RefinerUNet().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[refiner] params {n:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    steps = len(dl) * args.epochs
    wsteps = len(dl) * 2

    best_f, best_ep, step = -1.0, -1, 0

    # ---- r3 phase A: synthetic letterform-completion pretraining -------------
    if args.pretrain_epochs > 0:
        ds_pre = ProbTileDataset(args.first_stage, args.label_file, train=True,
                                 synth=True)
        fr_p = ds_pre.ink_fracs
        pos_p = fr_p >= 0.02
        wp = np.where(pos_p, 0.7 / max(pos_p.sum(), 1), 0.3 / max((~pos_p).sum(), 1))
        dl_pre = DataLoader(ds_pre, batch_size=args.batch,
                            sampler=WeightedRandomSampler(
                                wp.astype(np.float64), args.samples_per_epoch, True),
                            num_workers=8, pin_memory=True, drop_last=True,
                            persistent_workers=True)
        psteps = len(dl_pre) * args.pretrain_epochs
        pwarm = len(dl_pre) * 2
        pstep = 0
        for ep in range(args.pretrain_epochs):
            model.train()
            t0, lsum = time.time(), 0.0
            for xb, yb in dl_pre:
                lr = warmup_cosine_lr(pstep, psteps, pwarm, args.lr)
                for grp in opt.param_groups:
                    grp["lr"] = lr
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    loss = dice_bce(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                lsum += loss.item()
                pstep += 1
            msg = f"[pre e{ep:03d}] loss={lsum / len(dl_pre):.4f} ({time.time() - t0:.0f}s)"
            if (ep + 1) % 5 == 0 or ep == args.pretrain_epochs - 1:
                rmap = refine_map(model, vprob, device)
                f, p, r, t = sweep(rmap, vgt, vmask)
                msg += f" | refined val F0.5={f:.4f} @tau{t:.2f} (log only)"
            print(msg)
        print("[refiner] phase A done; fine-tuning on real maps")

    for ep in range(args.epochs):
        model.train()
        t0, lsum = time.time(), 0.0
        for xb, yb in dl:
            lr = warmup_cosine_lr(step, steps, wsteps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = dice_bce(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            lsum += loss.item()
            step += 1
        rmap = refine_map(model, vprob, device)
        f, p, r, t = sweep(rmap, vgt, vmask)
        print(f"[e{ep:03d}] loss={lsum / len(dl):.4f} refined valF0.5={f:.4f} "
              f"P={p:.3f} R={r:.3f} @tau{t:.2f} ({time.time() - t0:.0f}s)")
        if f > best_f:
            best_f, best_ep = f, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "f05": f,
                        "first_stage": args.first_stage}, os.path.join(out_dir, "best.pt"))
            np.save(os.path.join(out_dir, "val_prob_best.npy"), rmap.astype(np.float32))
            with open(os.path.join(out_dir, "summary.json"), "w") as fj:
                json.dump({"run": args.run_name, "first_stage": args.first_stage,
                           "val_from": val_from,
                           "baseline_f05": b0[0], "baseline_tau": float(b0[3]),
                           "best_f05": f, "best_tau": float(t),
                           "precision": p, "recall": r, "best_epoch": ep}, fj, indent=2)
        if ep - best_ep >= 15:
            print(f"early stop e{ep} (best {best_f:.4f} @ e{best_ep})")
            break
    print(f"done. refined best F0.5={best_f:.4f} @ e{best_ep} "
          f"(first-stage baseline {b0[0]:.4f})")


if __name__ == "__main__":
    main()
