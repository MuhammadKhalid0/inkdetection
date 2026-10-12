"""Training utilities: seeding, warmup-cosine LR, checkpointing, config load."""
import math
import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(1.0, max(0.0, prog))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, v, k=1):
        self.sum += float(v) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(1, self.n)


def save_checkpoint(path, model, optimizer, epoch, best_metric, cfg, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg": cfg,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelEMA:
    """Exponential moving average of model weights (2026-07-17, t1_ema run).

    Motivation: late-epoch swept val F0.5 oscillates +-0.008 (train_log of every
    t192 run); the saved checkpoint is a draw from that oscillation. EMA
    evaluates/keeps the smoothed weights instead. Gated behind ``train.ema`` in
    the config (absent -> this class is never constructed; zero change).
    """

    def __init__(self, model, decay: float = 0.999):
        import copy
        self.decay = float(decay)
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        esd = self.module.state_dict()
        msd = model.state_dict()
        for k, v in esd.items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach().to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(mv)
