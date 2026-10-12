"""A0 training entry point.

Trains the unified model on the 8-segment train split, validates by running
sliding-window inference on the held-out validation segment and computing F0.5
on the reconstructed segment (fixed tau for checkpoint selection; full tau sweep
logged periodically).  Best-val-F0.5 checkpoint is saved.

Usage:
    python -m engine.train --config configs/a0.yaml
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader

from data.dataset import InkDataset, build_sampler
from data.segments import (MANIFEST_ROOT, VAL_SEGMENT, proc_dir)
from engine.eval import pixel_metrics, sweep_threshold
from engine.infer import infer_segment
from engine.losses import build_loss, LOSSES_NEED_UNC
from engine.utils import (AverageMeter, ModelEMA, count_params, load_config,
                          save_checkpoint, seed_all, warmup_cosine_lr)
from models.wrapper import build_model


def _log_csv(path, row, header):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    tr, dt = cfg["train"], cfg["data"]
    seed_all(tr.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = cfg["output"]["run_name"]
    out_dir = os.path.join(cfg["output"]["dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    # resume (2026-09-12): DEFAULT OFF. Enabled only by train.resume: true. What each guard
    # prevents is documented in engine/resume.py. Preflight runs BEFORE anything is written.
    resume = bool(tr.get("resume", False))
    if resume:
        from engine import resume as _resume
        _t_start = time.time()
        _ck = _resume.preflight(out_dir, cfg)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # --- data -------------------------------------------------------------
    man_suffix = dt.get("manifest_suffix", "")
    train_ds = InkDataset(os.path.join(MANIFEST_ROOT, f"train{man_suffix}.csv"), dt, train=True,
                          proc_dir_fn=proc_dir)
    sampler = build_sampler(train_ds, dt)
    train_dl = DataLoader(train_ds, batch_size=tr["batch"], sampler=sampler,
                          num_workers=tr.get("num_workers", 8), pin_memory=True,
                          drop_last=True, persistent_workers=tr.get("num_workers", 8) > 0)

    # --- model ------------------------------------------------------------
    model = build_model(cfg).to(device)
    print(f"[model] trainable params: {count_params(model):,}")
    loss_fn = build_loss(tr)
    loss_needs_unc = tr.get("loss", "dice_bce") in LOSSES_NEED_UNC
    head3 = int(cfg.get("model", {}).get("head_classes", 1)) == 3
    # EMA (2026-07-17, t1_ema): validate/select/save the EMA weights when set.
    ema = ModelEMA(model, decay=float(tr["ema"])) if tr.get("ema") else None
    # aux_shat (2026-07-18 closing experiment): auxiliary self-supervision of the
    # collapse's surface estimate s_hat toward the tile's intensity depth peak
    # s_ref (label-free; the G2 probe's reference). Interpretability showed s_hat
    # never learns to track the surface because the data never demands it — this
    # loss demands it directly. Weight via train.aux_shat; absent -> no-op.
    aux_w = float(tr.get("aux_shat", 0.0))
    if aux_w > 0:
        model.collapse.aux = True
        _gk = torch.exp(-0.5 * (torch.arange(-3, 4, dtype=torch.float32) / 1.5) ** 2)
        _gk = (_gk / _gk.sum()).view(1, 1, -1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr.get("weight_decay", 3e-5),
                            betas=(0.9, 0.999))
    # amp_dtype 'bf16' (fp32-range exponent, no overflow -> fixes t192 fp16
    # activation blow-up at e12-13) or 'fp16' (legacy default). GradScaler is
    # only needed for fp16; bf16 needs no loss scaling.
    amp_enabled = tr.get("amp", True)
    amp_dtype = torch.bfloat16 if tr.get("amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16)

    steps_per_epoch = len(train_dl)
    epochs = tr["epochs"]
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * tr.get("warmup_epochs", 3)
    val_tau = tr.get("val_tau", 0.4)
    # Checkpoint-selection metric (2026-07-03 fix): 'fixed' = F0.5@val_tau (legacy,
    # mis-selects because the calibrated optimum sits at tau 0.45-0.75), 'sweep' =
    # best F0.5 over the tau sweep (costs nothing: the prob map is already built).
    select_on = tr.get("select_on", "fixed")

    best_f = -1.0
    best_epoch = -1
    best_tau = val_tau
    patience = tr.get("patience", 20)
    step = 0
    start_epoch, exit_reason = 0, "completed"
    if resume:
        _rs = _resume.restore(_ck, out_dir, epochs, patience, steps_per_epoch,
                              model, opt, scaler, ema)
        del _ck
        start_epoch, step = _rs["start_epoch"], _rs["step"]
        best_f, best_epoch = _rs["best_f"], _rs["best_epoch"]
        if _rs["best_tau"] is not None:
            best_tau = _rs["best_tau"]
        exit_reason, _max_epoch_s = _rs["exit_reason"], _rs["max_epoch_s"]
        _epochs_done = _rs["epochs_done"]

    for epoch in range(start_epoch, epochs):
        if resume and _resume.time_budget_exceeded(_t_start, tr.get("time_limit_min"),
                                                   _max_epoch_s):
            exit_reason = "time_budget"
            print(f"[resume] time budget ({tr.get('time_limit_min')} min) would be exceeded "
                  f"by another epoch; stopping this chunk before epoch {epoch}")
            break
        model.train()
        lmeter = AverageMeter()
        t0 = time.time()
        for batch in train_dl:
            lr = warmup_cosine_lr(step, total_steps, warmup_steps, tr["lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            u = batch["u"].to(device, non_blocking=True) if loss_needs_unc and "u" in batch else None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_enabled):
                logits = model(x)
                loss = loss_fn(logits, y, u) if loss_needs_unc else loss_fn(logits, y)
                if aux_w > 0:
                    s_map = model.collapse.aux_s_hat.float()      # [B,1,H,W] or [B,1]
                    s_hat_tile = s_map.mean(dim=tuple(range(1, s_map.dim())))  # [B]
                    with torch.no_grad():
                        prof = x.float().mean(dim=(1, 3, 4))       # [B,D] intensity profile
                        profs = TF.conv1d(prof.unsqueeze(1), _gk, padding=3).squeeze(1)
                        s_ref = profs.argmax(dim=1).float()        # [B] smoothed peak
                    loss = loss + aux_w * TF.huber_loss(s_hat_tile, s_ref)
            if not torch.isfinite(loss):
                # A non-finite loss would poison the loss meter and (via repeated
                # skipped steps) hide divergence. Skip batch, log loudly.
                print(f"[warn] non-finite loss at epoch {epoch} step {step}, batch skipped")
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            scaler.scale(loss).backward()
            if tr.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            lmeter.update(loss.item(), x.size(0))
            step += 1

        # --- validation: reconstruct val segment, F0.5 -------------------
        eval_model = ema.module if ema is not None else model
        if head3:
            prob, mask, gt, unc_map = infer_segment(eval_model, VAL_SEGMENT, cfg,
                                                    device, proc_dir, also_unc=True)
        else:
            prob, mask, gt = infer_segment(eval_model, VAL_SEGMENT, cfg, device, proc_dir)
            unc_map = None
        m_fixed = pixel_metrics(prob, gt, mask, val_tau)
        do_sweep = (select_on == "sweep") or (epoch % tr.get("sweep_every", 5) == 0) \
            or (epoch == epochs - 1)
        if do_sweep:
            best_sw, _ = sweep_threshold(prob, gt, mask)
        else:
            best_sw = m_fixed
        dt_s = time.time() - t0
        print(f"[e{epoch:03d}] loss={lmeter.avg:.4f} "
              f"valF0.5@{val_tau}={m_fixed['fbeta']:.4f} "
              f"(sweep best F0.5={best_sw['fbeta']:.4f}@tau{best_sw['tau']}) "
              f"P={m_fixed['precision']:.3f} R={m_fixed['recall']:.3f} "
              f"lr={lr:.2e} {dt_s:.0f}s")
        _log_csv(os.path.join(out_dir, "train_log.csv"),
                 [epoch, round(lmeter.avg, 5), round(m_fixed["fbeta"], 5),
                  round(m_fixed["precision"], 5), round(m_fixed["recall"], 5),
                  round(best_sw["fbeta"], 5), best_sw["tau"], round(lr, 8),
                  round(best_sw.get("iou_ink", 0.0), 5),
                  round(best_sw.get("iou_bg", 0.0), 5)],
                 ["epoch", "train_loss", "val_f05", "val_P", "val_R",
                  "val_f05_sweep", "sweep_tau", "lr",
                  "val_iou_ink_sweep", "val_iou_bg_sweep"])

        sel = best_sw["fbeta"] if select_on == "sweep" else m_fixed["fbeta"]
        if sel > best_f:
            best_f, best_epoch, best_tau = sel, epoch, best_sw["tau"]
            save_checkpoint(os.path.join(out_dir, "best.pt"), eval_model, opt, epoch,
                            best_f, cfg, extra={"val_metric": m_fixed,
                                                "val_sweep": best_sw,
                                                "select_on": select_on,
                                                "ema": bool(ema is not None)})
            np.save(os.path.join(out_dir, "val_prob_best.npy"), prob.astype(np.float32))
            if unc_map is not None:
                np.save(os.path.join(out_dir, "val_prob_unc_best.npy"),
                        unc_map.astype(np.float32))
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump({"run": run_name, "select_on": select_on,
                           "best_epoch": best_epoch, "best_f05": best_f,
                           "best_tau": float(best_tau),
                           "fixed_tau": val_tau,
                           "f05_at_fixed_tau": m_fixed["fbeta"],
                           "precision": best_sw["precision"] if select_on == "sweep" else m_fixed["precision"],
                           "recall": best_sw["recall"] if select_on == "sweep" else m_fixed["recall"],
                           "iou_ink": best_sw.get("iou_ink"),
                           "iou_bg": best_sw.get("iou_bg"),
                           "miou": best_sw.get("miou"),
                           "ema": bool(ema is not None),
                           "label_file": dt.get("label_file", "ink.npy")}, f, indent=2)
        if resume:
            _ep_s = time.time() - t0
            _max_epoch_s = _ep_s if _max_epoch_s is None else max(_max_epoch_s, _ep_s)
            _resume.save_last(out_dir, cfg, model, opt, scaler, ema, epoch, step,
                              steps_per_epoch, best_f, best_epoch, best_tau, _max_epoch_s)
            _epochs_done = epoch + 1
        if epoch - best_epoch >= patience:
            print(f"early stop at epoch {epoch} (best {best_f:.4f} @ e{best_epoch})")
            exit_reason = "early_stop"
            break
        if resume and tr.get("stop_after_epoch") is not None \
                and epoch >= int(tr["stop_after_epoch"]):
            exit_reason = "chunk_stop"
            print(f"[resume] stop_after_epoch={tr['stop_after_epoch']} reached")
            break

    print(f"done. best val F0.5={best_f:.4f} @ epoch {best_epoch} (tau={best_tau}, select_on={select_on})")
    if resume:
        _resume.finish(out_dir, exit_reason, best_f, best_epoch, best_tau, _epochs_done)


if __name__ == "__main__":
    main()
