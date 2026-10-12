"""Resume-from-latest-checkpoint for engine/train.py (2026-09-12). DEFAULT OFF.

Enabled only by `train.resume: true`. With it absent or false, engine/train.py runs
exactly as before: no file is read or written by this module, no RNG is consumed.

Each item below guards one specific way a naive resume silently damages a run:

  1. BEST-CHECKPOINT CLOBBER. train.py overwrites best.pt whenever validation beats
     `best_f`, which starts at -1. A resume that does not restore it makes the first
     resumed epoch a "new best" and overwrites the real best model. -> best_f /
     best_epoch / best_tau are restored, and reconciled upward against summary.json.
  2. (handled in the sbatch: no `rm -rf`, no batch halving on resumed jobs.)
  3. LR SCHEDULE RESTART. warmup+cosine is driven by a global `step` that train.py never
     saved. -> `step` is saved, restored, and checked against epoch x steps_per_epoch.
  4. DUPLICATE LOG ROWS. A kill between writing a train_log.csv row and saving last.pt
     duplicates that epoch on resume. -> rows after the checkpoint epoch are removed,
     and the remaining rows must be exactly epochs 0..k.
  5. EMA. With EMA on, best.pt holds the averaged weights. -> last.pt stores the raw
     model and the EMA weights separately, and refuses a mismatch.
  6. CONFIG DRIFT. A different batch / epochs / lr breaks the step->lr mapping. -> the
     full config must match the checkpoint's, except the chunk-control keys below.

Also: last.pt is written atomically (tmp file + fsync + os.replace), so a kill mid-write
cannot corrupt it; and a directory that already holds artifacts but no last.pt is refused
rather than silently overwritten.
"""
import copy
import csv
import json
import os
import random
import time

import numpy as np
import torch

FORMAT = "ink-resume-v1"
LAST = "last.pt"
DONE = "train_done.json"
# keys that legitimately differ between chunks of the same run
CHUNK_KEYS = (("train", "resume"), ("train", "time_limit_min"), ("train", "stop_after_epoch"))
_ARTIFACTS = ("train_log.csv", "best.pt", "summary.json", "val_prob_best.npy")


def _strip(cfg):
    c = copy.deepcopy(cfg)
    for sec, key in CHUNK_KEYS:
        if isinstance(c.get(sec), dict):
            c[sec].pop(key, None)
    return c


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append(f"{p}: absent in checkpoint, current={b[k]!r}")
            elif k not in b:
                out.append(f"{p}: checkpoint={a[k]!r}, absent in current config")
            else:
                out += _diff(a[k], b[k], p)
    elif a != b:
        out.append(f"{path}: checkpoint={a!r} current={b!r}")
    return out


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def preflight(out_dir, cfg):
    """Call BEFORE anything is written into out_dir. Returns the loaded checkpoint or None."""
    last = os.path.join(out_dir, LAST)
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("format") != FORMAT:
            raise RuntimeError(f"[resume] REFUSING: {last} has format {ck.get('format')!r}, "
                               f"expected {FORMAT!r}.")
        d = _diff(_strip(ck["cfg"]), _strip(cfg))
        if d:
            raise RuntimeError("[resume] REFUSING: the current config differs from the one this "
                               "checkpoint was trained with (only train.resume, "
                               "train.time_limit_min and train.stop_after_epoch may change "
                               "between chunks):\n  " + "\n  ".join(d))
        return ck
    dirty = [f for f in _ARTIFACTS if os.path.exists(os.path.join(out_dir, f))]
    if dirty:
        raise RuntimeError(f"[resume] REFUSING to start fresh: {out_dir} already contains "
                           f"{dirty} but no {LAST}. Starting fresh would overwrite them. "
                           "Archive that directory or use a new output.run_name.")
    return None


def _truncate_log(path, last_epoch):
    """Keep exactly the rows for epochs 0..last_epoch. Returns the number of rows removed."""
    if not os.path.exists(path):
        raise RuntimeError(f"[resume] REFUSING: checkpoint is at epoch {last_epoch} but "
                           f"{path} is missing.")
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if r]
    keep, seen = [], set()
    for r in body:
        e = int(r[0])
        if e <= last_epoch and e not in seen:
            seen.add(e)
            keep.append(r)
    if [int(r[0]) for r in keep] != list(range(last_epoch + 1)):
        raise RuntimeError(f"[resume] REFUSING: {path} does not contain exactly epochs "
                           f"0..{last_epoch} (found {[int(r[0]) for r in keep]}).")
    removed = len(body) - len(keep)
    if removed:
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(keep)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    return removed


def restore(ck, out_dir, epochs, patience, steps_per_epoch, model, opt, scaler, ema):
    """Restore all training state. Returns a dict the training loop consumes."""
    if ck is None:
        print("[resume] no last.pt — starting a fresh run with resume enabled")
        return dict(start_epoch=0, step=0, best_f=-1.0, best_epoch=-1, best_tau=None,
                    exit_reason="completed", max_epoch_s=None, epochs_done=0)
    if ck["steps_per_epoch"] != steps_per_epoch:
        raise RuntimeError(f"[resume] REFUSING: steps_per_epoch is {steps_per_epoch} now but "
                           f"{ck['steps_per_epoch']} in the checkpoint.")
    epoch, step = int(ck["epoch"]), int(ck["step"])
    if step != (epoch + 1) * steps_per_epoch:
        raise RuntimeError(f"[resume] REFUSING: step {step} is inconsistent with epoch {epoch} "
                           f"x {steps_per_epoch} steps.")
    if (ema is None) != (ck.get("ema") is None):
        raise RuntimeError("[resume] REFUSING: EMA on/off differs from the checkpoint.")

    model.load_state_dict(ck["model"])                       # strict
    opt.load_state_dict(ck["optimizer"])
    scaler.load_state_dict(ck["scaler"])
    if ema is not None:
        ema.module.load_state_dict(ck["ema"])

    best_f, best_epoch, best_tau = float(ck["best_f"]), int(ck["best_epoch"]), ck["best_tau"]
    # Reconcile UPWARD with summary.json: if the job died after best.pt/summary.json were
    # written but before last.pt was, the on-disk best is newer. Adopting it means the
    # re-run epoch must beat the real best to overwrite it -> best.pt can never get worse.
    sp = os.path.join(out_dir, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        if s.get("best_f05") is not None and float(s["best_f05"]) > best_f:
            print(f"[resume] summary.json best {s['best_f05']:.5f} @e{s['best_epoch']} is newer "
                  f"than last.pt's {best_f:.5f} @e{best_epoch}; adopting it")
            best_f, best_epoch = float(s["best_f05"]), int(s["best_epoch"])
            best_tau = s.get("best_tau", best_tau)

    removed = _truncate_log(os.path.join(out_dir, "train_log.csv"), epoch)
    if removed:
        print(f"[resume] removed {removed} train_log.csv row(s) written after the checkpoint")

    rng = ck["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if (torch.cuda.is_available() and rng.get("cuda") is not None
            and len(rng["cuda"]) == torch.cuda.device_count()):
        torch.cuda.set_rng_state_all(rng["cuda"])

    start_epoch, exit_reason = epoch + 1, "completed"
    if os.path.exists(os.path.join(out_dir, DONE)):
        exit_reason = json.load(open(os.path.join(out_dir, DONE))).get("reason", "completed")
        start_epoch = epochs
        print(f"[resume] {DONE} present (reason={exit_reason}) — nothing left to train")
    elif epoch - best_epoch >= patience:
        exit_reason, start_epoch = "early_stop", epochs
        print(f"[resume] early-stop condition already met at epoch {epoch} "
              f"(best @e{best_epoch}) — nothing left to train")
    else:
        print(f"[resume] resuming at epoch {start_epoch}/{epochs} (step {step}); "
              f"best {best_f:.5f} @e{best_epoch}")
    return dict(start_epoch=start_epoch, step=step, best_f=best_f, best_epoch=best_epoch,
                best_tau=best_tau, exit_reason=exit_reason,
                max_epoch_s=ck.get("max_epoch_s"), epochs_done=epoch + 1)


def save_last(out_dir, cfg, model, opt, scaler, ema, epoch, step, steps_per_epoch,
              best_f, best_epoch, best_tau, max_epoch_s):
    payload = {
        "format": FORMAT,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.module.state_dict() if ema is not None else None,
        "epoch": int(epoch), "step": int(step), "steps_per_epoch": int(steps_per_epoch),
        "best_f": float(best_f), "best_epoch": int(best_epoch),
        "best_tau": float(best_tau) if best_tau is not None else None,
        "max_epoch_s": max_epoch_s,
        "cfg": cfg,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "saved_at": time.time(), "torch_version": torch.__version__,
    }
    final = os.path.join(out_dir, LAST)
    tmp = final + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)


def time_budget_exceeded(t_start, limit_min, max_epoch_s, factor=1.25):
    """True if starting another epoch risks running past the chunk's budget."""
    if limit_min is None or max_epoch_s is None:
        return False
    return (time.time() - t_start) + factor * max_epoch_s > float(limit_min) * 60.0


def finish(out_dir, exit_reason, best_f, best_epoch, best_tau, epochs_run):
    if exit_reason in ("completed", "early_stop"):
        _atomic_json(os.path.join(out_dir, DONE),
                     {"reason": exit_reason, "best_f05": best_f, "best_epoch": best_epoch,
                      "best_tau": float(best_tau) if best_tau is not None else None,
                      "epochs_run": epochs_run, "finished_at": time.time()})
        print(f"[resume] training finished (reason={exit_reason}); wrote {DONE}")
    else:
        print(f"[resume] chunk ended early (reason={exit_reason}) after epoch {epochs_run - 1}; "
              f"resubmitting with the same config continues from {LAST}")
