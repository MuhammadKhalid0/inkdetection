"""Safety tests for resume-from-latest-checkpoint (engine/resume.py + engine/train.py).

Runs the REAL engine.train.main() on CPU: real model (SF-S), real dataset tiles (t64),
real loss, real optimizer and schedule. Only full-segment validation is replaced by a
deterministic stand-in (the same model scored on a fixed tensor), because sliding-window
inference over the whole validation segment is infeasible on CPU. num_workers=0 so the
comparison is exact.

Usage (from anywhere):  python tests/test_resume.py      ~5 min on CPU
"""
import copy, csv, hashlib, importlib.util, json, os, shutil, sys, tempfile, traceback

BENCH = "/home/woody/iwi5/iwi5401h/vesuvius/vesuvius-ink-bench"
os.chdir(BENCH)
sys.path.insert(0, BENCH)

import numpy as np
import torch
import yaml

torch.set_num_threads(4)

import engine.train as T_new                               # patched trainer
from engine import resume as R

_OLD_PY = os.path.join(tempfile.mkdtemp(prefix="train_old_"), "train_old.py")
shutil.copyfile(f"{BENCH}/engine/train.py.bak_pre_resume", _OLD_PY)   # importlib needs .py
_spec = importlib.util.spec_from_file_location("train_old", _OLD_PY)
T_old = importlib.util.module_from_spec(_spec)
sys.modules["train_old"] = T_old
_spec.loader.exec_module(T_old)                             # pre-change trainer, byte-for-byte

# --- deterministic stand-in for full-segment validation ------------------------------
_g = torch.Generator().manual_seed(1234)
_XV = torch.rand(1, 1, 32, 64, 64, generator=_g)
_GT = (torch.rand(64, 64, generator=_g) > 0.7).numpy().astype(np.float32)
_MASK = np.ones((64, 64), dtype=np.float32)


def fake_infer(model, seg, cfg, device, proc_dir_fn, load_labels=True, also_unc=False):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(_XV.to(device)).float())[0, 0].cpu().numpy()
    return p.astype(np.float32), _MASK, _GT


T_new.infer_segment = fake_infer
T_old.infer_segment = fake_infer

TMP = tempfile.mkdtemp(prefix="resume_test_")
BASE = {
    "model": {"name": "segformer3d", "feature_size": 16, "capacity": "S"},
    "shared": {"texture_stem": False, "collapse": "surface_relative"},
    "data": {"tile": 64, "depth": 32, "center_layer": 32, "n_layers": 65, "normalize": 255.0,
             "label_file": "ink_reg.npy", "manifest_suffix": "_reg", "pos_thresh": 0.05,
             "samples_per_epoch": 8, "use_uncertainty": True,
             "aug": {"hflip": True, "vflip": True, "rot90": True, "z_flip": False,
                     "z_shift": 2, "z_shift_p": 0.3}},
    "train": {"loss": "dice_bce_masked", "lr": 1.0e-4, "weight_decay": 3.0e-5, "epochs": 4,
              "warmup_epochs": 1, "batch": 2, "num_workers": 0, "amp": False,
              "grad_clip": 1.0, "val_tau": 0.4, "select_on": "sweep", "sweep_every": 1,
              "patience": 20, "seed": 42},
    "infer": {"stride": 32, "batch": 4},
    "output": {"dir": TMP, "run_name": "x"},
}
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}", flush=True)


def cfg_for(run, **train):
    c = copy.deepcopy(BASE)
    c["output"]["run_name"] = run
    for k, v in train.items():
        if v is None:
            c["train"].pop(k, None)
        else:
            c["train"][k] = v
    return c


def train(mod, cfg):
    path = os.path.join(TMP, f"{cfg['output']['run_name']}_{abs(hash(json.dumps(cfg, sort_keys=True)))}.yaml")
    yaml.safe_dump(cfg, open(path, "w"))
    old_argv = sys.argv
    sys.argv = ["train", "--config", path]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def d(run):
    return os.path.join(TMP, run)


def fbytes(p):
    return open(p, "rb").read()


def sha(p):
    return hashlib.sha256(fbytes(p)).hexdigest()


def same_tensors(a, b):
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def lr_col(run):
    return [r["lr"] for r in csv.DictReader(open(os.path.join(d(run), "train_log.csv")))]


def epochs_col(run):
    return [int(r["epoch"]) for r in csv.DictReader(open(os.path.join(d(run), "train_log.csv")))]


def ck(run, name):
    return torch.load(os.path.join(d(run), name), map_location="cpu", weights_only=False)


# =====================================================================================
def t1_inert_when_off():
    print("[1] resume OFF: patched trainer is byte-identical to the pre-change trainer")
    train(T_old, cfg_for("off_old"))
    train(T_new, cfg_for("off_new"))
    for f in ("train_log.csv", "summary.json", "config_used.json", "val_prob_best.npy"):
        # config_used.json contains the run name; compare with the name normalised
        a, b = fbytes(os.path.join(d("off_old"), f)), fbytes(os.path.join(d("off_new"), f))
        if f in ("config_used.json", "summary.json"):
            # both files record the run name, which necessarily differs between the two runs
            a, b = a.replace(b"off_old", b"X"), b.replace(b"off_new", b"X")
        check(f"{f} identical", a == b)
    co, cn = ck("off_old", "best.pt"), ck("off_new", "best.pt")
    check("best.pt model weights identical", same_tensors(co["model"], cn["model"]))
    check("best.pt optimizer state identical",
          all(torch.equal(co["optimizer"]["state"][i][k], cn["optimizer"]["state"][i][k])
              for i in co["optimizer"]["state"] for k in co["optimizer"]["state"][i]))
    check("no last.pt written when OFF", not os.path.exists(os.path.join(d("off_new"), "last.pt")))
    check("no train_done.json written when OFF",
          not os.path.exists(os.path.join(d("off_new"), "train_done.json")))


def t2_chunked_equals_straight():
    print("[2] resume ON: 4 epochs straight == 2 epochs, stop, resume, 2 epochs")
    train(T_new, cfg_for("straight", resume=True))
    train(T_new, cfg_for("chunked", resume=True, stop_after_epoch=1))
    check("chunk 1 stopped after epoch 1", epochs_col("chunked") == [0, 1])
    check("chunk 1 wrote no train_done.json",
          not os.path.exists(os.path.join(d("chunked"), "train_done.json")))
    train(T_new, cfg_for("chunked", resume=True))          # stop_after_epoch removed: allowed
    check("epochs 0..3 each exactly once", epochs_col("chunked") == [0, 1, 2, 3],
          str(epochs_col("chunked")))
    check("learning-rate column identical (schedule did not restart)",
          lr_col("chunked") == lr_col("straight"), f"{lr_col('chunked')} vs {lr_col('straight')}")
    check("train_log.csv byte-identical", fbytes(os.path.join(d("chunked"), "train_log.csv"))
          == fbytes(os.path.join(d("straight"), "train_log.csv")))
    s1 = json.load(open(os.path.join(d("straight"), "summary.json")))
    s2 = json.load(open(os.path.join(d("chunked"), "summary.json")))
    s1.pop("run"); s2.pop("run")
    check("summary.json identical", s1 == s2)
    check("best.pt weights identical", same_tensors(ck("straight", "best.pt")["model"],
                                                    ck("chunked", "best.pt")["model"]))
    check("final last.pt weights identical", same_tensors(ck("straight", "last.pt")["model"],
                                                          ck("chunked", "last.pt")["model"]))
    dn1 = json.load(open(os.path.join(d("straight"), "train_done.json")))
    dn2 = json.load(open(os.path.join(d("chunked"), "train_done.json")))
    check("train_done.json agrees", {k: dn1[k] for k in ("reason", "best_epoch", "best_f05", "epochs_run")}
          == {k: dn2[k] for k in ("reason", "best_epoch", "best_f05", "epochs_run")})
    check("no leftover .tmp files", not any(f.endswith(".tmp") for f in os.listdir(d("chunked"))))


def t3_duplicate_log_row_removed():
    print("[3] a log row written after the checkpoint is removed on resume")
    train(T_new, cfg_for("dup", resume=True, stop_after_epoch=1))
    lp = os.path.join(d("dup"), "train_log.csv")
    rows = list(csv.reader(open(lp, newline="")))
    fake = list(rows[-1]); fake[0] = "2"; fake[1] = "9.99999"
    with open(lp, "a", newline="") as f:
        csv.writer(f).writerow(fake)                        # simulate kill after log, before last.pt
    train(T_new, cfg_for("dup", resume=True))
    check("epochs 0..3 each exactly once", epochs_col("dup") == [0, 1, 2, 3], str(epochs_col("dup")))
    check("log identical to the uninterrupted run",
          fbytes(lp) == fbytes(os.path.join(d("straight"), "train_log.csv")))


def t4_best_checkpoint_never_clobbered():
    print("[4] a resumed run can never overwrite a better best.pt")
    train(T_new, cfg_for("clob", resume=True, stop_after_epoch=0))
    lp = os.path.join(d("clob"), "last.pt")
    c = torch.load(lp, map_location="cpu", weights_only=False)
    c["best_f"], c["best_epoch"] = 0.99, 0
    torch.save(c, lp)
    sp = os.path.join(d("clob"), "summary.json")
    s = json.load(open(sp)); s["best_f05"] = 0.99; json.dump(s, open(sp, "w"), indent=2)
    h_before = sha(os.path.join(d("clob"), "best.pt"))
    train(T_new, cfg_for("clob", resume=True))
    check("best.pt untouched (no epoch beat the restored best)",
          sha(os.path.join(d("clob"), "best.pt")) == h_before)
    check("summary.json best still 0.99",
          json.load(open(sp))["best_f05"] == 0.99)


def t5_summary_reconciled_upward():
    print("[5] summary.json newer than last.pt (kill between the two saves) is adopted")
    train(T_new, cfg_for("recon", resume=True, stop_after_epoch=1))
    sp = os.path.join(d("recon"), "summary.json")
    s = json.load(open(sp)); s["best_f05"] = 0.98; s["best_epoch"] = 1
    json.dump(s, open(sp, "w"), indent=2)
    h_before = sha(os.path.join(d("recon"), "best.pt"))
    train(T_new, cfg_for("recon", resume=True))
    check("summary.json best preserved at 0.98", json.load(open(sp))["best_f05"] == 0.98)
    check("best.pt preserved", sha(os.path.join(d("recon"), "best.pt")) == h_before)


def t6_config_drift_refused():
    print("[6] config drift is refused; chunk-control keys are allowed")
    train(T_new, cfg_for("drift", resume=True, stop_after_epoch=0))
    for key, val in (("lr", 5.0e-5), ("batch", 4), ("epochs", 6), ("warmup_epochs", 2)):
        bad = cfg_for("drift", resume=True); bad["train"][key] = val
        try:
            R.preflight(d("drift"), bad); ok = False; msg = "no error"
        except RuntimeError as e:
            ok = f"train.{key}" in str(e); msg = str(e).splitlines()[1].strip()
        check(f"train.{key} change refused", ok, msg)
    bad = cfg_for("drift", resume=True); bad["model"]["capacity"] = "M"
    try:
        R.preflight(d("drift"), bad); ok = False
    except RuntimeError as e:
        ok = "model.capacity" in str(e)
    check("model.capacity change refused", ok)
    good = cfg_for("drift", resume=True, time_limit_min=1320, stop_after_epoch=3)
    try:
        R.preflight(d("drift"), good); ok = True
    except RuntimeError:
        ok = False
    check("time_limit_min / stop_after_epoch changes allowed", ok)


def t7_dirty_directory_refused():
    print("[7] a directory with artifacts but no last.pt is refused, never overwritten")
    dd = os.path.join(TMP, "dirty"); os.makedirs(dd, exist_ok=True)
    open(os.path.join(dd, "train_log.csv"), "w").write("epoch\n0\n")
    try:
        R.preflight(dd, cfg_for("dirty", resume=True)); ok = False
    except RuntimeError as e:
        ok = "REFUSING to start fresh" in str(e)
    check("fresh start into a used directory refused", ok)
    check("the existing file is untouched", open(os.path.join(dd, "train_log.csv")).read() == "epoch\n0\n")


def t8_ema_resume():
    print("[8] EMA runs resume exactly (raw and EMA weights stored separately)")
    train(T_new, cfg_for("ema_s", resume=True, ema=0.9))
    train(T_new, cfg_for("ema_c", resume=True, ema=0.9, stop_after_epoch=1))
    train(T_new, cfg_for("ema_c", resume=True, ema=0.9))
    check("EMA: train_log.csv byte-identical",
          fbytes(os.path.join(d("ema_s"), "train_log.csv")) == fbytes(os.path.join(d("ema_c"), "train_log.csv")))
    check("EMA: best.pt (EMA weights) identical",
          same_tensors(ck("ema_s", "best.pt")["model"], ck("ema_c", "best.pt")["model"]))
    check("EMA: raw training weights identical",
          same_tensors(ck("ema_s", "last.pt")["model"], ck("ema_c", "last.pt")["model"]))
    try:
        R.restore(ck("ema_c", "last.pt"), d("ema_c"), 4, 20, 4, None, None, None, None)
        ok = False
    except RuntimeError as e:
        ok = "EMA" in str(e)
    check("resuming an EMA checkpoint without EMA is refused", ok)


def t9_time_guard():
    print("[9] time guard ends a chunk cleanly between epochs")
    train(T_new, cfg_for("tg", resume=True, time_limit_min=0))
    check("exactly one epoch ran", epochs_col("tg") == [0], str(epochs_col("tg")))
    check("no train_done.json (chunk, not finished)",
          not os.path.exists(os.path.join(d("tg"), "train_done.json")))
    check("last.pt holds epoch 0", ck("tg", "last.pt")["epoch"] == 0)
    train(T_new, cfg_for("tg", resume=True))
    check("resumed chunk completes identically to the straight run",
          fbytes(os.path.join(d("tg"), "train_log.csv")) == fbytes(os.path.join(d("straight"), "train_log.csv")))


def t10_already_done_is_a_noop():
    print("[10] resubmitting a finished run changes nothing")
    h = {f: sha(os.path.join(d("straight"), f)) for f in ("train_log.csv", "best.pt", "summary.json")}
    train(T_new, cfg_for("straight", resume=True))
    for f, v in h.items():
        check(f"{f} unchanged", sha(os.path.join(d("straight"), f)) == v)


def t11_early_stop_at_boundary():
    print("[11] an early-stop condition met right at the checkpoint is honoured")
    train(T_new, cfg_for("es", resume=True, stop_after_epoch=1))
    lp = os.path.join(d("es"), "last.pt")
    c = torch.load(lp, map_location="cpu", weights_only=False)
    c["best_f"], c["best_epoch"] = 1.0, 1 - 20                    # epoch - best_epoch == patience
    torch.save(c, lp)
    train(T_new, cfg_for("es", resume=True))
    check("no further epochs trained", epochs_col("es") == [0, 1], str(epochs_col("es")))
    dn = json.load(open(os.path.join(d("es"), "train_done.json")))
    check("train_done.json reason is early_stop", dn["reason"] == "early_stop", dn["reason"])


if __name__ == "__main__":
    tests = [t1_inert_when_off, t2_chunked_equals_straight, t3_duplicate_log_row_removed,
             t4_best_checkpoint_never_clobbered, t5_summary_reconciled_upward,
             t6_config_drift_refused, t7_dirty_directory_refused, t8_ema_resume,
             t9_time_guard, t10_already_done_is_a_noop, t11_early_stop_at_boundary]
    for t in tests:
        try:
            t()
        except Exception:
            FAIL.append(t.__name__)
            print(f"  [FAIL] {t.__name__} raised:\n" + traceback.format_exc(), flush=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed   (workdir {TMP})")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  ")
        sys.exit(1)
    shutil.rmtree(TMP, ignore_errors=True)
    print("ALL RESUME SAFETY CHECKS PASSED")
