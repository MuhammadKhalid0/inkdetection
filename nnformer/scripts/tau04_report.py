"""tau=0.4 (Vesuvius-convention) report for all 7 N1 reg2 runs (2026-07-20, per Omar).

For each run: recompute raw P/R/F0.5 at the FIXED tau=0.4 against ink_reg2 (the
labels these runs were trained/selected on), next to the run's own swept-optimal
tau, so the swept-vs-fixed gap is visible on THESE runs directly (methodology
plan §2.7: "Vesuvius convention is tau=0.4 ... still select on val and report
what you used" -- this is the comparison that motivates that footnote).
Also regenerates the val (with GT) and test (no GT) 4-panel figures at tau=0.4.
CPU only, no GPU needed -- reuses saved probability maps.
"""
import json
import os
import subprocess
import sys

import numpy as np

WORK = "/home/woody/iwi5/iwi5401h"
BENCH = f"{WORK}/vesuvius/vesuvius-ink-bench"
OUT_ROOT = f"{WORK}/nnFormer/outputs"
VAL = "20241113090990"
TEST = "20251115002740-auto_grown_20251115002740308_0_flatboi"
PROC = f"{WORK}/vesuvius/data/proc/{VAL}"

RUNS = ["n_base_t192_s42", "n_mean_t192_s42", "n_abs_t192_s42", "n_skipconv_t192_s42",
        "n_lvonly_t192_s42", "n_embedz2_t192_s42", "n_lr5e5_t192_s42"]
TAU_FIXED = 0.4


def f05(pred, gt, m):
    tp = float((pred & gt & m).sum()); fp = float((pred & ~gt & m).sum())
    fn = float((~pred & gt & m).sum())
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return 1.25 * p * r / max(0.25 * p + r, 1e-9), p, r


def main():
    gt = np.load(os.path.join(PROC, "ink_reg2.npy")) > 0
    mask = np.load(os.path.join(PROC, "mask.npy")) > 0
    rows = []
    for run in RUNS:
        rd = os.path.join(OUT_ROOT, run)
        prob = np.load(os.path.join(rd, "val_prob_best.npy")).astype(np.float32)
        s = json.load(open(os.path.join(rd, "summary.json")))
        sweep_tau = s["best_tau"]
        f_sw, p_sw, r_sw = f05(prob >= sweep_tau, gt, mask)
        f_fx, p_fx, r_fx = f05(prob >= TAU_FIXED, gt, mask)
        rows.append((run, sweep_tau, f_sw, p_sw, r_sw, f_fx, p_fx, r_fx))
        print(f"{run:<22} sweep(tau={sweep_tau:.2f}): F0.5={f_sw:.4f} P={p_sw:.3f} R={r_sw:.3f}  |  "
              f"fixed(tau=0.4): F0.5={f_fx:.4f} P={p_fx:.3f} R={r_fx:.3f}  ddF0.5={f_fx-f_sw:+.4f}")

        # regenerate figures at tau=0.4 (val with GT, test without)
        subprocess.run([sys.executable, "scripts/visualize.py",
                        "--prob", os.path.join(rd, "val_prob_best.npy"), "--seg", VAL,
                        "--label-file", "ink_reg2.npy", "--tau", str(TAU_FIXED),
                        "--downsample", "2",
                        "--out", os.path.join(rd, "figs", f"{VAL}_tau0.4.png")],
                       cwd=BENCH, check=False)
        test_probs = [f for f in os.listdir(rd) if f.startswith("test_prob_") and f.endswith(".npy")]
        if test_probs:
            subprocess.run([sys.executable, "scripts/visualize.py",
                            "--prob", os.path.join(rd, test_probs[0]), "--seg", TEST,
                            "--no-gt", "--tau", str(TAU_FIXED), "--downsample", "6",
                            "--out", os.path.join(rd, "figs", f"test_tau0.4.png")],
                           cwd=BENCH, check=False)

    lines = ["# tau=0.4 (Vesuvius convention) vs swept-optimal tau -- N1 reg2 runs", "",
             "| run | sweep tau | sweep F0.5 | sweep P | sweep R | fixed F0.5 (tau=0.4) | fixed P | fixed R | delta F0.5 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.2f} | {r[2]:.4f} | {r[3]:.3f} | {r[4]:.3f} | "
                     f"{r[5]:.4f} | {r[6]:.3f} | {r[7]:.3f} | {r[5]-r[2]:+.4f} |")
    out = f"{WORK}/nnFormer/docs/tau04_report.md"
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n->", out)


if __name__ == "__main__":
    main()
