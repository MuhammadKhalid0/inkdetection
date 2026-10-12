#!/bin/bash -l
#SBATCH --job-name=ink3_abl64
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-10
#SBATCH --export=NONE

# 2026-07-17: t64 ablation suite RE-RUN under the FIXED training loop
# (user request — the original t64 verdicts were measured under the broken
# loop: LR never annealed, val oscillated 0.04-0.09/epoch; this decouples
# "tile-conditional" from "broken-loop-conditional").
#   0-2  abl64_ref_s42/43/44  — reference (a2bs_reg recipe) seed band
#   3    abl64_stem           — texture stem OFF
#   4    abl64_da             — depth attention OFF
#   5    abl64_surfrel        — surface-relative family OFF
#   6    abl64_perpix         — per-pixel warp OFF
#   7    abl64_warpstab       — warp stabilization OFF
#   8    abl64_labels         — unregistered labels
#   9    abl64_loss_recall    — Tversky 0.3/0.7
#   10   abl64_loss_prec      — Tversky 0.7/0.3
# Outputs -> outputs3/ (separate from outputs/ history and outputs2/ round-1).

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs3"

RUNS=(abl64_ref_s42 abl64_ref_s43 abl64_ref_s44 abl64_stem abl64_da \
      abl64_surfrel abl64_perpix abl64_warpstab abl64_labels \
      abl64_loss_recall abl64_loss_prec)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CFG="configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py engine/losses.py \
    data/dataset.py data/segments.py models/shared.py models/wrapper.py \
    scripts/postprocess.py scripts/plot_train_curves.py
echo "== task $SLURM_ARRAY_TASK_ID run=$RUN cfg=$CFG =="

python scripts/smoke_test.py --config "$CFG" --device cpu

srun python -m engine.train --config "$CFG"

# postprocess scores vs REGISTERED labels for every run (common yardstick,
# including abl64_labels — mirrors how the t192 suite was reported).
python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"

TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg "$VALSEG" \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE $RUN =="
