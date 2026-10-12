#!/bin/bash -l
#SBATCH --job-name=ink2_screens
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-7
#SBATCH --export=NONE

# 2026-07-17 follow-up round (plan.md): seed-42 screening runs.
#   0 s3_wordbox_p67_t192  — 2:1 word:neg ratio           (§5)
#   1 s4_wordbox_p33_t192  — 1:2 word:neg ratio           (§5)
#   2 u1_ignore_t192       — uncertainty band ignored     (§6.3)
#   3 u2_soft3_t192        — 3-class low-prob softmax     (§6.4)
#   4 t1_ema_t192          — EMA weights                  (§7)
#   5 t2_nozflip_t192      — drop z_flip aug              (§7)
#   6 g3_zshift6_t192      — z_shift ±6 aug               (§9.3)
#   7 g4_center_t192       — per-segment depth centering  (§9.4)
# Outputs go to outputs2/ — the historical outputs/ folder is never written.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs2"

RUNS=(s3_wordbox_p67_t192 s4_wordbox_p33_t192 u1_ignore_t192 u2_soft3_t192 \
      t1_ema_t192 t2_nozflip_t192 g3_zshift6_t192 g4_center_t192)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CFG="configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py engine/losses.py \
    engine/utils.py data/dataset.py data/segments.py models/shared.py \
    models/wrapper.py scripts/postprocess.py scripts/plot_train_curves.py
echo "== task $SLURM_ARRAY_TASK_ID run=$RUN cfg=$CFG =="

python scripts/smoke_test.py --config "$CFG" --device cpu

srun python -m engine.train --config "$CFG"

python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"

TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg "$VALSEG" \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

# u2 extra deliverable: the predicted P(uncertain) map of the val segment
if [ -f "$OUT/val_prob_unc_best.npy" ]; then
    python scripts/visualize.py --prob "$OUT/val_prob_unc_best.npy" --seg "$VALSEG" \
        --no-gt --tau 0.5 --downsample 2 \
        --out "$OUT/figs/${VALSEG}_uncertainty.png" || true
fi

srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE $RUN =="
