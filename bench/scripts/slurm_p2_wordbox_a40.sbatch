#!/bin/bash -l
#SBATCH --job-name=ink2_wordbox
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-2
#SBATCH --export=NONE

# 2026-07-17 follow-up round, Phase 2 / S2 (plan.md §5): word-box sampling,
# 3 seeds (supervisor question C1 — needs error bars either way).
# Outputs go to outputs2/ — the historical outputs/ folder is never written.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs2"

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
if [ "$SEED" = "42" ]; then RUN="s2_wordbox_t192"; else RUN="s2_wordbox_t192_s${SEED}"; fi
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

python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"

TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg "$VALSEG" \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE $RUN =="
