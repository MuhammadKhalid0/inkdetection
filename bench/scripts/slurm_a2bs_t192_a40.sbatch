#!/bin/bash -l
#SBATCH --job-name=ink_a2bs_t192_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# a2bs_t192: context-scaling continuation of the winning direction
# (a2bs_reg 0.639 -> a2bs_t128 0.719; letters up to ~1300px wide).
# Pipeline: smoke -> train -> postprocess sweep -> val figs -> test predict -> test figs.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

CONFIG="configs/a2bs_t192.yaml"
RUN="a2bs_t192"
OUT="$WORK/vesuvius/outputs/$RUN"

set -e
python scripts/smoke_test.py --config "$CONFIG" --device cpu

srun python -m engine.train --config "$CONFIG"

TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
echo "== best tau (val sweep) = $TAU =="

# post-processing operating point (component filter) on val
python -m scripts.postprocess --run "$RUN"

# validation figures: 4-panel (prob|pred|GT|overlay) + readable chunked strip
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg 20241113090990 \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2
python scripts/visualize_val.py --run "$RUN" --label-file ink_reg.npy --tau "$TAU"

# test segment: predict + figures (no labels -> qualitative, tau from val)
srun python -m scripts.predict --run "$RUN"
python scripts/visualize.py \
    --prob "$OUT"/test_prob_*.npy --seg 20251115002740-auto_grown_20251115002740308_0_flatboi \
    --no-gt --tau "$TAU" --downsample 6 \
    --out "$OUT/figs/test_prob_pred_tau${TAU}.png"
python scripts/visualize_test.py --run "$RUN" --tau "$TAU" --chunks 8 --ds 2
python scripts/show_pred.py --run "$RUN" --tau "$TAU" --ds 2

echo "== DONE $RUN =="
