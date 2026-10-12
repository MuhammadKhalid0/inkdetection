#!/bin/bash -l
#SBATCH --job-name=ink_r2_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# r2 = stage-2 refiner trained on OUT-OF-FOLD t192 maps (submit with
# --dependency=afterok:<loo array job id>).  Selection input = FULL a2bs_t192
# val map (deployment condition).  Then postprocess sweep, figures, and a
# refined TEST map for qualitative comparison.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

RUN="r2"
FS_FULL="a2bs_t192"
OUT="$WORK/vesuvius/outputs/$RUN"

set -e

# all 8 OOF maps must exist
for SEG in 20241025145341 20241025145701 20241030152031 20241108111522 \
           20241108115232 20241108120732 20241113070770 20241113080880; do
  test -f "$WORK/vesuvius/outputs/a2bs_t192_loo/train_prob_${SEG}.npy" \
    || { echo "MISSING OOF map for $SEG"; exit 1; }
done

# --- train refiner on OOF maps ----------------------------------------------
srun python -m engine.refiner --first-stage a2bs_t192_loo --val-from "$FS_FULL" \
    --run-name "$RUN"

# --- postprocess + figures ----------------------------------------------------
python -m scripts.postprocess --run "$RUN"
TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
echo "== r2 best tau = $TAU =="
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg 20241113090990 \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2
python scripts/visualize_val.py --run "$RUN" --label-file ink_reg.npy --tau "$TAU"

# --- refined TEST map (qualitative), if the full model's test map exists ------
TEST_PROB=$(ls "$WORK/vesuvius/outputs/$FS_FULL"/test_prob_*.npy 2>/dev/null | head -1)
if [ -n "$TEST_PROB" ]; then
  python -m scripts.refine_predict --refiner "$RUN" --prob "$TEST_PROB" \
      --out "$OUT/test_prob_refined.npy"
  python scripts/visualize.py --prob "$OUT/test_prob_refined.npy" \
      --seg 20251115002740-auto_grown_20251115002740308_0_flatboi \
      --no-gt --tau "$TAU" --downsample 6 || true
  python scripts/visualize_test.py --run "$RUN" --tau "$TAU" --chunks 8 --ds 2 || true
  python scripts/show_pred.py --run "$RUN" --tau "$TAU" --ds 2 || true
else
  echo "no test map found for $FS_FULL; skipping test refinement"
fi

echo "== DONE r2 =="
