#!/bin/bash -l
#SBATCH --job-name=ink_r4_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# r4 = two-phase refiner: phase A synthetic Greek-letterform pretrain (as r3),
# phase B fine-tune on LOO OUT-OF-FOLD t192 maps (as r2).  Completes the 2x2:
#   phase-B maps \ pretrain |  none   |  letterform
#   self (a2bs_t192)        |  r2_self|  r3
#   OOF  (a2bs_t192_loo)    |  r2     |  r4
# Hypothesis: the letter prior may rescue the OOF variant (r2 underperformed
# because fold maps are weaker than deployment maps; a strong shape prior
# could make the refiner less sensitive to that input gap).

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

FS_FULL="a2bs_t192"
RUN="r4"
OUT="$WORK/vesuvius/outputs/$RUN"

set -e

# OOF maps must exist (produced by the LOO array job)
for SEG in 20241025145341 20241025145701 20241030152031 20241108111522 \
           20241108115232 20241108120732 20241113070770 20241113080880; do
  test -f "$WORK/vesuvius/outputs/a2bs_t192_loo/train_prob_${SEG}.npy" \
    || { echo "MISSING OOF map for $SEG"; exit 1; }
done

# --- two-phase refiner train ---------------------------------------------------
srun python -m engine.refiner --first-stage a2bs_t192_loo --val-from "$FS_FULL" \
    --run-name "$RUN" --pretrain-epochs 25

# --- postprocess + figures -------------------------------------------------------
python -m scripts.postprocess --run "$RUN"
TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
echo "== r4 best tau = $TAU =="

python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg 20241113090990 \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2
python scripts/visualize_val.py --run "$RUN" --label-file ink_reg.npy --tau "$TAU"

# --- refined TEST map + test figures ---------------------------------------------
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

echo "== DONE r4 =="
