#!/bin/bash -l
#SBATCH --job-name=ink_r3_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# r3 = two-phase "Greek letters first" refiner (2026-07-06).
# Phase A: pretrain on synthetic degraded-GT letterform tiles (explicit
#          Greek-letter completion prior; no first-stage maps involved).
# Phase B: fine-tune on real a2bs_t192 self maps (= the r2_self recipe).
# Compare vs r2_self 0.7876/0.7972 to isolate the value of the explicit prior.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

FS="a2bs_t192"
RUN="r3"
OUT="$WORK/vesuvius/outputs/$RUN"

set -e

# self maps must exist (produced by the r2_self job)
for SEG in 20241025145341 20241025145701 20241030152031 20241108111522 \
           20241108115232 20241108120732 20241113070770 20241113080880; do
  test -f "$WORK/vesuvius/outputs/$FS/train_prob_${SEG}.npy" \
    || { echo "MISSING self map for $SEG (run r2_self job first)"; exit 1; }
done

# --- two-phase refiner train --------------------------------------------------
srun python -m engine.refiner --first-stage "$FS" --run-name "$RUN" \
    --pretrain-epochs 25

# --- postprocess + figures ------------------------------------------------------
python -m scripts.postprocess --run "$RUN"
TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
echo "== r3 best tau = $TAU =="

python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg 20241113090990 \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2
python scripts/visualize_val.py --run "$RUN" --label-file ink_reg.npy --tau "$TAU"

# --- refined TEST map + test figures --------------------------------------------
TEST_PROB=$(ls "$WORK/vesuvius/outputs/$FS"/test_prob_*.npy 2>/dev/null | head -1)
if [ -n "$TEST_PROB" ]; then
  python -m scripts.refine_predict --refiner "$RUN" --prob "$TEST_PROB" \
      --out "$OUT/test_prob_refined.npy"
  python scripts/visualize.py --prob "$OUT/test_prob_refined.npy" \
      --seg 20251115002740-auto_grown_20251115002740308_0_flatboi \
      --no-gt --tau "$TAU" --downsample 6 || true
  python scripts/visualize_test.py --run "$RUN" --tau "$TAU" --chunks 8 --ds 2 || true
  python scripts/show_pred.py --run "$RUN" --tau "$TAU" --ds 2 || true
else
  echo "no test map found for $FS; skipping test refinement"
fi

echo "== DONE r3 =="
