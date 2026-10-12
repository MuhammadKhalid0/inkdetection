#!/bin/bash -l
#SBATCH --job-name=ink_r2self_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# r2_self = refiner on the FULL a2bs_t192 model's OWN train-segment maps
# (self-prediction, like r1 but at t192).  Ablation partner for r2:
# r2 - r2_self isolates the out-of-fold effect; r2_self - a2bs_t192 isolates
# the refiner effect at t192.  Starts immediately (needs only a2bs_t192/best.pt).

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

FS="a2bs_t192"
RUN="r2_self"
OUT="$WORK/vesuvius/outputs/$RUN"

set -e

# --- 1) self-prediction maps for the 8 train segments (resumable) ------------
for SEG in 20241025145341 20241025145701 20241030152031 20241108111522 \
           20241108115232 20241108120732 20241113070770 20241113080880; do
  F="$WORK/vesuvius/outputs/$FS/train_prob_${SEG}.npy"
  if [ -f "$F" ]; then
    echo "skip $SEG (exists)"
  else
    srun python -m scripts.predict --run "$FS" --seg "$SEG" --labelled --out "$F"
  fi
done

# --- 2) train refiner ---------------------------------------------------------
srun python -m engine.refiner --first-stage "$FS" --run-name "$RUN"

# --- 3) postprocess + figures -------------------------------------------------
python -m scripts.postprocess --run "$RUN"
TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
echo "== r2_self best tau = $TAU =="

# val figures
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg 20241113090990 \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2
python scripts/visualize_val.py --run "$RUN" --label-file ink_reg.npy --tau "$TAU"

# --- 4) refined TEST map + test figures ---------------------------------------
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

echo "== DONE r2_self =="