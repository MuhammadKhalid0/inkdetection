#!/bin/bash -l
#SBATCH --job-name=ink_refiner_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# Stage-2 letterform refiner (2026-07-04).
# 1) Predict first-stage (a2bs_t128) prob maps for the 8 TRAIN segments (skip existing).
# 2) Train RefinerUNet(prob->GT) with letter-sized patch-dropout augmentation.
# 3) Evaluate refined val map, postprocess sweep, render comparison figures.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

FS="a2bs_t128"
RUN="r1"
OUT="$WORK/vesuvius/outputs/$RUN"

set -e

# --- 1) first-stage maps for train segments (resumable) ---------------------
for SEG in 20241025145341 20241025145701 20241030152031 20241108111522 \
           20241108115232 20241108120732 20241113070770 20241113080880; do
  F="$WORK/vesuvius/outputs/$FS/train_prob_${SEG}.npy"
  if [ -f "$F" ]; then
    echo "skip $SEG (exists)"
  else
    srun python -m scripts.predict --run "$FS" --seg "$SEG" --labelled --out "$F"
  fi
done

# --- 2) train refiner --------------------------------------------------------
srun python -m engine.refiner --first-stage "$FS" --run-name "$RUN"

# --- 3) postprocess + figures ------------------------------------------------
python -m scripts.postprocess --run "$RUN"
TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
echo "== refined best tau = $TAU =="
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg 20241113090990 \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2
python scripts/visualize_val.py --run "$RUN" --label-file ink_reg.npy --tau "$TAU"

echo "== DONE refiner =="
