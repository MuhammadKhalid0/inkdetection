#!/bin/bash -l
#SBATCH --job-name=ink_loo_t192
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-7
#SBATCH --export=NONE

# LOO 8-fold first stage for the r2 out-of-fold refiner (2026-07-05).
# Task i: train a2bs_t192 recipe on 7/8 train segments (fold config, holds out
# TRAIN_SEGMENTS[i]) -> predict the held-out segment with the fold's best.pt ->
# save the OOF map to outputs/a2bs_t192_loo/train_prob_<seg>.npy.
# Val segment is used for selection only, never trained on (same as always).

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

I=$SLURM_ARRAY_TASK_ID
SEGS=(20241025145341 20241025145701 20241030152031 20241108111522 \
      20241108115232 20241108120732 20241113070770 20241113080880)
HELD=${SEGS[$I]}
CFG="configs/loo_t192_f${I}.yaml"
RUN="a2bs_t192_f${I}"
OOF_DIR="$WORK/vesuvius/outputs/a2bs_t192_loo"
mkdir -p "$OOF_DIR"

set -e
echo "== fold $I: held-out $HELD =="
python -m py_compile models/shared.py models/unetrpp/transformerblock.py engine/train.py

srun python -m engine.train --config "$CFG"

srun python -m scripts.predict --run "$RUN" --seg "$HELD" --labelled \
    --out "$OOF_DIR/train_prob_${HELD}.npy"

echo "== DONE fold $I ($HELD) =="
