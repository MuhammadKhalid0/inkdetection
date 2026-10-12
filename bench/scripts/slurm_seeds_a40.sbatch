#!/bin/bash -l
#SBATCH --job-name=ink_seeds_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-1
#SBATCH --export=NONE

# Seed repeats for mean+-std reporting (2026-07-06).
# Task 0: a2bs_t192 seed 43 -> then r4-recipe refiner seed 43 on the OOF maps.
# Task 1: a2bs_t192 seed 44 -> then r4-recipe refiner seed 44.
# First-stage seed42 = a2bs_t192 (exists); refiner seed42 = r4 (exists).
# Refiner seeds use the SAME OOF maps + the SAME full-model val map as r4, so
# they isolate refiner-training noise; first-stage seeds isolate backbone noise.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

SEEDS=(43 44)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CFG="configs/a2bs_t192_s${S}.yaml"
RUN="a2bs_t192_s${S}"

set -e
echo "== seed $S =="
python -m py_compile models/shared.py models/unetrpp/transformerblock.py engine/train.py engine/refiner.py

# --- first-stage seed repeat --------------------------------------------------
srun python -m engine.train --config "$CFG"
python -m scripts.postprocess --run "$RUN"

# --- refiner seed repeat (r4 recipe: letterform pretrain + OOF fine-tune) -----
srun python -m engine.refiner --first-stage a2bs_t192_loo --val-from a2bs_t192 \
    --run-name "r4_s${S}" --pretrain-epochs 25 --seed "$S"
python -m scripts.postprocess --run "r4_s${S}"

echo "== DONE seed $S =="
