#!/bin/bash -l
#SBATCH --job-name=ink_a0
#SBATCH --gres=gpu:a100:1          # VERIFY partition/GPU flavour on Alex
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# --- Alex / NHR@FAU environment (VERIFY module names for your allocation) ----
unset SLURM_EXPORT_ENV
module load python                 # VERIFY
module load cuda                   # VERIFY (match the torch build)

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

CONFIG="${CONFIG:-configs/a0.yaml}"

# Pre-flight on CPU (fast, fails early on shape/import bugs)
python scripts/smoke_test.py --config "$CONFIG" --device cpu

# Train
srun python -m engine.train --config "$CONFIG"
