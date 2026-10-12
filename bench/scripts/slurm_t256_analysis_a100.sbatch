#!/bin/bash -l
#SBATCH --job-name=ink5_t256analysis
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# 2026-07-18: inference-only extensions to the t256/t320 checkpoints (outputs4,
# read-only): (1) depth-offset stress test — does surfrel's robustness survive
# at the deployed scale? (2) TTA rescoring — new best numbers with TTA.
# Artifacts -> outputs5/{zshift_stress,tta}.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs5"

set -e
python -m py_compile scripts/stress_zshift.py scripts/predict_val_tta.py

echo "== [1/2] t256/t320 depth-offset stress =="
srun python -m scripts.stress_zshift --root "$WORK/vesuvius/outputs4" \
    --runs t256_lean_s42 t256_lean_s43 t256_lean_s44 t256_nosurf_s42 t320_lean_s42

echo "== [2/2] t256 TTA rescoring =="
srun python -m scripts.predict_val_tta --root "$WORK/vesuvius/outputs4" \
    --runs t256_lean_s42 t256_lean_s43 t256_lean_s44 t256_nosurf_s42

echo "== DONE t256 analysis =="
