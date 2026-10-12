#!/bin/bash -l
#SBATCH --job-name=ink2_analysis
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# 2026-07-17 follow-up round: inference-only analyses on EXISTING checkpoints
# (read-only on outputs/; all artifacts written to outputs2/).
#   1. plan §4  P1  — TTA re-scoring of 6 checkpoints (adopt/reject gate +0.003)
#   2. plan §9.1 G1 — depth-offset robustness stress test (8 ckpts × 9 shifts)
#   3. plan §9.2 G2 — s-hat fidelity probe (3 lean seeds)

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs2"

set -e
python -m py_compile engine/infer.py engine/eval.py models/shared.py \
    scripts/predict_val_tta.py scripts/stress_zshift.py scripts/probe_shat.py

echo "== [1/3] TTA re-scoring =="
srun python -m scripts.predict_val_tta

echo "== [2/3] G1 depth-offset stress test =="
srun python -m scripts.stress_zshift

echo "== [3/3] G2 s-hat fidelity probe =="
srun python -m scripts.probe_shat

echo "== DONE analysis job =="
