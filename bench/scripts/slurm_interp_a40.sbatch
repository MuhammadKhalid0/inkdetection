#!/bin/bash -l
#SBATCH --job-name=ink_interp
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# 2026-07-18 interpretability round (plan §14.4) — inference-only, reads
# existing checkpoints, writes to output_inter/. Probes:
#   1 error anatomy (CPU)  2 ERF ladder (GPU)  4 phi_rel readout (CPU)
#   5 frequency-band occlusion (GPU). EPA-hook probe deferred by decision.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

set -e
python -m py_compile scripts/interp_phirel.py scripts/interp_error_anatomy.py \
    scripts/interp_erf.py scripts/interp_freqband.py

echo "== [1/4] phi_rel depth readout =="
python -m scripts.interp_phirel

echo "== [2/4] error anatomy (t256 + t192) =="
python -m scripts.interp_error_anatomy --root "$WORK/vesuvius/outputs4" --run t256_lean_s42
python -m scripts.interp_error_anatomy --root "$WORK/vesuvius/outputs" --run lean_t192_s42

echo "== [3/4] ERF ladder =="
srun python -m scripts.interp_erf

echo "== [4/4] frequency-band occlusion =="
srun python -m scripts.interp_freqband --root "$WORK/vesuvius/outputs4" --run t256_lean_s42

echo "== DONE interpretability round =="
