#!/bin/bash -l
#SBATCH --job-name=ink_inkalyzer_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# Inkalyzer (input-space IBA) depth + 3D attribution across a representative set of
# first-stage UNETR++ models (all depth=32; tiles chosen per model tile size).
# Story: architecture ladder (t64) -> reg -> tile scaling (t64->t128->t192) ->
# key t192 ablations -> best seed.  Depth-band concentration table + compare figure.
# Refiners (r1-r4) excluded: 2D prob-map input, no depth axis.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

set -e
python -c "import py_compile; [py_compile.compile(f,doraise=True) for f in ['xai/ink_iba.py','xai/distributions.py','scripts/inkalyzer_run.py']]; print('compile OK')"

RUNS="a0,a0s,a0b,a1,a2,a2b,a2bs,a1_reg,a2bs_reg,a2bs_t128,a2bs_t192,abl_surfrel,abl_da,abl_labels,a2bs_t192_s44"

srun python -m scripts.inkalyzer_run \
    --runs "$RUNS" \
    --label-file ink_reg.npy \
    --n-tiles 24 --ink-thr 0.06 \
    --n-steps 100 --n-steps-3d 20 --n-3d 3 --window-size 16 \
    --noise median --beta 0.5 --lr 0.1

echo "== inkalyzer done -> outputs/<run>/inkalyzer/ + outputs/inkalyzer_compare/ =="
