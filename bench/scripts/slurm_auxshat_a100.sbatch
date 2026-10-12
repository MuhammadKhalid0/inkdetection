#!/bin/bash -l
#SBATCH --job-name=ink6_auxshat
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# 2026-07-18 geometry CLOSING experiment (plan §14.5): can the surface estimator
# be made to actually learn the surface, and does learned geometry buy
# robustness the frozen prior doesn't? Train lean t192 + aux s_hat supervision,
# then evaluate with BOTH existing geometry probes:
#   G2 probe  -> does s_hat now track s_ref?    (prediction: r 0.2 -> >0.8)
#   G1 stress -> flatter offset-robustness curve? (~50%)
# Outputs -> outputs6/.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs6"

RUN="aux_shat_t192_s42"
CFG="configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py models/shared.py \
    data/dataset.py scripts/postprocess.py scripts/plot_train_curves.py \
    scripts/probe_shat.py scripts/stress_zshift.py
echo "== run=$RUN =="

python scripts/smoke_test.py --config "$CFG" --device cpu

srun python -m engine.train --config "$CFG"

python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"

TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg "$VALSEG" \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

echo "== G2 probe: did s_hat learn the surface? =="
srun python -m scripts.probe_shat --root "$INK_OUTPUT_ROOT" --runs "$RUN"

echo "== G1 stress: does learned geometry buy robustness? =="
srun python -m scripts.stress_zshift --root "$INK_OUTPUT_ROOT" --runs "$RUN"

srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE $RUN =="
