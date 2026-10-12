#!/bin/bash -l
#SBATCH --job-name=ink_lean_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-2
#SBATCH --export=NONE

# R1 decisive run: LEAN config = a2bs_t192 recipe MINUS depth-attention MINUS
# texture stem (both shown harmful/neutral by ablations abl_da +.014/+.015 and
# abl_stem +.000/+.007), KEEPING the surface-relative family (only load-bearing
# block, -.009 when removed). 3 seeds -> compare mean+-std vs first-stage
# (raw .7795+-.0052 / post .7945+-.0081) and vs abl_da single-seed .7877/.8009.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CFG="configs/lean_t192_s${SEED}.yaml"
RUN="lean_t192_s${SEED}"
OUT="$WORK/vesuvius/outputs/$RUN"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py \
    models/unetrpp/transformerblock.py models/shared.py scripts/postprocess.py
echo "== task $SLURM_ARRAY_TASK_ID seed=$SEED cfg=$CFG =="

python scripts/smoke_test.py --config "$CFG" --device cpu

srun python -m engine.train --config "$CFG"

python -m scripts.postprocess --run "$RUN"

# val 4-panel at swept tau + test map + strip (tau 0.6-0.85 set rendered later if wanted)
TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg 20241113090990 \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE lean seed $SEED =="
