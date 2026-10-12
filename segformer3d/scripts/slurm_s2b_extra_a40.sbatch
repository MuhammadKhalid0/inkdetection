#!/bin/bash -l
#SBATCH --job-name=sf_s2b
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
module load cuda

SF_DIR="$WORK/segFormer"
BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$SF_DIR/outputs"
mkdir -p "$INK_OUTPUT_ROOT"

RUNS=(sf_base_d96_t320_s42 sf_25d_t256_s42)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CFG="$SF_DIR/configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py engine/losses.py \
    data/dataset.py models/wrapper.py models/shared.py \
    "$SF_DIR"/src/segformer3d_ink/*.py
echo "== task $SLURM_ARRAY_TASK_ID run=$RUN cfg=$CFG (labels: ink_reg) =="
python scripts/smoke_test.py --config "$CFG" --device cpu

set +e
srun python -m engine.train --config "$CFG"
RC=$?
set -e
if [ $RC -ne 0 ]; then
    echo "== train exited rc=$RC — halving batch and retrying once =="
    python - <<PYEOF
import yaml
p = "$CFG"
c = yaml.safe_load(open(p))
c["train"]["batch"] = max(1, c["train"]["batch"] // 2)
open(p, "w").write("# batch auto-halved after rc!=0 on first attempt\n" + yaml.safe_dump(c, sort_keys=False))
PYEOF
    rm -rf "$OUT"
    srun python -m engine.train --config "$CFG"
fi

python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"

TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg "$VALSEG" \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

echo "== DONE $RUN =="
