#!/bin/bash -l
#SBATCH --job-name=nnf_n1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-6
#SBATCH --export=NONE

# 2026-07-19 nnFormer Phase N1 (nnformer_plan.md §5): reference + 6 stock
# component screens at t192, seed 42, nnF-S. Deltas read vs n_base (noise floor
# ±0.008: screens only kill or promote — no single-seed claims). CSSA/SR-PB
# screens follow after port step 2. Output root: $WORK/nnFormer/outputs.

unset SLURM_EXPORT_ENV
module load python
module load cuda

NNF_DIR="$WORK/nnFormer"
BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$NNF_DIR/outputs"
mkdir -p "$INK_OUTPUT_ROOT"

RUNS=(n_base_t192_s42 n_mean_t192_s42 n_abs_t192_s42 n_skipconv_t192_s42 \
      n_lvonly_t192_s42 n_embedz2_t192_s42 n_lr5e5_t192_s42)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CFG="$NNF_DIR/configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"

set -e
python -m py_compile engine/train.py engine/infer.py engine/losses.py \
    data/dataset.py models/wrapper.py models/shared.py \
    "$NNF_DIR"/src/nnformer_ink/*.py
echo "== task $SLURM_ARRAY_TASK_ID run=$RUN cfg=$CFG =="
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
