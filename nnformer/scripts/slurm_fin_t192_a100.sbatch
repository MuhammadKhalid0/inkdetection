#!/bin/bash -l
#SBATCH --job-name=nnf_t192
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-1
#SBATCH --export=NONE

# 2026-08-10 nnFormer FINAL batch — everything the plan has left.
#   0-1  SR-PB confirmation seeds 43/44 at t192 (s42 gave +0.0127 raw, above the
#        +-0.008 floor: a single-seed delta may only promote, never be claimed)
#   2-4  nnf_c1_t256 seeds 42/43/44 = best N2 config (S + dense stride-64) + SR-PB
#        -> SR-PB's deployment-scale gate AND, if it holds, the reported final row.
# CSSA excluded: failed its pre-registered 2x2 gate (plan says do not spend seeds).
# Labels ink_reg (reg1). 12h wall (measured max so far: 7.7h at t256-dense).

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

RUNS=(n_base_t192_r1_s43 n_base_t192_r1_s44)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CFG="$NNF_DIR/configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py engine/losses.py \
    data/dataset.py models/wrapper.py models/shared.py \
    "$NNF_DIR"/src/nnformer_ink/*.py
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

echo "== test-segment prediction =="
srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE $RUN (ink_reg, t256) =="
