#!/bin/bash -l
#SBATCH --job-name=ink4_t256
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-2
#SBATCH --export=NONE

# 2026-07-17: t256 trial, 3 seeds (plan.md §14.2). Lean + tile-level surfrel
# (per t192 ablations), bf16, batch 4 (same batch/steps-per-epoch as t192).
# Decision rule: adopt t256 iff mean raw F0.5 > lean_t192 mean + 0.008;
# also read LEFT-WORD recall — that is where context pays.
# Outputs -> outputs4/ (isolated from outputs/, outputs2/, outputs3/).

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs4"

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN="t256_lean_s${SEED}"
CFG="configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py data/dataset.py \
    data/segments.py models/shared.py models/wrapper.py \
    scripts/postprocess.py scripts/plot_train_curves.py
echo "== task $SLURM_ARRAY_TASK_ID run=$RUN cfg=$CFG =="

python scripts/smoke_test.py --config "$CFG" --device cpu

# batch-4 first; on failure (OOM is the expected mode) retry once at batch 2.
set +e
srun python -m engine.train --config "$CFG"
RC=$?
set -e
if [ $RC -ne 0 ]; then
    echo "== train exited rc=$RC — retrying once at batch 2 =="
    python - <<PYEOF
import yaml
p = "configs/${RUN}.yaml"
c = yaml.safe_load(open(p))
c["train"]["batch"] = 2
open(p, "w").write("# batch auto-fallback 4->2 after rc!=0 on first attempt\n" + yaml.safe_dump(c, sort_keys=False))
PYEOF
    rm -rf "$OUT"          # discard the partial first attempt inside outputs4 only
    srun python -m engine.train --config "$CFG"
fi

python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"

TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg "$VALSEG" \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE $RUN =="
