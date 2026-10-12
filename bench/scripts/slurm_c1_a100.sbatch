#!/bin/bash -l
#SBATCH --job-name=ink5_c1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-3
#SBATCH --export=NONE

# 2026-07-18 final combo round (plan §14.3): c1_t256 = t256 lean + u1 ignore-band
# + no z_flip. Tasks 0-2 = seeds 42/43/44 (60 ep); task 3 = e90 variant (seed 42,
# 90-ep cosine — closes the 'is 60 epochs enough' question on the final recipe).
# Outputs -> outputs5/.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs5"

RUNS=(c1_t256_s42 c1_t256_s43 c1_t256_s44 c1_t256_e90_s42)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
CFG="configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
VALSEG="20241113090990"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile engine/train.py engine/infer.py engine/losses.py \
    data/dataset.py data/segments.py models/shared.py models/wrapper.py \
    scripts/postprocess.py scripts/plot_train_curves.py
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
p = "configs/${RUN}.yaml"
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

srun python -m scripts.predict --run "$RUN"
PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6 || true

echo "== DONE $RUN =="
