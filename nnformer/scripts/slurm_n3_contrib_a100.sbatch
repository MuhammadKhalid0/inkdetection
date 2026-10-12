#!/bin/bash -l
#SBATCH --job-name=nnf_n3c
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-7
#SBATCH --export=NONE

# 2026-07-20 nnFormer Phase N3 + contributions (port step 2), all on ink_reg
# (reg1 -- the UNETR++ yardstick). Task order is deliberate because the account
# limit serializes GPU work (1 concurrent):
#   0-2  t192 stock reference + CSSA + SR-PB screens (~5h each, cheapest first)
#   3-4  benchmark-row seeds 43/44 of the best N2 config (S + dense stride-64)
#   5-6  CSSA 2x2 low rung (stock/CSSA at t128) -> plan Table N-V
#   7    e90 budget test (N2 flag: nnf_s_t256 peaked at epoch 59/60)
# All runs are capacity S (~20M, 20.1GB peak at t256 per the N0 probe), so the
# 80GB constraint is dropped here to widen the eligible node pool.

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

RUNS=(n_base_t192_r1_s42 n_cssa_t192_r1_s42 n_srpb_t192_r1_s42 \
      nnf_s_d64_t256_s43 nnf_s_d64_t256_s44 \
      n_base_t128_r1_s42 n_cssa_t128_r1_s42 nnf_s_d64_t256_e90_s42)
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
