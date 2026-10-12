#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE
#
# Resumable, self-continuing training chain for ONE run (2026-09-12).
#   submit:  cd $WORK/segFormer/scripts && sbatch --job-name=sfc_<run> slurm_chain_a100.sbatch <run>
#
# Safety properties, each deliberate:
#   * NO `rm -rf`, and NO batch halving anywhere (both are incompatible with resume).
#   * Refuses any config that does not set train.resume: true.
#   * The continuation is queued at the START of a training chunk (afterany), so even a hard
#     wall-clock kill continues the chain. It is cancelled once the run is fully finished.
#   * A crash (non-zero exit) marks the chain crashed and cancels the continuation, leaving
#     every artifact untouched for inspection. It never retries blindly.
#   * chain_guard.py stops the chain on: crashed / complete / max_chunks / no_progress.
#   * train.time_limit_min makes each chunk stop cleanly BETWEEN epochs before the wall.

unset SLURM_EXPORT_ENV
module load python
module load cuda

RUN="${1:?usage: sbatch --job-name=sfc_<run> slurm_chain_a100.sbatch <run_name>}"
SF_DIR="$WORK/segFormer"
BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
SELF="$SF_DIR/scripts/slurm_chain_a100.sbatch"
MAX_CHUNKS=6
VALSEG="20241113090990"

cd "$BENCH_DIR"
source "$WORK/vesuvius/.venv/bin/activate"
export INK_OUTPUT_ROOT="$SF_DIR/outputs"
CFG="$SF_DIR/configs/${RUN}.yaml"
OUT="$INK_OUTPUT_ROOT/$RUN"
GUARD=(python "$SF_DIR/scripts/chain_guard.py")

set -euo pipefail
echo "== chain job ${SLURM_JOB_ID} run=${RUN} host=$(hostname) $(date) =="

python - "$CFG" "$RUN" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
assert c["train"].get("resume") is True, "REFUSING: chained jobs require train.resume: true"
assert c["output"]["run_name"] == sys.argv[2], "REFUSING: run_name does not match the config file"
assert c["output"]["dir"].rstrip("/").endswith("segFormer/outputs"), "REFUSING: unexpected output.dir"
print("config ok: resume on, run_name matches, time_limit_min =", c["train"].get("time_limit_min"))
PY

ACTION=$("${GUARD[@]}" decide --out "$OUT" --max-chunks "$MAX_CHUNKS")
echo "== guard decision: ${ACTION} =="
case "$ACTION" in
  stop:*) echo "== chain ends: ${ACTION} =="; "${GUARD[@]}" status --out "$OUT"; exit 0 ;;
esac

python -m py_compile engine/train.py engine/resume.py engine/infer.py engine/losses.py \
    data/dataset.py models/wrapper.py models/shared.py "$SF_DIR"/src/segformer3d_ink/*.py

NEXT=""
if [ "$ACTION" = "train" ]; then
  python scripts/smoke_test.py --config "$CFG" --device cpu
  "${GUARD[@]}" begin --out "$OUT"
  NEXT=$(sbatch --parsable --chdir="$SF_DIR/scripts" --job-name="sfc_${RUN}" \
         --dependency="afterany:${SLURM_JOB_ID}" "$SELF" "$RUN")
  echo "== continuation queued as ${NEXT} (cancelled automatically if not needed) =="

  set +e
  srun python -m engine.train --config "$CFG"
  RC=$?
  set -e
  if [ "$RC" -ne 0 ]; then
    echo "== TRAINING FAILED rc=${RC}: marking crashed, cancelling ${NEXT}, touching NO files =="
    "${GUARD[@]}" crash --out "$OUT" --rc "$RC"
    scancel "$NEXT" || true
    exit "$RC"
  fi
  if [ ! -f "$OUT/train_done.json" ]; then
    echo "== chunk ended cleanly before finishing; ${NEXT} will resume from last.pt =="
    exit 0
  fi
fi

echo "== training finished: consistency check, then post-processing =="
"${GUARD[@]}" consistent --out "$OUT"
python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"
TAU=$(python -c "import json;print(json.load(open('$OUT/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$OUT/val_prob_best.npy" --seg "$VALSEG" \
    --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true

# only now is the continuation provably unnecessary
if [ -n "$NEXT" ]; then scancel "$NEXT" || true; fi
"${GUARD[@]}" status --out "$OUT"
echo "== DONE ${RUN} $(date) =="
