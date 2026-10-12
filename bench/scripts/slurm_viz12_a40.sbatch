#!/bin/bash -l
#SBATCH --job-name=ink_viz12_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=6:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-11
#SBATCH --export=NONE

# Test-segment prediction maps + strip figures for the 12 recently finished runs
# (8 ablations + 2 first-stage seeds + 2 r4-recipe refiner seeds), at 6 taus.
# Val 4-panels are rendered separately on the login node (CPU-only).

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

RUNS=(abl_stem abl_da abl_surfrel abl_perpix abl_warpstab abl_labels \
      abl_loss_recall abl_loss_prec a2bs_t192_s43 a2bs_t192_s44 r4_s43 r4_s44)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
OUT="$WORK/vesuvius/outputs/$RUN"
TESTSEG="20251115002740-auto_grown_20251115002740308_0_flatboi"

set -e
python -m py_compile scripts/predict.py scripts/refine_predict.py scripts/visualize.py engine/infer.py
echo "== task $SLURM_ARRAY_TASK_ID run=$RUN =="

if [[ "$RUN" == r4_s4* ]]; then
  # refiner seeds: refine the full model's existing test map (same input as r4)
  TEST_PROB_IN=$(ls "$WORK/vesuvius/outputs/a2bs_t192"/test_prob_*.npy | head -1)
  test -n "$TEST_PROB_IN" || { echo "missing a2bs_t192 test map"; exit 1; }
  PROB="$OUT/test_prob_refined.npy"
  if [ ! -f "$PROB" ]; then
    srun python -m scripts.refine_predict --refiner "$RUN" --prob "$TEST_PROB_IN" --out "$PROB"
  fi
else
  # first-stage models: fresh 3D inference on the test segment
  PROB=$(ls "$OUT"/test_prob_*.npy 2>/dev/null | head -1)
  if [ -z "$PROB" ]; then
    srun python -m scripts.predict --run "$RUN"
    PROB=$(ls "$OUT"/test_prob_*.npy | head -1)
  fi
fi

echo "== test map: $PROB =="
for TAU in 0.6 0.65 0.7 0.75 0.8 0.85; do
  python scripts/visualize.py --prob "$PROB" --seg "$TESTSEG" --no-gt --tau "$TAU" --downsample 6
done
echo "== DONE $RUN =="
