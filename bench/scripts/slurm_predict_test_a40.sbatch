#!/bin/bash -l
#SBATCH --job-name=ink_predict_test
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# Predict the TEST segment (flatboi, no labels) with a trained checkpoint.
# Stacks the 65 raw tifs into proc_infer/<seg>/ (one-time, ~36 GB, slow Vault
# read) if not already present, then runs Gaussian-blended sliding-window
# inference and saves the full-res probability map.  RUN=<run name> selects the
# checkpoint (default a2bs = best rescored model).

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

RUN="${RUN:-a2bs}"
SEG="${SEG:-20251115002740-auto_grown_20251115002740308_0_flatboi}"

# 1) prepare (stack tifs) if not already done
if [ ! -f "$WORK/vesuvius/data/proc_infer/$SEG/vol.npy" ]; then
  echo "== preparing test segment (stacking 65 tifs) =="
  python -m scripts.prepare_infer --segment "$SEG"
else
  echo "== proc_infer volume already present, skipping prepare =="
fi

# 2) predict
echo "== predicting with run=$RUN =="
srun python -m scripts.predict --run "$RUN" --seg "$SEG"
