#!/bin/bash -l
#SBATCH --job-name=ink_abl_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# Generic single-config ablation runner (component ablations vs the a2bs_t192
# reference).  Usage:  sbatch scripts/slurm_abl_a40.sbatch configs/abl_stem.yaml
# Steps: (manifest build if missing) -> smoke -> train -> postprocess.

unset SLURM_EXPORT_ENV
module load python
module load cuda

BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"

CFG="$1"
test -f "$CFG" || { echo "usage: sbatch $0 configs/<abl>.yaml"; exit 1; }
RUN=$(python -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['output']['run_name'])")
SUFFIX=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['data'].get('manifest_suffix',''))")
LABEL=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['data'].get('label_file','ink.npy'))")
TILE=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['data']['tile'])")

set -e
echo "== ablation $RUN (config $CFG) =="
python -m py_compile models/shared.py models/unetrpp/transformerblock.py engine/train.py

# manifest build if missing (e.g. abl_labels needs train_t192.csv from ink.npy)
if [ ! -f "$WORK/vesuvius/data/manifests/train${SUFFIX}.csv" ]; then
  echo "building manifest train${SUFFIX}.csv (tile $TILE, label $LABEL)"
  python -m data.tiling --tile "$TILE" --stride $((TILE/2)) --min-mask 0.5 \
      --label-file "$LABEL" --suffix "$SUFFIX"
fi

python scripts/smoke_test.py --config "$CFG" --device cpu
srun python -m engine.train --config "$CFG"
python -m scripts.postprocess --run "$RUN"

echo "== DONE $RUN =="
