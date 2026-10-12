#!/bin/bash
# 2026-07-20: rebuild training-side data artifacts for the ink_reg2 labels.
# Run AFTER the reg2 QC figures are reviewed. CPU only, ~minutes.
# New files only — nothing historical is overwritten:
#   manifests: train_reg2_t192.csv / val_reg2_t192.csv / *_t256 / dense _t256_d64
#   trimaps:   proc/<seg>/unc_reg2.npy (+ unc_reg2_stats.json), r=4, NO dy-widening
# Training configs then use: data.label_file: ink_reg2.npy,
#   data.manifest_suffix: _reg2_t192 (etc.), data.unc_file: unc_reg2.npy
set -e
BENCH="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH"
source "$VENV/bin/activate"

echo "== manifests (t192, t256, dense t256 stride 64) =="
python -m data.tiling --tile 192 --stride 96  --label-file ink_reg2.npy --suffix _reg2_t192
python -m data.tiling --tile 256 --stride 128 --label-file ink_reg2.npy --suffix _reg2_t256
python -m data.tiling --tile 256 --stride 64  --label-file ink_reg2.npy --suffix _reg2_t256_d64

echo "== trimaps (r=4, no residual-dy widening — reg2 corrected those offsets) =="
python -m scripts.make_uncertainty --r 4 --label-file ink_reg2.npy \
    --out-file unc_reg2.npy --no-dy-widen

echo "DONE — reg2 data layer ready"
