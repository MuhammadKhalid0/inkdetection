#!/bin/bash -l
#SBATCH --job-name=sf_s4
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# Phase S4 — the shared analysis layer (inference-only). All probes are model-agnostic
# and already written; only the run names and the output root change.

unset SLURM_EXPORT_ENV
module load python
module load cuda

SF_DIR="$WORK/segFormer"
BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
cd "$BENCH_DIR"
source "$WORK/vesuvius/.venv/bin/activate"
export INK_OUTPUT_ROOT="$SF_DIR/outputs"
OUT="$SF_DIR/outputs"
PROBES="$SF_DIR/probes"
mkdir -p "$PROBES"

have () { [ -f "$OUT/$1/best.pt" ]; }

echo "=== [1] ERF — the decisive third point of the ERF law (prediction 2) ==="
# also PDK on/off at the same tile = C1's mechanism figure
M=""
for r in sf_base_t128_s42:val_reg_t128.csv sf_base_t192_s42:val_reg_t192.csv \
         sf_base_t256_s42:val_reg_t256.csv sf_pdk_t192_s42:val_reg_t192.csv; do
  run="${r%%:*}"; man="${r##*:}"
  have "$run" && M="$M,$OUT:$run:$man"
done
M="${M#,}"
[ -n "$M" ] && srun python -m scripts.interp_erf --models "$M" --out "$PROBES" || true

echo "=== [2] Inkalyzer depth attribution (third replication + C2's table-vs-probe) ==="
R=""
for run in sf_base_t192_s42 sf_dcm_abs_t192_s42 sf_dcm_surf_t192_s42 sf_base_t256_s42; do
  have "$run" && R="$R,$run"
done
R="${R#,}"
[ -n "$R" ] && srun python -m scripts.inkalyzer_run --runs "$R" --label-file ink_reg.npy \
    --n-tiles 24 --n-steps 100 --n-3d 3 --compare-dir "$PROBES/inkalyzer_compare" || true

echo "=== [3] G1 depth-offset stress — BAND vs BAND from the start (seed-42 lesson) ==="
S=""
for run in sf_base_t192_s42 sf_dcm_abs_t192_s42 sf_dcm_surf_t192_s42; do
  have "$run" && S="$S $run"
done
[ -n "$S" ] && srun python -m scripts.stress_zshift --runs $S --root "$OUT" \
    --deltas -8 -6 -4 -2 0 2 4 6 8 || true

echo "=== [4] frequency band-stop (C3/OSR mechanism readout: 8-16px and 16-32px) ==="
for run in sf_base_t192_s42 sf_osr_t192_s42 sf_base_t256_s42; do
  have "$run" && srun python -m scripts.interp_freqband --root "$OUT" --run "$run" \
      --out "$PROBES" || true
done

echo "=== [5] error anatomy — per-stratum recall + C4/DSM FP-island statistics ==="
for run in sf_base_t192_s42 sf_base_t256_s42 sf_dsm_t192_s42 sf_pdk_t192_s42; do
  have "$run" && python -m scripts.interp_error_anatomy --root "$OUT" --run "$run" \
      --out "$PROBES" || true
done

echo "=== [6] resolution transfer (SegFormer3D-only; the PE-free claim) ==="
have sf_base_t192_s42 && srun python -m scripts.stress_tile --runs sf_base_t192_s42 \
    --root "$OUT" --tiles 128 192 256 320 384 || true
have sf_nonorm_t192_s42 && srun python -m scripts.stress_tile --runs sf_nonorm_t192_s42 \
    --root "$OUT" --tiles 128 192 256 320 384 || true
have sf_base_t192_s42 && srun python -m scripts.stress_tile --runs sf_base_t192_s42 \
    --root "$OUT" --tiles 128 192 256 320 384 --no-bn-runstats || true

echo "=== [7] TTA + IoU backfill ==="
for run in sf_base_t256_s42 sf_s_d64_t256_s42; do
  have "$run" && srun python -m scripts.predict_val_tta --runs "$run" --root "$OUT" || true
done
python -m scripts.rescore_iou --root "$OUT" || true

echo "=== [8] gates ==="
python "$SF_DIR/scripts/collect_results.py"
echo "=== S4 COMPLETE ==="
