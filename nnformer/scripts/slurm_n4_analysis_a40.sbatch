#!/bin/bash -l
#SBATCH --job-name=nnf_n4
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# 2026-07-20 nnFormer Phase N4 — the shared analysis layer (nnformer_plan.md §5 N4).
# Submitted with a dependency on the N3+contributions array, so it fires as soon
# as training finishes. All probes are model-agnostic and already written for
# UNETR++; running them on nnFormer is what makes the cross-model figures.
#   1. TTA rescore (8-fold dihedral) of the benchmark config -> headline row
#   2. depth-offset stress (G1) — SR-PB's PRIMARY readout (plan §2.1), not dF0.5
#   3. error anatomy — per-stratum recall vs UNETR++'s 32/49/18.7% (plan E9)
#   4. ERF — with CSSA on/off = the mechanism figure for C2 (plan §5 N4.2)
#   5. frequency occlusion — morphology-vs-crackle, cross-model
#   6. contribution gate report (CSSA 2x2 + SR-PB stress verdicts)

unset SLURM_EXPORT_ENV
module load python
module load cuda

NNF_DIR="$WORK/nnFormer"
BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
export INK_OUTPUT_ROOT="$NNF_DIR/outputs"
OUT="$NNF_DIR/outputs"
mkdir -p "$NNF_DIR/probes"

BENCH_RUN=nnf_s_d64_t256_s42     # best N2 config (benchmark row)
have () { [ -f "$OUT/$1/best.pt" ]; }

echo "===== [1/6] TTA rescore ====="
for r in nnf_s_d64_t256_s42 nnf_s_d64_t256_s43 nnf_s_d64_t256_s44 \
         nnf_c1_t256_s42 nnf_c1_t256_s43 nnf_c1_t256_s44; do
    have "$r" && (srun python -m scripts.predict_val_tta --runs "$r" --root "$OUT" || true)
done

echo "===== [2/6] G1 depth-offset stress (SR-PB primary readout) ====="
STRESS=""
for r in n_base_t192_r1_s42 n_srpb_t192_r1_s42 n_srpb_t192_r1_s43 n_cssa_t192_r1_s42; do
    have "$r" && STRESS="${STRESS}${STRESS:+,}$r"
done
[ -n "$STRESS" ] && srun python -m scripts.stress_zshift --runs ${STRESS//,/ } \
    --root "$OUT" --deltas -8 -6 -4 -2 0 2 4 6 8 || echo "(stress skipped)"

echo "===== [3/6] error anatomy (per-stratum recall, cross-model axis) ====="
for r in "$BENCH_RUN" nnf_c1_t256_s42 n_base_t192_r1_s42 n_srpb_t192_r1_s42 n_cssa_t192_r1_s42; do
    have "$r" && python -m scripts.interp_error_anatomy --root "$OUT" --run "$r" \
        --out "$NNF_DIR/probes" || true
done

echo "===== [4/6] ERF (CSSA on/off = C2 mechanism figure) ====="
ERF=""
for r in n_base_t192_r1_s42 n_cssa_t192_r1_s42 "$BENCH_RUN"; do
    have "$r" && ERF="${ERF}${ERF:+,}$r"
done
ERFSPEC=""
for r in n_base_t192_r1_s42 n_cssa_t192_r1_s42; do
    have "$r" && ERFSPEC="${ERFSPEC}${ERFSPEC:+,}$OUT:$r:val_reg_t192.csv"
done
for r in "$BENCH_RUN" nnf_c1_t256_s42; do
    have "$r" && ERFSPEC="${ERFSPEC}${ERFSPEC:+,}$OUT:$r:val_reg_t256.csv"
done
[ -n "$ERFSPEC" ] && srun python -m scripts.interp_erf --models "$ERFSPEC" \
    --out "$NNF_DIR/probes" || echo "(erf skipped)"

echo "===== [5/6] frequency occlusion ====="
for r in "$BENCH_RUN" nnf_c1_t256_s42; do
    have "$r" && srun python -m scripts.interp_freqband --root "$OUT" \
        --run "$r" --out "$NNF_DIR/probes" || true
done

echo "===== [6/6] contribution gate report ====="
python "$NNF_DIR/scripts/collect_results.py" || true

echo "== DONE N4 analysis =="
