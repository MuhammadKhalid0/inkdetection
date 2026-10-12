#!/bin/bash -l
#SBATCH --job-name=nnf_inkal
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# 2026-08-10 nnFormer Inkalyzer depth attribution (plan §5 N4 item 1) — the last
# missing probe, closing the nnFormer analysis layer.
#
# Two questions it answers:
#  (a) TRUST / cross-model figure: does nnFormer also concentrate its depth
#      attribution on the physical surface band z27-33, as all 15 UNETR++ models
#      did (surf% 45-60, ~0 at window edges)? A second architecture agreeing is
#      an architecture-independent statement about the DATA.
#  (b) SR-PB MECHANISM: SR-PB is literally a depth-axis prior, so the depth
#      profile is its natural probe. For UNETR++, surface-relative handling was
#      INVISIBLE to this metric (abl_surfrel entropy 0.899 == 0.899, peak z32 ==
#      z32) because it acted per-pixel/spatially. If nnFormer's SR-PB *does*
#      move the profile, that is a genuinely new mechanism result and resolves
#      the limitation stated in Inklayzer_UNTER++.md §3.3.
#
# Runs are all t192 (comparable group — cross-tile entropy is a METHOD ARTEFACT,
# per Inklayzer_UNTER++.md §3.5: larger tiles average more heterogeneous pixels
# and admit flatter gates. The two t256 models are included but MUST be read as
# their own group, never against the t192 rows.)

unset SLURM_EXPORT_ENV
module load python
module load cuda

NNF_DIR="$WORK/nnFormer"
BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
VENV="$WORK/vesuvius/.venv"
cd "$BENCH_DIR"
source "$VENV/bin/activate"
# makes inkalyzer_run.py read outputs/<run>/best.pt and write
# outputs/<run>/inkalyzer/ inside the nnFormer tree
export INK_OUTPUT_ROOT="$NNF_DIR/outputs"

RUNS="n_base_t192_r1_s42,n_srpb_t192_r1_s42,n_cssa_t192_r1_s42,nnf_s_d64_t256_s42,nnf_c1_t256_s42"

set -e
python -m py_compile scripts/inkalyzer_run.py models/wrapper.py \
    "$NNF_DIR"/src/nnformer_ink/*.py
echo "== Inkalyzer on: $RUNS =="

srun python -m scripts.inkalyzer_run --runs "$RUNS" \
    --label-file ink_reg.npy --n-tiles 24 --n-steps 100 --n-3d 3 \
    --compare-dir "$NNF_DIR/probes/inkalyzer_compare"

echo "== DONE Inkalyzer =="
ls -la "$NNF_DIR"/probes/inkalyzer_compare/ 2>/dev/null || true
