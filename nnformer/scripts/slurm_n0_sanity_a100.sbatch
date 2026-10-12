#!/bin/bash -l
#SBATCH --job-name=nnf_n0_sanity
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

# 2026-07-19 nnFormer Phase N0 (nnformer_plan.md §5): sanity run + GPU bench.
# 1) n0_sanity_t64 — 2 epochs end-to-end through the bench pipeline;
# 2) t256 batch-4 memory probe (one fwd+bwd step per capacity);
# 3) bench.py — params/GFLOPs/ms/peak-mem for S/M/F at t192/t256 -> docs/bench_table.md.
# Output root: $WORK/nnFormer/outputs (NEW tree — the vesuvius outputs* history is
# never written by nnFormer jobs).

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

RUN=n0_sanity_t64
CFG="$NNF_DIR/configs/${RUN}.yaml"

set -e
python -m py_compile engine/train.py engine/infer.py engine/losses.py \
    data/dataset.py models/wrapper.py models/shared.py \
    "$NNF_DIR"/src/nnformer_ink/*.py
python "$NNF_DIR/tests/test_smoke.py"
python scripts/smoke_test.py --config "$CFG" --device cpu

echo "== [1/3] sanity training run =="
srun python -m engine.train --config "$CFG"
python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"

echo "== [2/3] t256 batch-4 memory probe (fwd+bwd, bf16) =="
python - <<'PYEOF'
import sys, torch
sys.path.insert(0, "/home/woody/iwi5/iwi5401h/nnFormer/src")
from nnformer_ink import build_nnformer_ink
for cap in ("S", "M", "F"):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m = build_nnformer_ink(img_size=(32, 256, 256), capacity=cap).cuda()
        x = torch.randn(4, 1, 32, 256, 256, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x); loss = y.float().mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  nnF-{cap} t256 batch4 fwd+bwd OK  peak {peak:.1f} GB")
        del m, x, y, loss
    except torch.cuda.OutOfMemoryError:
        print(f"  nnF-{cap} t256 batch4 OOM -> use batch 2 + grad-accum 2 (plan §8)")
        torch.cuda.empty_cache()
PYEOF

echo "== [3/3] efficiency bench =="
python "$NNF_DIR/scripts/bench.py" --caps S,M,F --tiles 192,256 --device cuda \
    --out "$NNF_DIR/docs/bench_table.md"

echo "== DONE N0 sanity =="
