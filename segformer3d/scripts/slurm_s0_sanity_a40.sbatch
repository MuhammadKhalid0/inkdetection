#!/bin/bash -l
#SBATCH --job-name=sf_s0
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
module load cuda

SF_DIR="$WORK/segFormer"
BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
cd "$BENCH_DIR"
source "$WORK/vesuvius/.venv/bin/activate"
export INK_OUTPUT_ROOT="$SF_DIR/outputs"
mkdir -p "$INK_OUTPUT_ROOT"

set -e
echo "=== [1] unit tests (exact-reduction proofs, param counts, geometry) ==="
python "$SF_DIR/tests/test_smoke.py"

echo "=== [2] measured efficiency table on the real GPU ==="
python "$SF_DIR/scripts/bench.py" --device cuda --out "$SF_DIR/docs/bench_table.md" \
    --tiles 192 256 320 || true

echo "=== [3] t256 batch-4 memory probe (fwd+bwd, bf16) BEFORE committing the schedule ==="
python - <<'PYEOF'
import sys, torch
sys.path.insert(0, "/home/woody/iwi5/iwi5401h/segFormer/src")
from segformer3d_ink import build_segformer3d_ink as B
for cap, tile, bs in [("S",256,4), ("M",256,4), ("L",256,4), ("S",320,2), ("M",320,2)]:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m = B(img_size=(32,tile,tile), capacity=cap).cuda().train()
        x = torch.randn(bs,1,32,tile,tile, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x).mean().backward()
        print(f"  SF-{cap} t{tile} batch {bs}: OK, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB")
        del m, x
    except RuntimeError as e:
        print(f"  SF-{cap} t{tile} batch {bs}: OOM/ERROR -- {str(e)[:90]}")
    torch.cuda.empty_cache()
PYEOF

echo "=== [4] 2-epoch pipeline sanity (train -> sweep -> postprocess -> figs) ==="
CFG="$SF_DIR/configs/s0_sanity_t64.yaml"
python scripts/smoke_test.py --config "$CFG" --device cpu
srun python -m engine.train --config "$CFG"
python -m scripts.postprocess --run s0_sanity_t64
python scripts/plot_train_curves.py --run s0_sanity_t64
echo "=== S0 COMPLETE ==="
