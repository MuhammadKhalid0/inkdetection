# vesuvius-ink-bench — A0 (vanilla UNETR++ baseline)

Ink detection on Scroll 5 (PHerc. 0172). 3D surface-volume tile in → 2D ink mask out.
This is **A0**: pure published-style UNETR++ (EPA, medical embed, mean depth collapse,
no texture stem) — the Phase-0 spine and the honest baseline every later ablation
(A0s → A2b) builds on.

## Data & split (authoritative, per Fei Wu)
- 9 labelled segments, all Scroll 5. Labels are real ground truth (near-binary).
- **Train** = 8 segments · **Val** = `20241113090990` · **Test** = `…flatboi` (no labels → qualitative only).
- Quantitative F0.5 is selected/reported on **val**; test is qualitative.

## Layout
```
models/unetrpp/   lifted UNETR++ net code + unetr_pp_ink.py (adapted, correct sizes)
models/shared.py  TextureStem / DepthCollapse / Head  (A0 uses mean collapse only)
models/wrapper.py build_model(cfg) -> UnifiedModel: stem→backbone→collapse→head
data/prepare.py   Vault npy → aligned vol/ink/mask on $WORK  (+ meta.json)
data/tiling.py    64×64 stride-32 tile manifests (train.csv, val.csv)
data/dataset.py   InkDataset + ink-aware WeightedRandomSampler
engine/train.py   training loop (AMP, warmup-cosine, val F0.5 checkpoint)
engine/infer.py   sliding-window + Gaussian blend → full prob map
engine/eval.py    F0.5/Dice/P/R + τ sweep (over valid mask)
scripts/smoke_test.py   PRE-FLIGHT validation (run before every training)
scripts/slurm_a0.sbatch A100 job
```

## How to run (in order)

```bash
source $WORK/vesuvius/.venv/bin/activate
cd $WORK/vesuvius/vesuvius-ink-bench

# 0. VALIDATE FIRST (no GPU, no data needed) — must print "ALL CHECKS PASSED"
python scripts/smoke_test.py --config configs/a0.yaml --device cpu

# 1. Preprocess Vault → $WORK (one-time, ~87 GB write; slow Vault read).
#    Dry-run first to sanity-check alignment (no writes):
python -m data.prepare --dry-run
python -m data.prepare                      # writes data/proc/<seg>/{vol,ink,mask}.npy

# 2. Build tile manifests
python -m data.tiling --tile 64 --stride 32 --min-mask 0.5

# 3. Train (GPU). On Alex submit the job:
sbatch scripts/slurm_a0.sbatch
#    or interactively on a GPU node:
python -m engine.train --config configs/a0.yaml
```

Outputs → `$WORK/vesuvius/outputs/a0/`: `best.pt`, `train_log.csv`,
`val_prob_best.npy`, `config_used.json`.

## Environment
- venv: `$WORK/vesuvius/.venv` (torch 2.6.0+cu124, torchvision, monai, timm, einops).
- The CUDA torch wheel bundles its CUDA libs, so it imports on the login node and
  uses the GPU on A100 nodes without a `module load cuda`. If a node complains,
  `module load cuda` and retry, or reinstall torch for that node's CUDA.

## Key adaptation notes (why this differs from the repo)
- `UNETR_PP_Ink` recomputes encoder `input_size` and decoder `out_size` from the
  actual (32,64,64) patch. The published synapse class hardcodes medical
  (64,128,128) token counts and would crash on our input (EPA `E/F` projection +
  positional-embedding shape mismatch).
- Inherits `nn.Module` (not `SegmentationNetwork`) → no nnU-Net/batchgenerators deps.
- Backbone returns the last **decoder feature volume**; the wrapper does the
  depth→2D collapse + 2D head (unified interface for all future models).

## Validated (2026-07-01)
- Smoke test: 23,121,513 params; (2,1,32,64,64)→(2,1,64,64); finite loss+grads.
- Stage res [(16,16,16),(8,8,8),(4,4,4),(2,2,2)], tokens [4096,512,64,8], bottleneck (2,2,2).
- `data.prepare --dry-run`: all 9 segments read; alignment crops correct.
