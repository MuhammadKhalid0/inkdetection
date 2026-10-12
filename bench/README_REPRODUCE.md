# README — How to Reproduce

Copy-paste guide to reproducing any result in this project, from a clean checkout to the final
headline number. For *what* the pipeline does and *why*, see
`plan(md_files)/PIPELINE_EXPLAINER.md` and `plan(md_files)/UNTER_Plus_Plus_Final.md`. This file is
only about *running the commands*.

---

## 0. Environment setup

```bash
# on NHR@FAU Alex — allocate an interactive session or just prep on the login node
cd $WORK/vesuvius
python3 -m venv .venv
source .venv/bin/activate

# CUDA-matched torch first (bundles its own CUDA libs — no `module load cuda` needed
# to import; load it only if a compute node complains at runtime)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# then everything else
cd vesuvius-ink-bench
pip install -r requirements.txt
```

`requirements.txt` pulls: numpy, pyyaml, pillow, einops, timm==0.6.13, monai==1.3.0,
tensorboard, tqdm, scikit-image, scipy.

**Every session starts with:**
```bash
module load python
module load cuda
source $WORK/vesuvius/.venv/bin/activate
cd $WORK/vesuvius/vesuvius-ink-bench
```

**Sanity check the install** (no GPU, no data needed — must print `ALL CHECKS PASSED`):
```bash
python scripts/smoke_test.py --config configs/a0.yaml --device cpu
```

---

## 1. One-time data preparation

Raw segments live on Vault (`/home/vault/iwi5/iwi5401h/new_labeled/`, `new_inference/`). This step
reads them once and writes fast, aligned arrays to `$WORK/vesuvius/data/proc/`.

```bash
# dry-run first — prints shapes/crop sizes, writes nothing
python -m data.prepare --dry-run

# the real pass (~87 GB write; slow because Vault reads are slow, not because of compute)
python -m data.prepare
```

This produces `data/proc/<segment>/{vol.npy, ink.npy, mask.npy, meta.json}` for all 9 labelled
segments, and (separately) `data/prepare_infer.py` does the same for the unlabelled test segment
into `data/proc_infer/` if you need to re-derive that too:
```bash
python -m data.prepare_infer
```

### 1.1 Label registration (fixes the 20–54 px label misalignment)

Run once, after `data.prepare`. Writes a new file `ink_reg.npy` next to `ink.npy` in every
segment's proc folder — **the original `ink.npy` is never modified.**

```bash
python -m scripts.register_labels
# or a subset while testing:
python -m scripts.register_labels --segments 20241113090990
```

Every training config in this project sets `data.label_file: ink_reg.npy`. If you ever see
`data.label_file: ink.npy` in a config, that run is a deliberate ablation against the
*unregistered* labels (e.g. `abl_labels`, `abl64_labels`) — not a mistake.

### 1.2 Tile manifests (one per tile size you plan to train at)

```bash
# tile 64  (stride 32)
python -m data.tiling --tile 64  --stride 32  --min-mask 0.5 --label-file ink_reg.npy --suffix _reg
# tile 128 (stride 64)
python -m data.tiling --tile 128 --stride 64  --min-mask 0.5 --label-file ink_reg.npy --suffix _reg_t128
# tile 192 (stride 96)  -- the main workhorse size
python -m data.tiling --tile 192 --stride 96  --min-mask 0.5 --label-file ink_reg.npy --suffix _reg_t192
# tile 256 (stride 128) -- the final reported model's tile size
python -m data.tiling --tile 256 --stride 128 --min-mask 0.5 --label-file ink_reg.npy --suffix _reg_t256
# tile 320 (stride 160) -- the saturation-check rung
python -m data.tiling --tile 320 --stride 160 --min-mask 0.5 --label-file ink_reg.npy --suffix _reg_t320
```
Each call writes `data/manifests/train<suffix>.csv` + `val<suffix>.csv`. A config's
`data.manifest_suffix` key tells you which one it needs — build that one before training with it
(training will fail loudly with a missing-file error if you forget).

### 1.3 Extra manifests needed by specific experiments

```bash
# word-box (annotation-anchored) manifest — needed only by configs/s2_wordbox_*.yaml,
# s3_wordbox_*.yaml, s4_wordbox_*.yaml (this variant was TESTED AND REJECTED — see the
# final report — but the configs still reference this file if you want to reproduce it)
python -m scripts.make_word_manifests --tile 192 --stride 96 --jitter 48 \
    --per-area 9216 --neg-margin 32 --min-mask 0.5 \
    --label-file ink_reg.npy --suffix _wb_t192 --seed 0

# per-pixel uncertainty trimap — needed by configs/u1_ignore_*.yaml, u2_soft3_*.yaml,
# and any c1_t256_* config (the final reported model uses u1's masked loss)
python -m scripts.make_uncertainty --r 4 --label-file ink_reg.npy
```
(`make_uncertainty` writes `proc/<seg>/unc.npy` — a data artifact, not a manifest — for the 8
training segments; it does not need to be re-run per tile size.)

### 1.4 LOO (leave-one-out) manifests — only needed for the refiner's OOF training

```bash
python -m scripts.make_loo --tile 192 --stride 96 --label-file ink_reg.npy
```
Writes 8 fold manifests `train_reg_t192_loo{0..7}.csv` (each holds out one of the 8 train
segments) — used to train the 8 first-stage LOO models the refiner's OOF variant (`r2`, `r4`)
consumes.

---

## 2. Training one run

Every run is fully specified by one YAML config. Always smoke-test before submitting to the
cluster — it costs seconds and catches config typos before they cost queue time.

```bash
python scripts/smoke_test.py --config configs/<name>.yaml --device cpu

# interactively on an already-allocated GPU node:
python -m engine.train --config configs/<name>.yaml

# on the cluster, via SLURM (see §4 for which sbatch script to use):
sbatch scripts/slurm_<...>.sbatch
```

Outputs land under `<output.dir>/<output.run_name>/` (from the config — usually
`$WORK/vesuvius/outputs*/<run_name>/`):
- `best.pt` — checkpoint at the best swept-F0.5 epoch (saved every time it improves, not just at
  the end)
- `train_log.csv` — per-epoch loss, val F0.5 (fixed-τ and swept), val IoU
- `val_prob_best.npy` — the corresponding full-resolution validation probability map
- `summary.json` — best epoch/τ/F0.5/precision/recall/IoU
- `config_used.json` — exact config snapshot (for provenance)

### 2.1 Post-training steps (run after training finishes)

```bash
# sweep (tau, min-component-size) on the val probability map -> postprocess.json
python -m scripts.postprocess --run <run_name>

# plot train loss + val F0.5 (fixed/swept) + val IoU per epoch
python scripts/plot_train_curves.py --run <run_name>

# render the standard 4-panel validation figure at the selected tau
TAU=$(python -c "import json;print(json.load(open('outputs/<run_name>/summary.json'))['best_tau'])")
python scripts/visualize.py --prob outputs/<run_name>/val_prob_best.npy \
    --seg 20241113090990 --label-file ink_reg.npy --tau "$TAU" --downsample 2

# run + visualize on the (unlabeled) test segment — qualitative only, no F0.5
python -m scripts.predict --run <run_name>
PROB=$(ls outputs/<run_name>/test_prob_*.npy | head -1)
python scripts/visualize.py --prob "$PROB" \
    --seg 20251115002740-auto_grown_20251115002740308_0_flatboi --no-gt --tau "$TAU" --downsample 6
```
If you use a non-default output root (see §3), set `INK_OUTPUT_ROOT` before any of the above, or
pass the run's actual output directory explicitly where a script accepts `--root`.

---

## 3. Isolating an experiment round (don't overwrite existing results)

`data/segments.py` reads `INK_OUTPUT_ROOT` from the environment; if unset, everything defaults to
`$WORK/vesuvius/outputs/`. Every follow-up round in this project set this to a fresh directory
before running, so nothing was ever overwritten:

```bash
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs7"     # pick a new number
mkdir -p "$INK_OUTPUT_ROOT"
# now engine.train / postprocess / predict / rescore scripts all write under outputs7/
```

Existing rounds, for reference:

| root | contents |
|---|---|
| `outputs/` | the original study — 44 runs (t64 ladder, refiners, LOO folds, t192 ablation suite) |
| `outputs2/` | t192 screens (uncertainty variants, word-box, EMA, z_flip, geometry screens) + TTA/stress/probe analyses |
| `outputs3/` | the t64 ablation suite re-run under the fixed training loop |
| `outputs4/` | tile 256 and 320 |
| `outputs5/` | the final combo (`c1_t256`) + its analyses |
| `outputs6/` | the geometry closing experiment (`aux_shat`) |
| `output_inter/` | interpretability probe artifacts (no trained checkpoints here — inference-only outputs) |

---

## 4. SLURM scripts — what's available and when to use each

All under `scripts/slurm_*.sbatch`. Every one follows the same shape: `py_compile` the touched
files → CPU smoke test → `srun python -m engine.train` → postprocess → plot curves → visualize
val → predict + visualize test. Submit with `sbatch scripts/slurm_<name>.sbatch`; array jobs run
several seeds/variants as `sbatch --array=<range> ...` (already set inside the script — just
`sbatch` it).

**Single-config launchers** (one architecture ladder rung each; A40, `outputs/`):
`slurm_a0.sbatch`, `slurm_a0s.sbatch` / `slurm_a0s_a40.sbatch`, `slurm_a0b_a40.sbatch`,
`slurm_a1_a40.sbatch`, `slurm_a2_a40.sbatch`, `slurm_a2b_a40.sbatch`, `slurm_a2bs_a40.sbatch`,
`slurm_a5_a40.sbatch`, `slurm_a6_a40.sbatch` — reproduce one rung of the original t64 ladder.

**Preprocessing-variant launchers** (A40, `outputs/`, all measured harmful — reproduce only to
verify the negative result): `slurm_a1_ph_a40.sbatch` (photometric aug), `slurm_a1_pn_a40.sbatch`
/ `slurm_a2bs_pn_a40.sbatch` (per-tile z-score), `slurm_a1_spn_a40.sbatch` /
`slurm_a2bs_spn_a40.sbatch` (spatial z-score), `slurm_a3_a40.sbatch` (reg labels + photometric).

**Fixed-loop / tile-size launchers** (A40, `outputs/`): `slurm_a1_reg_a40.sbatch`,
`slurm_a2bs_reg_a40.sbatch`, `slurm_a1_t128_a40.sbatch`, `slurm_a2bs_t128_a40.sbatch`,
`slurm_a2bs_t192_a40.sbatch` (the t192 reference model), `slurm_seeds_a40.sbatch` (3-seed band for
a2bs_t192).

**Ablation suites:**
- `slurm_abl_a40.sbatch` — the 8 final-recipe ablations at t192 (stem/DA/surfrel/perpix/warpstab/
  labels/loss×2), array job.
- `slurm_lean_a40.sbatch` — the lean config, 3 seeds, array job. **This is the reference band
  every follow-up screen is compared against.**
- `slurm_abl64_a40.sbatch` — the t64 ablation suite re-run under the fixed loop (11-task array:
  3-seed reference band + 8 toggles), writes to `outputs3/`.

**Other stage-1 model families:** `slurm_loo_t192_a40.sbatch` (8 leave-one-out folds, array job) ·
`slurm_refiner_a40.sbatch`, `slurm_r2_a40.sbatch`, `slurm_r2self_a40.sbatch`, `slurm_r3_a40.sbatch`,
`slurm_r4_a40.sbatch` (the stage-2 refiner variants — r4 is the reported one).

**Follow-up round (outputs2/, A40 unless noted):**
- `slurm_p2_wordbox_a40.sbatch` — word-box sampling, 3 seeds, array job (rejected finding).
- `slurm_screens_a40.sbatch` — 8-task array: word-box ratio screens, u1/u2 uncertainty variants,
  EMA, no-z_flip, z_shift±6, per-segment centering.
- `slurm_tta_stress_a40.sbatch` — TTA rescoring + depth-offset stress test + ŝ-fidelity probe on
  existing checkpoints (inference-only, no training).

**Tile scaling (outputs4/):**
- `slurm_t256_a40.sbatch` — `t256_lean`, 3 seeds, array job, A40 (has an automatic batch-4→2
  fallback if a seed OOMs).
- `slurm_t256plus_a100.sbatch` — `t256_nosurf` + `t320_lean`, A100 (2-task array).
- `slurm_t256_analysis_a100.sbatch` — extends the stress test + TTA rescoring to the t256/t320
  checkpoints, inference-only, A100.

**Final combo (outputs5/, A100):**
- `slurm_c1_a100.sbatch` — the reported final model: `c1_t256`, 3 seeds + one 90-epoch variant,
  4-task array. **This is the launcher for the headline number.**

**Geometry closing experiment (outputs6/, A100):**
- `slurm_auxshat_a100.sbatch` — trains the auxiliary-supervised surface estimator, then
  self-evaluates with both geometry probes (ŝ fidelity + depth-offset stress) in the same job.

**Interpretability (output_inter/, A40, inference-only — no training):**
- `slurm_interp_a40.sbatch` — the depth-readout probe, error anatomy, effective receptive field,
  and frequency-band occlusion, chained in one job.
- `slurm_inkalyzer_a40.sbatch` — the Inkalyzer depth-attribution XAI run (15 checkpoints).

**Visualization / prediction utility jobs:** `slurm_viz12_a40.sbatch`,
`slurm_predict_test_a40.sbatch`.

---

## 5. Reproducing the exact headline number

The reported final model is `configs/c1_t256_s42.yaml` (tile 256, lean architecture — no texture
stem, no depth attention, surface-relative collapse kept — plus the uncertainty-masked loss and
z_flip disabled).

```bash
export INK_OUTPUT_ROOT="$WORK/vesuvius/outputs5"     # or a fresh root of your own

# manifests this config needs (build once if not already present):
python -m data.tiling --tile 256 --stride 128 --min-mask 0.5 --label-file ink_reg.npy --suffix _reg_t256
python -m scripts.make_uncertainty --r 4 --label-file ink_reg.npy

# submit all 3 seeds + the 90-epoch check in one array job:
sbatch scripts/slurm_c1_a100.sbatch

# or reproduce just seed 42 interactively:
python scripts/smoke_test.py --config configs/c1_t256_s42.yaml --device cpu
python -m engine.train --config configs/c1_t256_s42.yaml
python -m scripts.postprocess --run c1_t256_s42
```
The headline number is `postprocess.json`'s `postprocessed.f05` — this is the **post-filter**
F0.5, on the validation segment, against registered labels. The reported 3-seed figure is the
mean ± std of that field across `c1_t256_s42`, `c1_t256_s43`, `c1_t256_s44`.

To add test-time augmentation on top of an existing checkpoint (no retraining):
```bash
python -m scripts.predict_val_tta --runs c1_t256_s42
```

---

## 6. Rescoring without retraining

Useful for checking a different label file, adding IoU to an old run, or applying TTA after the
fact — all of these read a saved `val_prob_best.npy` and never touch the GPU.

```bash
# IoU (ink/background/mean) + F0.5, backfilled for any run(s) that already trained
python -m scripts.rescore_iou --runs <run_name> [<run_name> ...]

# F0.5 against a specific label file (e.g. sanity-check against unregistered labels)
python -m scripts.rescore_runs --runs <run_name> --label-file ink.npy
```

---

## 7. Non-negotiables (things that silently break comparability if you skip them)

- **Score against `ink_reg.npy`, not `ink.npy`**, unless you are deliberately reproducing an
  unregistered-labels ablation (`abl_labels`, `abl64_labels`).
- **Never select τ or min-size on the test segment** — it has no ground-truth labels; F0.5 is
  never computed on it, only qualitative figures. τ/min-size are always chosen on val and only
  *applied* to test.
- **Seeds are not interchangeable with a 3-seed mean.** Measured seed-to-seed spread on this val
  segment is ±0.003–0.008 F0.5 — a single run can land anywhere in that band. If comparing against
  a reported number, compare against the mean of the same number of seeds, not a single run.
- **Match the manifest to the config.** `data.manifest_suffix` in the YAML must correspond to a
  `train<suffix>.csv`/`val<suffix>.csv` you've actually built (§1.2–1.4); training fails loudly
  (missing file) if you haven't, so this is safe but easy to forget.
- **`smoke_test.py` before every `sbatch`.** It's a CPU-only check that takes seconds and has
  caught real config mistakes before they cost hours of GPU queue time.
