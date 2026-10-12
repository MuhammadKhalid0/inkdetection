# Ink Detection in Herculaneum Papyri with 3D Transformers

A controlled benchmark of three 3D transformer segmentation models (**UNETR++**, **nnFormer** and
**SegFormer3D**) for ink detection on Scroll 5 (PHerc. 0172) of the Vesuvius Challenge data.
This is the code and results for an MSc thesis at the Pattern Recognition Lab, FAU Erlangen-Nürnberg.
For the short version, see [PROJECT.md](PROJECT.md).

> **Data is not included.** The scroll volumes and ink labels belong to their providers
> (Vesuvius Challenge / EduceLab). This repository contains only our code, configs, documentation
> and result files (JSON, training logs, analysis summaries). Model checkpoints (`*.pt`) and
> probability maps (`*.npy`) are excluded for size.

---

## 1. Task

- **Input:** a 3D sub-volume of the surface volume, 32 of the 65 surface layers (window `[16,48)`
  centred on the papyrus surface) × a `t × t` in-plane tile. Intensities are scaled to `/255`.
- **Output:** a 2D ink probability map for the tile. Tiles are combined into a full-segment map by
  sliding-window inference with Gaussian blending.
- **Data:** 9 labelled segments of Scroll 5, with 5.3–15.8 % ink per segment.
- **Split (fixed by the supervisor):** 8 segments for training · **val = `20241113090990`** (all
  model selection and every reported number) · test = the unlabelled `flatboi` segment (qualitative
  only).
- **Metric:** masked **F0.5**, which weights precision over recall, scored against **registered** labels
  (`ink_reg.npy`, see §4). The threshold τ is swept over {0.10…0.90} every epoch and selected on val.
  - **raw** is the swept F0.5 of the probability map.
  - **post** adds an 8-connected minimum-component-size filter, with (τ, min size) selected on val.
  - IoU (ink / background) is also reported but never used for selection.
- **Noise floor:** a measured single-seed |Δ| ≤ **0.008**. No claim is made below 3 seeds
  (seeds 42/43/44).

## 2. Results at a glance

### Benchmark row (val segment, post-processed F0.5)

| Model | Configuration | Params | Seeds | raw F0.5 | **post F0.5** | best single |
|---|---|---|---|---|---|---|
| **UNETR++** | `c1_t256` (lean + uncertainty ignore-band, no z-flip) | 79.1 M | 3 | 0.8111 ± 0.0039 | **0.8190 ± 0.0062** | **0.8261** |
| nnFormer | `nnf_c1_t256` (+ SR-PB) | 19.9 M | 3 | 0.7801 ± 0.0037 | 0.8005 ± 0.0102 | 0.8122 (0.8133 with TTA) |
| nnFormer | stock `nnf_s_d64_t256` | 19.9 M | 3 | 0.7825 ± 0.0117 | 0.7981 ± 0.0123 | 0.8120 |
| SegFormer3D | `sf_m_t256` (SF-M) | 20.0 M | 1 | 0.8031 | 0.8231 | n/a |
| SegFormer3D | `sf_base_t256` (SF-S) | 4.4 M | 1 | 0.7948 | 0.8204 | n/a |
| SegFormer3D | `sf_base_t320` (SF-S) | 4.4 M | 1 | 0.8003 | 0.8191 | n/a |

The baseline (vanilla UNETR++ at t64) started at **F0.5 ≈ 0.42–0.48**. Most of the gain came from
context scale, a corrected training/selection protocol, and label registration, not from
architecture. The SegFormer3D rows are single-seed (seed 42); its 3-seed band has not been run yet.

### Efficiency (bf16 inference, batch 1)

| Model | tile | params | GFLOPs | ms / tile | peak mem |
|---|---|---|---|---|---|
| UNETR++ | 256 | 79.1 M | n/a | n/a | n/a |
| nnFormer S | 256 | 19.9 M | 305.6 | 36.5 | 1.26 GB |
| SegFormer3D S | 256 | 4.44 M | 497.5 | 36.8 | 1.37 GB |
| SegFormer3D M | 256 | 19.97 M | 898.1 | 49.1 | 1.48 GB |

Full tables: [nnformer/docs/bench_table.md](nnformer/docs/bench_table.md),
[segformer3d/docs/bench_table.md](segformer3d/docs/bench_table.md).

---

## 3. Repository layout

```
inkdetection/
├── PROJECT.md                 short project description
├── bench/                     the shared harness (data → model → train → infer → eval) + UNETR++
│   ├── data/                  prepare.py (volume/label alignment), tiling.py, dataset.py, segments.py
│   ├── engine/                train.py, resume.py, infer.py, eval.py, losses.py, refiner.py
│   ├── models/
│   │   ├── unetrpp/           the adapted UNETR++ backbone (unetr_pp_ink.py, EPA / DA-EPA blocks)
│   │   ├── shared.py          TextureStem · DepthCollapse (mean/learned/surface-relative) · Head
│   │   ├── wrapper.py         build_model(cfg): stem → backbone → depth collapse → 2D head
│   │   ├── nnformer     ->    ../../nnformer/src/nnformer_ink     (symlink)
│   │   └── segformer3d  ->    ../../segformer3d/src/segformer3d_ink (symlink)
│   ├── scripts/               training/inference/analysis/figure scripts + SLURM job files
│   ├── xai/                   attribution (Inkalyzer / IBA) utilities
│   ├── configs/               73 UNETR++ run configs (one YAML per run)
│   ├── README_REPRODUCE.md    step-by-step reproduction of the UNETR++ chapter
│   └── TUTORIAL.md
├── nnformer/
│   ├── src/nnformer_ink/      port: windows, attention, blocks, sampling, backbone, contrib (SR-PB, CSSA)
│   ├── configs/ scripts/ tests/
│   ├── probes/                ERF, frequency-band, error-anatomy, Inkalyzer outputs (JSON + plots)
│   ├── rereg/                 second label re-registration round (JSON + reports)
│   └── docs/                  FINAL_SUMMARY.md, RESULTS.md, PORT_LOG.md, TUTORIAL, presentation
├── segformer3d/
│   ├── src/segformer3d_ink/   port: geometry, blocks, decoder, backbone, contrib (PDK, DCM, OSR, DSM)
│   ├── configs/ scripts/ tests/   (50 port checks + resume/chain tests)
│   └── docs/                  RESULTS.md (auto-generated gates), PORT_LOG.md, TUTORIAL, presentation
├── results/                   every run's result files (no checkpoints / prob maps)
│   ├── unetrpp/outputs{,2..6}/<run>/   summary.json · postprocess.json · config_used.json · train_log.csv
│   ├── unetrpp/output_inter/           interpretability probes (ERF, φ_rel, freq-band, error anatomy)
│   ├── nnformer/<run>/ …               + tta/, zshift_stress/
│   └── segformer3d/<run>/ …
└── docs/                      chapter reports and plans (UNETR++ final report, nnFormer/SegFormer3D plans,
                               preprocessing, pipeline explainer, data facts)
```

### What each result file holds

| file | content |
|---|---|
| `config_used.json` | the exact resolved config for the run |
| `summary.json` | best swept F0.5, best τ, best epoch, IoU, label file used |
| `postprocess.json` | post-filter F0.5 / precision / recall and the selected (τ, min size) |
| `train_log.csv` | per-epoch train loss, val F0.5 (swept and fixed), τ, lr |
| `*_meta.json` | metadata for saved test/train probability maps (the maps themselves are not included) |
| `zshift_stress.json` | F0.5 under artificial depth offsets δ ∈ [−8, +8] at frozen τ |
| `tta*.json` / `tta_summary.md` | 8-fold dihedral test-time augmentation rescoring |
| `rescore_*.json` | rescoring of all runs against original vs registered labels |

---

## 4. What we did

### 4.1 One harness, three backbones

All three models plug into the same wrapper:

```
x [B,1,D=32,t,t] → (optional TextureStem) → 3D backbone → feature volume [B,C,D,t,t]
                 → DepthCollapse (mean | learned | surface-relative) → [B,C,t,t] → 2D Head → logits
```

Data, split, labels, sampler, augmentation, loss, optimiser schedule, inference, post-processing,
metrics and probes are identical across models. Only the backbone changes. Every new component
sits behind a default-off flag and starts as an **exact no-op at initialisation** (zero-init gates
and tables, verified bitwise in unit tests). An ablation is therefore always a single toggle, and
the parameter counts check each rung.

**Training recipe (final):** Dice+BCE (w = 0.5) · AdamW lr 1e-4, wd 3e-5 · 3-epoch warm-up then
cosine to 60 epochs · patience 20 · grad-clip 1 · 1:1 positive-tile oversampling · flips/rot90 +
z-shift ±2 augmentation · bf16 autocast from t192 upwards · early stopping and selection on
**swept** F0.5.

### 4.2 Data-side work

- **Label registration (data contribution).** The provided ink labels were offset from the
  volume by 10–40 px vertically, and on several segments the offset varied along x. We estimated the
  per-segment (and warped) offsets with a detrended central-slab contrast method. Three independent
  checks confirmed them: model-probability correlation, visual overlays, and depth-band stability.
  Rescoring against the registered labels improved **every** model by +0.03…+0.07 and kept their
  ranking. Retraining on them gives +0.013 at t192.
  Scripts: `bench/scripts/register_labels.py`, `nnformer/scripts/reregister.py`.
- **Uncertainty ignore-band (u1).** A trimap band around label edges is excluded from the loss:
  **+0.012** at t192. Modelling uncertainty as a third class (u2) or using word-box sampling performed
  worse (−0.008 and −0.04…−0.17). Where the fix is applied matters: it works in the loss and fails in
  the sampler.
- Rejected preprocessing (all measured): photometric augmentation, per-tile 3D z-score, spatial
  z-score, CLAHE, background detrending, depth refocusing.

### 4.3 Protocol fixes (a methods result)

1. **Swept vs fixed τ.** Fixed-τ selection under-reported the same weights by up to **0.163** F0.5.
2. **Completed cosine schedule.** The original loop never finished its LR schedule, so val F0.5
   oscillated by ±0.04–0.09 per epoch at t64. Re-running the ablation suite under the fixed loop
   **erased the apparent +0.05 of DA-EPA** and **confirmed the texture stem's −0.073**.
3. **Band-vs-band comparison.** A contribution is compared against a 3-seed baseline band, never a
   single baseline seed. This caught the "seed-42 trap" twice: on UNETR++ (DA-EPA) and on nnFormer
   (SR-PB), where single-seed deltas of +0.013…+0.015 fell to +0.0065, which is noise.
4. 60 epochs is enough. In every case the 90-epoch run did not beat the 60-epoch run with the same seed.

### 4.4 Model 1: UNETR++ (reference)

Port of Shaker et al. (IEEE TMI 2024), adapted to a `(32, t, t)` input.
- **Port fixes:** the published code hardcodes medical token counts and silently mis-sizes on other
  inputs. We recompute the EPA projections and positional embeddings from `patch_size`, make the stem
  patch-aware, use a depth-preserving `(1,4,4)` embed with a 2D positional embedding tiled over depth,
  fix the block-internal axis order ("axis trap"), and use bf16 from t192 (fp16 overflows).
- **Our components:** TextureStem, DA-EPA (a depth-attention gate inside EPA), and the
  **surface-relative geometry family**: a soft-argmax surface estimate ŝ, a surface-relative depth
  collapse φ_rel, and an optional per-pixel warp with stabilisation.
- **Tile ladder (lean, raw F0.5):** t64 0.639 → t128 0.719 → t192 0.774 → **t256 0.804** →
  t320 0.800 (overfits at 112 M params). Context scale is the strongest single lever.
- **Ablations at t192 (3-seed reference band):** stem 0, DA-EPA neutral at every scale,
  surfrel −0.009 when removed (the only architectural survivor), per-pixel warp and warp
  stabilisation neutral, unregistered labels −0.012.
- **Stage-2 refiner** (a 2D U-Net on probability maps with letterform pretraining, `r4`):
  +0.010 and 60 % lower variance at t192.
- **Geometry, the full story.** Unsupervised, ŝ does not follow the physical surface (r ≈ 0.2) and
  φ_rel stays at its initial value, yet the prior still gives robustness to depth offsets
  (+0.05…+0.095 at |δ| ≥ 4). Supervising ŝ makes it accurate (r = 0.907) but moves performance from
  the centre (−0.050) to the tails, with the same mean over δ ∈ [−8, 8] (0.596 = 0.596).
- **Final model `c1_t256`:** 0.8190 ± 0.0062 post (3 seeds), best seed 0.8261, ink-IoU ≈ 0.60.

Full report: [docs/unetrpp_final_report.md](docs/unetrpp_final_report.md) ·
reproduction: [bench/README_REPRODUCE.md](bench/README_REPRODUCE.md).

### 4.5 Model 2: nnFormer

Port of Zhou et al. (nnFormer) with local windowed attention (Swin-style), inside the same harness.
- **Port finding:** the published relative-position-bias indexing is unsafe for anisotropic windows,
  because it silently maps distinct offsets to the same entry. Fixed in `windows.py`. All
  deviations D1–D9 are logged in [nnformer/docs/PORT_LOG.md](nnformer/docs/PORT_LOG.md).
- **Contributions:**
  - **SR-PB** (surface-relative position bias, an additive key-side depth bias in the frame of ŝ).
  - **CSSA** (cross-scale skip attention).
  Both are bitwise no-ops at init.
- **Verdicts (27 runs):**
  - Capacity: S 19.9 M ≈ M 79.1 M ≈ F 177.8 M within noise, and F overfits.
  - Stock toggles: all neutral. Removing skip attention gives −0.0002.
  - **SR-PB:** accuracy-neutral band-vs-band (+0.0065), but seed variance is **3–5× lower** and
    depth attribution is visibly broader (entropy 0.828 vs 0.792).
  - **CSSA:** raises the effective receptive field (r50 +19 %) but leaves F0.5 unchanged.
  - Supervising ŝ hurts (0.7607), which replicates the UNETR++ finding.
  - vFFC prior art (Quattrini et al. 2023): best t192 number (0.7844 / 0.7992), but only one seed
    and capacity-confounded, so it stays provisional.
- **Why nnFormer trails UNETR++ (−0.0185):** the effective receptive field covers r90/tile ≈ 51 % at
  t192 and **shrinks** to 45 % at t256, while UNETR++ stays at a constant ~53 %. Windowed local
  attention limits how much of a larger tile the model can use, so nnFormer's tile ladder is almost
  flat (+0.0063 for t128→t192, against +0.055 for UNETR++).
- **Cross-model agreement:** 5 of 6 diagnostic questions get the same answer from both models:
  - stock toggles do not matter;
  - capacity does not help;
  - attribution peaks in the same depth band around the surface (z ≈ 29–34);
  - fine "crackle" texture carries no signal;
  - the residual error profile is the same.
  These are properties of the task. The one disagreement (does context dominate?) is explained by
  the ERF result above.

Full summary: [nnformer/docs/FINAL_SUMMARY.md](nnformer/docs/FINAL_SUMMARY.md) ·
gate report: [nnformer/docs/RESULTS.md](nnformer/docs/RESULTS.md).

### 4.6 Model 3: SegFormer3D

Port of Perera et al. (SegFormer3D, CVPRW 2024), with sequence-reduced attention, Mix-FFN and an
all-MLP decoder.
- **Port findings:**
  - The reference code is unusable as published. Its `cube_root(N)` reshapes crash at
    t64/128/192/320 and **silently scramble the geometry at t256** (32·64·64 = 32³ is the only
    perfect cube on the ladder).
  - `@torch.jit.script` does not import on torch 2.6.
  - The port reproduces the reference parameter count exactly (4,491,809).
  - A `test_config_keys_reach_the_model` test guards against YAML keys the wrapper silently ignores.
- **Contributions** (all bitwise no-ops at init):
  - **C1 PDK** (peripheral downsampled keys), tested in a 2×2 design against a mandatory same-cost
    control.
  - **C2 DCM** (depth-coordinate Mix-FFN).
  - **C3 OSR** (overlapping sequence reduction).
  - **C4 DSM** (decoder spatial mixing).
- **Results so far (single seed, 31 runs):**
  - Tile ladder SF-S (raw): t64 0.682 → t128 0.758 → t192 0.786 → t256 0.795 → t320 0.800.
    This is steeper than nnFormer (+0.028 for t128→t192) and flatter than UNETR++.
  - **C1 PDK:** +0.0039 over stock, but **−0.0003 against the same-cost control**, so the
    "new information" claim is not supported.
  - **C2 DCM:** absolute +0.0012 and surface −0.0006, both inside noise.
  - **C3 OSR:** **+0.0094**, outside the noise floor. It is marked for 3-seed confirmation.
  - **C4 DSM:** neutral, and it does not substitute for the post-filter.
  - Capacity × context: SF-M t256 = **0.8031 raw / 0.8231 post**, the best SegFormer3D cell.
    SF-S at 4.4 M parameters reaches 0.8204 post at t256.
  - The full 65-layer depth stack (`sf_depth65`) gives no gain over the 32-layer window
    (0.7874 vs 0.7857).
  - 2.5D control: 0.7254, so 3D input matters.
- **Engineering:** a resumable training engine (`bench/engine/resume.py`). It is proven byte-identical
  to uninterrupted training on CPU and runs as self-chaining SLURM jobs for runs longer than the
  wall-time limit.
- **Still open:** the 3-seed band (Phase S3) and the analysis job (ERF / Inkalyzer / stress, Phase S4).
  SF-L t256 is deferred.

Gate report: [segformer3d/docs/RESULTS.md](segformer3d/docs/RESULTS.md) ·
port log: [segformer3d/docs/PORT_LOG.md](segformer3d/docs/PORT_LOG.md).

### 4.7 Interpretability probes (applied identically to every model)

| probe | question | UNETR++ finding |
|---|---|---|
| ERF (`interp_erf.py`) | how much of the tile is used? | r90 ≈ 53 % of the tile at every size |
| frequency-band occlusion (`interp_freqband.py`) | which spatial scales carry the signal? | 2–8 px ≈ 0; 128+ px −0.31, so the model reads **morphology**, not texture |
| error anatomy (`interp_error_anatomy.py`) | why is ink missed? | only 18.7 % of missed ink is invisible (p < 0.07); 32 % is just below τ. The model is under-confident, not blind |
| Inkalyzer depth attribution (`inkalyzer_run.py`) | which depth layers matter? | z ≈ 27–33 surface band, ≈ 0 at window edges; misregistered labels smear it |
| z-shift stress (`zshift_stress`) | robustness to depth offset | the surfrel prior adds +0.05…+0.095 at \|δ\| ≥ 4 (t192) |

### 4.8 Overall lever ranking

> context scale (+0.16, saturating at t256) ≫ selection protocol (up to +0.163 per run) >
> uncertainty masking ≈ component filter ≈ label registration ≈ no z-flip ≈ refiner (each ≈ +0.01) >
> surface-relative prior > TTA (+0.0035) > everything else ≈ 0 (including every architecture swap
> at matched protocol).

---

## 5. Running the code

The code was developed and run on the NHR@FAU *Alex* cluster (A100 / A40). Some scripts and
`data/segments.py` contain absolute cluster paths (`$WORK=/home/woody/iwi5/iwi5401h`, Vault data
path). Adjust them to your setup.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r bench/requirements.txt

cd bench
# 0. pre-flight (CPU, no data needed)
python scripts/smoke_test.py --config configs/a0.yaml --device cpu
# 1. preprocess the segments (you need your own copy of the Scroll 5 data)
python -m data.prepare
python -m data.tiling --tile 256 --stride 128 --min-mask 0.5
# 2. train / evaluate a run
python -m engine.train --config configs/c1_t256_s42.yaml
# nnFormer / SegFormer3D configs live in their own folders and use the same entry point:
python -m engine.train --config ../nnformer/configs/nnf_c1_t256_s42.yaml
python -m engine.train --config ../segformer3d/configs/sf_base_t256_s42.yaml

# unit tests for the ports
python ../nnformer/tests/test_smoke.py
python ../segformer3d/tests/test_smoke.py
```

Result tables are regenerated from the run folders by `nnformer/scripts/collect_results.py` and
`segformer3d/scripts/collect_results.py`, which apply the pre-registered gates mechanically.

## 6. References

- Shaker et al., *UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation*, IEEE TMI 2024.
- Zhou et al., *nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer*, IEEE TIP 2023.
- Perera et al., *SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation*, CVPRW 2024.
- Quattrini et al., *Volumetric Fast Fourier Convolution for Detecting Ink on the Carbonized
  Herculaneum Papyri*, ICCVW 2023.
- Vesuvius Challenge — https://scrollprize.org (data source; not redistributed here).

The original upstream repositories were used as read-only references and are not included. Our
adapted ports live under `bench/models/unetrpp`, `nnformer/src` and `segformer3d/src`. Each port
folder keeps the upstream license as `LICENSE_UPSTREAM`: UNETR++ is Apache-2.0, nnFormer is MIT,
and SegFormer3D is GPL-3.0.
