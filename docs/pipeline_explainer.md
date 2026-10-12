# The Pipeline, Explained (for talking a friend through it)

A quick-reference companion to `UNTER_Plus_Plus_Final.md`. Four questions: what preprocessing, how is sampling done, what does each script do, how do you reproduce a number.

---

## 1. Preprocessing — what actually happens to the raw data

Raw data lives on Vault: per segment, a pre-stacked `(65, H, W)` uint8 volume (65 depth layers sampled along the surface normal) + an `inklabels.png` (RGBA, near-binary ground truth) + a `mask.png` (which pixels are real papyrus vs empty/background).

**`data/prepare.py`** turns that into fast, aligned arrays, once per segment:
1. Load the volume as a memmap (no full read into RAM).
2. Load `inklabels.png`, threshold at 127 → binary `{0,1}` ink mask.
3. Load `mask.png`, threshold at 127 → binary `{0,1}` valid-papyrus mask.
4. **Crop all three to their common minimum (H, W).** The mask PNGs are consistently 1 px smaller than the volume on every segment (an upstream rasterization off-by-one), and one segment (`120732`) has its ink label 16 rows short. Cropping `[:Hc, :Wc]` from the top-left fixes this — verified correct by an independent contrast scan (the alternative, bottom-anchoring, is clearly worse).
5. Compute a per-segment **depth-intensity profile** (mean intensity per layer, subsampled) and its argmax → `depth_peak`, saved in `meta.json`. This is later used by `center_mode: per_segment`.
6. Write `vol.npy`, `ink.npy`, `mask.npy`, `meta.json` to `data/proc/<segment>/`.

**That's the entire pixel-level preprocessing.** Two important non-obvious facts:
- **Intensity normalization is just `/255`.** Nothing else. Every fancier attempt — per-tile z-score, spatial z-score, CLAHE, background detrending, photometric augmentation — was tried and **made things worse** (by −0.06 to −0.15 F0.5), because the ink signal *is* faint contrast, and anything that "fixes" brightness variation destroys exactly that signal. This was tested five separate times before it was believed.
- **`scripts/register_labels.py` is a second, separate preprocessing step** run once, offline, before any training: it detects that the ink labels sit 20–54 px below where the ink actually is (found by sweeping a vertical shift and maximizing detrended ink-vs-papyrus contrast in a central depth slab), fits a linear `dy(x)` correction per segment, and only applies it if the fit beats doing nothing by a real margin (`GAIN_MIN = 0.4`). 5 of 9 segments get warped; 4 are left alone. Output is a **new file**, `ink_reg.npy`, sitting next to the original `ink.npy` — the original is never touched. All final training uses `ink_reg.npy`.

There is **no depth-axis preprocessing** beyond picking a fixed window: every model reads layers `[16, 48)` of the 65 (32 layers, centered on layer 32, which is where the surface-sampling construction puts the papyrus by design).

---

## 2. Sampling — how tiles are chosen for training

The segments are far too big to fit a GPU (up to 10,580 × 22,000 px), so training operates on small square crops ("tiles") cut from them.

**Step 1 — build the tile manifest (`data/tiling.py`, run once per tile size):**
- Slide a `tile × tile` window across each segment at `stride = tile/2` (so tiles overlap 50%), always including one extra tile flush against the far edge so nothing is missed.
- Keep a tile only if ≥50% of it falls inside the valid-papyrus mask (`min_mask=0.5`) — this throws away tiles that are mostly empty background.
- Record `(segment, y, x, ink_frac, mask_frac)` for every surviving tile into a CSV. **A manifest row is a coordinate, not pixels** — actual voxels are read lazily at training time.
- Train manifest = every kept tile from the 8 training segments, pooled. Val manifest = every kept tile from the validation segment.

**Step 2 — oversample positives at training time (`data/dataset.py: build_sampler`):**
Ink is only ~9–13% of pixels, so uniform sampling would train mostly on blank tiles. A tile is "positive" if ≥5% of it is ink (`pos_thresh`). A `WeightedRandomSampler` gives all positives combined 50% draw probability and all negatives combined the other 50%, regardless of how many of each exist — so every training batch is roughly 1:1 ink:background **in expectation**, no matter the true ~9:1 imbalance. One epoch = 4000 tiles drawn *with replacement* under this scheme (not a full pass over the manifest).

**Step 3 — read a tile (`InkDataset.__getitem__`):**
- Slice the volume at the fixed depth window `[16,48)` and the tile's `(y,x)` box, `/255` normalize.
- Slice the label the same way.
- Apply augmentation if training: random hflip/vflip/rot90 (applied identically to input and label), a random ±2-layer shift of the depth window (`z_shift`, teaches the model not to rely on one exact depth index), and (until it was measured harmful and dropped) a random depth-axis flip.
- Return `x: [1,32,tile,tile]`, `y: [1,tile,tile]`.

**A late refinement (the "u1" uncertainty variant):** because some label pixels are less trustworthy than others (near stroke boundaries, or on segments where the registration fit was too weak to act on), a separate per-pixel confidence map (`unc.npy`, 2=confident-ink / 1=uncertain / 0=confident-background) can be shipped alongside each tile, and the loss simply **excludes uncertain pixels** instead of training on them as if they were certain. This measurably helped (+0.012 F0.5) and is the one supervisor-suggested idea from this round that survived.

**What was tried and rejected on the sampling side:** anchoring positive tiles directly on the labeled ink components ("word-box" sampling, restricting negatives to background far from any label) — this consistently made things *worse* (−0.04 to −0.17), because training the model on hand-picked word-centered crops doesn't match how it's actually run at inference (a blind sliding window over the whole segment). The lesson: fixing label noise at the **loss level** (down-weight bad pixels) worked; fixing it at the **sampling level** (change what the model sees) didn't.

---

## 3. What every script does

### Data pipeline
| script | what it does |
|---|---|
| `data/prepare.py` | raw Vault PNGs/volume → aligned `vol.npy`/`ink.npy`/`mask.npy`/`meta.json` per segment (§1) |
| `data/tiling.py` | segment + mask → tile manifest CSV (§2, step 1) |
| `data/dataset.py` | `InkDataset` (reads one tile) + `build_sampler` (1:1 oversampling / word-box mode / uncertainty trimap loading) |
| `data/segments.py` | canonical paths, the fixed train/val/test split, `OUTPUT_ROOT` (env-overridable so different experiment rounds never overwrite each other) |
| `scripts/register_labels.py` | the label-misregistration detector + corrector → writes `ink_reg.npy` per segment (§1) |
| `scripts/make_word_manifests.py` | (tested, rejected) builds annotation-anchored manifests instead of a blind grid |
| `scripts/make_uncertainty.py` | builds the per-pixel confidence trimap `unc.npy` (boundary erosion/dilation + residual-misregistration widening) |

### Model
| script | what it does |
|---|---|
| `models/unetrpp/*.py` | the adapted UNETR++ backbone (EPA/DA-EPA blocks, shape-fixed for our input, depth-preserving embed) |
| `models/shared.py` | the "Class A" shared pieces every architecture gets identically: `TextureStem` (11-channel engineered input), `DepthCollapse` (mean / learned / surface-relative, ± per-pixel ± stabilized — turns the 3D feature volume into a 2D map), `Head` (final conv → logits) |
| `models/wrapper.py` | assembles stem → backbone → collapse → head into one model from a YAML config |

### Training / evaluation
| script | what it does |
|---|---|
| `engine/train.py` | the training loop: builds data+model+optimizer from a config, trains with AMP (bf16 at large tiles), sweeps the decision threshold τ every epoch, saves the checkpoint **whenever swept F0.5 improves** (not just at the end), early-stops on patience, logs everything to `train_log.csv` |
| `engine/infer.py` | sliding-window inference: cuts a segment into overlapping tiles, runs the model on each, blends overlapping predictions with a Gaussian window, reassembles one full-resolution probability map. Also implements optional 8-fold test-time augmentation (TTA) |
| `engine/losses.py` | the loss functions: `dice_bce` (default), `tversky_bce` (tested, dead), `dice_bce_masked` (the u1 uncertainty-aware variant) |
| `engine/eval.py` | F0.5/precision/recall/IoU at one threshold, and the full threshold sweep |
| `engine/utils.py` | config loading, seeding, the warmup-cosine LR schedule, checkpoint saving, the EMA weight-averaging helper |

### Post-run scripts
| script | what it does |
|---|---|
| `scripts/postprocess.py` | after training: sweeps (τ, min-component-size) on the val probability map, keeps the best combo (the min-size filter removes small false-positive islands — this alone is worth +0.01–0.02 F0.5 on every model), writes `postprocess.json` |
| `scripts/predict.py` | loads a trained checkpoint, runs `infer.py` on a chosen segment (usually the unlabeled test segment), saves the probability map |
| `scripts/visualize.py` | turns a saved probability map into the standard figure: probability heatmap / thresholded prediction / (ground truth / TP-FP-FN overlay if labels exist) |
| `scripts/plot_train_curves.py` | plots `train_log.csv` → training loss + validation F0.5 (fixed-τ and swept) + validation IoU per epoch, best epoch marked |
| `scripts/rescore_runs.py` / `rescore_iou.py` | recompute F0.5 / IoU for a batch of existing runs against a given label file, without retraining |
| `scripts/smoke_test.py` | **run before every SLURM submission**: builds the model on CPU, checks shapes, runs one forward+backward pass, checks it doesn't crash — catches config mistakes before they cost GPU-queue time |

### Analysis / interpretability (the closing round)
| script | what it does |
|---|---|
| `scripts/predict_val_tta.py` | rescoring existing checkpoints with test-time augmentation, no retraining |
| `scripts/stress_zshift.py` | evaluates a checkpoint with the depth window deliberately shifted, to measure robustness to a surface offset that doesn't occur naturally in this dataset |
| `scripts/probe_shat.py` | checks whether the model's learned "surface position" estimate actually correlates with the true intensity-based surface (it mostly doesn't, unless trained with `aux_shat`) |
| `scripts/interp_erf.py` | effective receptive field: how far away input pixels actually influence one output pixel, per tile size |
| `scripts/interp_freqband.py` | removes different spatial-frequency bands from the input and measures the F0.5 drop, to find which texture scale the model actually reads |
| `scripts/interp_error_anatomy.py` | breaks down every missed ink pixel by cause (near threshold / faint / boundary-uncertain / genuinely invisible) |
| `scripts/interp_phirel.py` | reads the model's learned depth-collapse weights straight out of the checkpoint and compares them to their initialization |
| `scripts/inkalyzer_run.py` | the depth-attribution XAI tool (information-bottleneck attribution — what depth layers the model actually needs to make its prediction) |

### SLURM launchers (`scripts/slurm_*.sbatch`)
Each one is a thin wrapper around the same pattern: `py_compile` the touched files → `smoke_test.py` on CPU → `engine.train` on GPU → `postprocess.py` → `plot_train_curves.py` → `visualize.py` on the val segment → `predict.py` + `visualize.py` on the test segment. One sbatch per experiment or small group of experiments (e.g. `slurm_lean_a40.sbatch` runs the 3-seed lean config as a SLURM array job; `slurm_c1_a100.sbatch` runs the final combo).

---

## 4. How to reproduce a result

Everything is config-driven — a run is fully specified by one YAML file plus a manifest.

**A. One-time data prep (only needed once, ever):**
```bash
python -m data.prepare                        # all 9 segments -> data/proc/<seg>/
python -m scripts.register_labels              # -> ink_reg.npy per segment
python -m data.tiling --tile 192 --stride 96 --min-mask 0.5 \
    --label-file ink_reg.npy --suffix _reg_t192   # -> manifests for that tile size
```
(Repeat `tiling.py` for each tile size you need: 64/32, 128/64, 192/96, 256/128, 320/160.)

**B. Train + evaluate one run:**
```bash
python scripts/smoke_test.py --config configs/<name>.yaml --device cpu   # sanity check first
python -m engine.train --config configs/<name>.yaml       # writes outputs/<run_name>/
python -m scripts.postprocess --run <run_name>
python scripts/plot_train_curves.py --run <run_name>
```
Everything a run produces — `best.pt`, `summary.json` (best epoch/τ/F0.5/IoU), `postprocess.json` (best τ+min-size), `train_log.csv`, `val_prob_best.npy`, `figs/` — lands under `outputs/<run_name>/` (or whichever root `output.dir` in the config points at).

**C. Reproduce the exact headline number:**
The final reported model is `configs/c1_t256_s42.yaml` (tile 256, lean architecture, uncertainty-masked loss, no z_flip). Run it via B above, or submit `scripts/slurm_c1_a100.sbatch` directly to SLURM — it chains all of B automatically for seeds 42/43/44. The number in the thesis (post-filter F0.5) comes straight from that run's `postprocess.json`.

**D. Rescore without retraining** (e.g. against a different label file, or with TTA):
```bash
python -m scripts.rescore_iou --runs <run_name>            # IoU + F0.5 from the saved probability map
python -m scripts.predict_val_tta --runs <run_name>        # same, with test-time augmentation
```

**E. Keep experiment rounds separate:** every follow-up round set `INK_OUTPUT_ROOT` (an environment variable read by `data/segments.py`) to a fresh directory (`outputs2`, `outputs3`, …) before running, so nothing ever overwrites an earlier round's results. To reproduce any specific run from this study, check its config's `output.dir` — that tells you which root it lives under.

**Non-negotiables if you rerun anything:** always score against `ink_reg.npy`, never the original `ink.npy`, unless deliberately reproducing the "unregistered" ablation; always select τ on the validation segment, never on the test segment (which has no labels — F0.5 on it is never computed at all, only qualitative figures); and match seeds if you want numbers to line up — the seed-to-seed spread on this dataset is real (±0.003–0.008 F0.5), so a single run is not directly comparable to a 3-seed mean.
