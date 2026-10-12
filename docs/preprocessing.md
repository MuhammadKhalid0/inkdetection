# Preprocessing — what happens to the raw data, for every model

**Written:** 2026-09-24 · **Scope:** UNETR++ (model 1), nnFormer (model 2), SegFormer3D (model 3).
`$WORK = /home/woody/iwi5/iwi5401h` · bench = `$WORK/vesuvius/vesuvius-ink-bench`.

> **The short answer: preprocessing is IDENTICAL for all three models, by design.**
> The benchmark contract is *identical data, split, labels, recipe, inference, post-processing,
> metrics and probes; only the backbone changes*. Every model reads the same `vol.npy`,
> `ink_reg.npy`, `mask.npy`, `unc.npy` and the same tile manifests. Nothing below is
> model-specific until §8, where the only real difference is how each architecture consumes the
> depth axis **inside the network** — which is an architecture choice, not preprocessing.

---

## 0. The raw data as it arrives

Per labelled segment, on Vault (`/home/vault/iwi5/iwi5401h/new_labeled/<seg>/`):

- `layers/00.tif … 64.tif` — **65 depth layers**, 8-bit greyscale, plus a pre-stacked `<seg>.npy`
  of shape `(65, H, W)`.
- `<seg>_inklabels.png` — RGBA, R=G=B, effectively binary.
- `<seg>_mask.png` — the valid-papyrus mask.

**9 labelled segments**, all Scroll 5 / PHerc. 0172. Sizes range from 1340×7620 to
**10580×22000**. Ink fraction 5.3–15.8%.
Split, fixed by the supervisor and never changed: **train = 8 segments · validation =
`20241113090990` · test = the `flatboi` auto-grown segment (no labels, qualitative only)**.

Two alignment defects exist in the raw data and are fixed in step 1: the mask is 1 px smaller
than the volume in both axes on **every** segment, and `20241108120732` has a 16 px height
mismatch between volume and labels.

---

## 1. `python -m data.prepare` → aligned arrays

For each segment:

1. read the pre-stacked volume as a memmap `(65, H0, W0)`;
2. read the labels, take channel 0, threshold at **> 127** → binary `{0,1}`;
3. read the mask, threshold at **> 127** → binary `{0,1}`;
4. **crop all three to the common minimum (H, W)**, which repairs both defects above;
5. copy the volume **layer by layer** through an output memmap, so peak RAM is one layer
   (≤ ~250 MB even for the 22000×10580 segment);
6. record a depth profile (mean intensity per layer on a 1-in-50 spatial subsample) and its peak.

Writes `data/proc/<seg>/`: **`vol.npy` (uint8), `ink.npy` (uint8 {0,1}), `mask.npy` (uint8 {0,1}),
`meta.json`** (shapes, `depth_profile`, `depth_peak`, `ink_fraction`).

*Note on `depth_peak`: an early note recorded one segment peaking at layer 42. That was a
documentation error; re-measured against the processed volumes it peaks at 29, and all train
segments peak at 28–30 (validation 30). The correction matters because a whole contribution family
was originally motivated by a per-segment depth drift that does not exist.*

**No intensity transform happens here.** The volume stays raw uint8.

---

## 2. `python -m scripts.register_labels` → `ink_reg.npy` (the data-side fix)

**The finding that forced this step:** on all 9 segments the hand-drawn labels sit **10–45 px below
the physical ink signal** in the volume. The offset is stable across depth bands, so it is a true
in-plane misregistration of the 2D label map, not sheet tilt. On the validation segment it varies
smoothly along x (−45 px mid-left to −25 px right), so a single global shift is not enough.

Per segment the tool:

1. picks the best depth band among candidates `(22,32) (26,36) (28,38) (30,40) (34,44)`;
2. scans `dy ∈ [−70, 25] step 5` and `dx ∈ [−30, 30] step 10`, maximising **detrended
   ink-vs-papyrus contrast**, in x-windows of width 2500 stepping 1250, requiring ≥ 20000 ink px
   in a window to trust its fit;
3. fits a confidence-weighted **linear `dy(x)`** (degree ≤ 1, deliberately conservative) plus a
   weighted median scalar `dx`;
4. **acceptance gate:** the fitted warp must beat identity by a contrast gain ≥ **0.4**, otherwise
   the segment is left unwarped (`identity_low_gain`). Four segments failed this gate and kept
   their original labels.

Writes `ink_reg.npy` and `reg_report.json`. **`ink.npy` is never modified.** Registration uses only
the volume and the labels — no model predictions — so it is preprocessing, not leakage.

Measured worth: **+0.013 F0.5 at tile 192** on model 1.

*A second, more aggressive round (`ink_reg2.npy`, per-window 2D field) was produced during chapter 2
and gave +0.009–0.012 on every checkpoint, but was **reverted** for comparability. Numbers from the
two label versions must never be mixed; `summary.json` records `label_file` for exactly this reason.*

---

## 3. `python -m scripts.make_uncertainty --r 4 --label-file ink_reg.npy` → `unc.npy`

A per-pixel **trimap** marking label trustworthiness, so the loss can ignore what the annotator
could not have got right:

```
2 = confident ink core    0 = confident background    1 = uncertain band
core = erosion(ink, r=4)   halo = dilation(ink, r=4)   band = halo AND NOT core
```

plus **residual-misregistration widening** on the four segments the registration left at identity
(`145341: 30 px · 152031: 20 · 080880: 20 · 120732: 10`): rows up to `dy` above strokes may be
unlabelled ink, and the bottom `dy` rows of stroke interiors may be papyrus. Both go to class 1.

Built for **train segments only** — the validation segment deliberately has no trimap, because it is
never trained on. Typical scale: 73% of ink pixels survive erosion as core; the uncertain band is
about 3.9% of all pixels.

Used by the `dice_bce_masked` loss, which drops class-1 pixels from the loss entirely.
Measured worth: **+0.012 F0.5**.

---

## 4. `python -m data.tiling` → tile manifests

A tile is a `(segment, y, x)` top-left coordinate; **depth is added at read time, not here**. A tile
is kept if its mask coverage ≥ **0.5**, and its ink fraction is recorded so the sampler can
oversample. The last start per axis is forced to `size - tile` so the edges are covered.

Columns: `segment,y,x,ink_frac,mask_frac`.

| manifest | tile | stride | train tiles | used by |
|---|---|---|---|---|
| `_reg` | 64 | 32 | 338,623 | all three models (t64 rung) |
| `_reg_t128` | 128 | 64 | 84,680 | models 2, 3 |
| `_reg_t192` | 192 | 96 | 37,700 | all three (the screen scale) |
| `_reg_t256` | 256 | 128 | 21,360 | all three (deployment) |
| `_reg_t256_d64` | 256 | **64** | 84,523 | models 2, 3 (density study) |
| `_reg_t320` | 320 | 160 | 13,768 | models 1, 3 |
| `_reg_t320_d96` | 320 | **96** | 37,558 | model 3 only (new, 2026-09-10) |

Denser strides raise overlap (75% at stride 64), so the extra tiles are **not** independent samples.
The honest claim is "more distinct letter-crop views under the same optimizer budget", and it is
stated wherever density appears.

---

## 5. Read time — what the model actually receives (`data/dataset.py`)

Per sampled tile:

1. **Depth window.** `z0 = clamp(center_layer − depth//2, 0, n_layers − depth)`. With the standard
   `center_layer 32`, `depth 32`, `n_layers 65` this is **`[16, 48)`** — 32 of the 65 layers,
   centred on the sheet. Chapter 1 measured the ink signal at depth **27–33**, so roughly 6 of
   these 32 layers carry signal and the rest are context.
2. **Intensity normalization: `x / 255`, and nothing else.** Per-tile z-score, percentile stretch
   and spatial z-score are all implemented and were all measured **harmful**; `global` is the only
   mode used in any reported run.
3. **Augmentation**, geometric only: horizontal flip, vertical flip and `rot90` with `k ∈ {1,2,3}`,
   each p = 0.5; **depth shift ±2 layers** with p = 0.3 (applied by moving `z0` before the read, so
   the label is unaffected). **Depth flip is OFF** — it was measured harmful, worth +0.009–0.011 to
   disable. The trimap receives the identical geometric transform as the tile and label.
   No photometric augmentation of any kind.
4. **Class balance:** a weighted sampler draws ink-positive tiles (`ink_frac ≥ 0.05`) against
   negatives at **1:1 in expectation**, `samples_per_epoch = 4000` regardless of manifest size, so
   every configuration gets exactly 1000 optimizer steps per epoch at batch 4.

Output tensors: `x [1, D, t, t]` float32 in [0,1] · `y [1, t, t]` · `u [1, t, t]` trimap.

---

## 6. Inference-time preprocessing (`engine/infer.py`)

Identical normalization and depth window as training. The segment is covered by a sliding window at
**stride t/2**, each tile's prediction weighted by a **Gaussian centred on the tile (σ = t/6)** and
accumulated, then divided by the accumulated weight and multiplied by the valid mask. Optional
**8-fold dihedral test-time augmentation** averages predictions over the 4 rotations × 2 flips,
inverse-transforming each before accumulation.

---

## 7. The test segment (`python -m scripts.prepare_infer`)

`flatboi` ships as 65 raw tifs plus a mask and **no labels**, and has no pre-stacked volume. The tool
stacks the tifs into a uint8 memmap one layer at a time and binarises the mask, writing
`proc_infer/<seg>/{vol.npy, mask.npy, meta.json}` in the layout the inference path expects. It is
scored **never** — qualitative figures only.

---

## 8. What differs between the three models

Nothing in §§1–7. The differences are inside the network:

| | UNETR++ | nnFormer | SegFormer3D |
|---|---|---|---|
| input tensor | `[B,1,32,t,t]` | identical | identical |
| labels | `ink_reg.npy` (earliest t64 rounds used `ink.npy`; reported runs use `ink_reg`) | `ink_reg.npy` (plus a separate, never-mixed `ink_reg2` round) | `ink_reg.npy` only |
| trimap / loss | `unc.npy` + `dice_bce_masked` | identical | identical |
| how depth enters | patch embedding `(1,4,4)`, i.e. depth preserved at stage 1 | same, `(1,4,4)` | depth-preserving stem `(3,7,7)/s(1,4,4)`; the isotropic original is ablation A1 |
| depth reduction inside | halved down the hierarchy | halved down the hierarchy | halved down the hierarchy; the key/value path can reduce depth separately (per-axis `sr`) |
| depth → 2D | shared `DepthCollapse(surface_relative)` | identical shared code | identical shared code |
| extra input, default OFF | — | — | a 2× wider, 2× downsampled **halo** as input channel 1 (contribution C1) |

The one genuinely model-specific preprocessing addition in the whole benchmark is SegFormer3D's
halo channel, and it is default-off, byte-identity-proven against the pre-change dataset, and used
only by the `sf_pdk_*` cells.

---

## 9. The depth-window ablation now running (`sf_depth65_t192_s42`)

Every cell above feeds the 32-layer window. The new cell feeds **all 65 layers** at the reference
tile, so the only change is how much of the depth axis the model sees:

- `data.depth: 32 → 65`; the window becomes `[0, 65)` (verified: the loader returns `x [1,65,192,192]`).
- **Parameter-almost-identical:** the backbone is unchanged at **4,437,568**; only the shared
  collapse's relative-offset table grows (63 → 129 entries), so the full model goes 4,439,985 →
  **4,440,051**, i.e. **+66 parameters**. This is as close to a pure "more input" comparison as the
  architecture allows.
- Stage-1 tokens **73,728 → 149,760** (2.03×); grids `(65,48,48) → (33,24,24) → (17,12,12) → (9,6,6)`.
- **Protocol caveat that must be reported:** with `depth == n_layers` the ±2 depth-shift
  augmentation has no room to move and becomes inert (verified: every draw returns `[0,65)`). The
  32-layer reference does use it. So the comparison is "32-layer window + depth jitter" versus
  "full stack, no jitter" — unavoidable, since a full stack cannot be shifted.
- Everything else byte-identical to `sf_base_t192_s42`: tile 192, `_reg_t192`, `ink_reg.npy`, u1
  masking, batch 4 / 1000 steps, 60-epoch cosine, seed 42.

A port fix was needed: the geometry contract required every axis to divide exactly, which 65 does
not (65 → 33 → 17 → 9). The real invariant for a **padded** patch-embed convolution is not exact
divisibility but (i) the standard ceil output size and (ii) that every input position is seen by at
least one kernel window. The check now tests that, and still refuses genuinely lossy geometry. The
strict divisibility rule remains where truncation can actually happen: the sequence-reduction
convolution, which has no padding. All 50 port tests still pass.

---

## 10. Reproducing the whole chain

```bash
source $WORK/vesuvius/.venv/bin/activate
cd $WORK/vesuvius/vesuvius-ink-bench
python -m data.prepare                                          # vol/ink/mask/meta
python -m scripts.register_labels                               # ink_reg.npy (+ report)
python -m scripts.make_uncertainty --r 4 --label-file ink_reg.npy   # unc.npy trimaps
python -m data.tiling --tile 192 --stride 96 --min-mask 0.5 \
       --label-file ink_reg.npy --suffix _reg_t192              # one manifest per rung
python -m scripts.prepare_infer                                 # test segment
```

**Deliberately NOT done, each measured harmful or rejected:** photometric augmentation · CLAHE ·
per-tile or global z-scoring · background detrending · depth flip · morphological cleanup of labels ·
geometric resampling of the sheet · word-box / annotation-anchored sampling · any use of the test
segment in selection.
