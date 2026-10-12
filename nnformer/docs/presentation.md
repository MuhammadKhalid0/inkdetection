# nnFormer for Ink Detection — Presentation Script

> **INSTRUCTIONS FOR THE SLIDE-GENERATING LLM**
> - This file is a **complete script**: each `## Slide N` block gives the slide title, the bullet content to render, speaker notes, and (where relevant) an exact figure path.
> - **Every number in this file is measured** and traceable to a file under
>   `/home/woody/iwi5/iwi5401h/nnFormer/`. Do **not** invent, round differently, or extrapolate numbers.
> - Wherever you see `**[ADD FIGURE]**` followed by a path, insert an image placeholder with that path as the caption/source. Do not generate synthetic charts unless a slide explicitly says "make a table/chart from the numbers below".
> - Run names in the tables are **real output directory names**: `$WORK/nnFormer/outputs/<run_name>/`, where `$WORK=/home/woody/iwi5/iwi5401h`. Keep them verbatim so the author can open each one.
> - Tone: MSc thesis progress presentation, technical audience (Pattern Recognition Lab).
> - 18 slides total.

**Project:** MSc thesis — Benchmarking 3D medical segmentation models for ink detection in carbonized Herculaneum papyrus (Scroll 5 / PHerc 0172).
**Benchmark position:** Model 1 = UNETR++ (complete, frozen) · **Model 2 = nnFormer (this deck)** · Model 3 = SegFormer3D (not started).
**Status date:** 2026-08-14.

---

## Slide 1 — What nnFormer is

**Title:** nnFormer — an interleaved convolution + attention U-shape

**Bullets:**
- nnFormer (Zhou et al., IEEE TIP 2023) — 3D medical segmentation network; published at ~150.5M parameters, trained on Synapse (18 CT volumes) and BraTS.
- Encoder–decoder with **four interleaved ideas**, all kept in our port:
  1. **Convolutional embedding** — 4× 3×3×3 convs (GELU + LayerNorm), not a linear patchifier.
  2. **LV-MSA** — Local Volume Multi-head Self-Attention: Swin-style *3D windows* with a learned 3D relative-position bias; shifted-window alternation between blocks.
  3. **GV-MSA** — Global Volume MSA at the bottleneck (attention over the full coarse grid).
  4. **Skip attention** — the decoder *queries* the encoder (encoder = K/V, decoder = Q) instead of concatenating skip connections. This is nnFormer's signature component.
- Source of record: official repo `282857341/nnFormer`, vendored read-only at `$WORK/nnFormer/reference/nnFormer/`.

**Speaker notes:** Position nnFormer as the "windowed local + global attention" family, in contrast to UNETR++'s EPA (paired channel/spatial attention). The benchmark question is exactly whether that architectural difference matters for ink.

---

## Slide 2 — Our port: nnFormer-Ink

**Title:** nnFormer-Ink — capacity ladder and cost

**Bullets:**
- Our implementation: `$WORK/nnFormer/src/nnformer_ink/` (`backbone.py`, `blocks.py`, `attention.py`, `windows.py`, `sampling.py`, `contrib.py`).
- Resolution schedule at t256 → S1 (32,64,64) → S2 (16,32,32) → S3 (8,16,16) → S4 (4,8,8) = bottleneck. Windows (4,8,8)/(4,8,8)/(4,4,4); **no padding paths** — divisibility asserted at build time for t ∈ {64,128,192,256}.
- Three **measured** capacity rungs (`$WORK/nnFormer/docs/bench_table.md`):

| Variant | params (t192) | GFLOPs t192 / t256 | ms/tile t192 / t256 | peak mem t192 / t256 |
|---|---|---|---|---|
| nnF-S | **19.87 M** | 171.8 / 305.6 | 25.8 / 36.5 | 0.77 / 1.26 GB |
| nnF-M | **79.11 M** | 506.6 / 900.9 | 33.6 / 59.5 | 1.54 / 2.38 GB |
| nnF-F | **177.74 M** | 1046.1 / 1860.3 | 47.8 / 86.4 | 2.54 / 3.73 GB |

- nnF-M is **parameter-matched to the deployed UNETR++ t256 (79.1 M)** — that is the fair-comparison rung.
- Params are essentially **tile-invariant** (window attention); the only t192→t256 change is the GV bottleneck bias table. This is *cleaner* than UNETR++, whose parameters grew 25M→79M→112M with tile size.

**Speaker notes:** Bench numbers are bf16 inference, batch 1, A100. Emphasise the tile-invariance point — it removes a confound UNETR++ had.

---

## Slide 3 — Domain shift, part 1: what the source model assumes

**Title:** Medical CT ≠ carbonized papyrus

**Bullets — the four assumptions that break:**
1. **Output geometry.** nnFormer: 3D volume in → **3D** multi-class mask out. Ink: 3D surface volume in → **2D** binary ink map out.
2. **Anisotropy.** Medical anatomy is near-isotropic, so the embedding halves depth. Our signal is a **thin band at z27–33** inside a 32-layer window — halving depth destroys it.
3. **Hardcoded geometry.** The official repo fixes window sizes and token counts to medical crop sizes; it silently mis-sizes on other shapes.
4. **Training recipe.** nnU-Net trainer (SGD, poly LR 0.01, CE+Dice, deep supervision) vs. our frozen benchmark protocol.

**Speaker notes:** Frame this as "the port is a research contribution, not a copy" — each break needed a decision, and each decision is recorded in `$WORK/nnFormer/docs/PORT_LOG.md`.

---

## Slide 4 — Domain shift, part 2: how we adapted it

**Title:** The adaptation map

| Medical assumption | Ink adaptation |
|---|---|
| 3D multi-class output | Backbone returns a **feature volume** `[B,16,D,t,t]`; the *shared* Class-A `DepthCollapse` (surface-relative) + `Head` produce `[B,1,t,t]`. Identical wrapper contract as UNETR++. |
| Depth-halving embedding | **Depth-preserving embedding**, patch (1,4,4) → stage-1 grid (32, t/4, t/4). Depth is reduced only inside the hierarchy (32→16→8→4). |
| Cubic scalar windows | **Anisotropic tuple windows** per stage, per-axis clamp (depth 32 vs in-plane t/4 is anisotropic). |
| Deep supervision ON | **DS OFF for every benchmark model** — an explicit *fairness* decision (UNETR++ ran DS-off; a silent asymmetry would confound the headline). Replaced by a full-resolution conv skip, same as UNETR++. |
| nnU-Net recipe | Benchmark protocol: AdamW 1e-4, wd 3e-5, warmup 3 + cosine 60 epochs, `dice_bce_masked` (u1 trimaps), aug hflip/vflip/rot90 + z_shift ±2, z_flip OFF, bf16, seeds 42/43/44. |
| Synapse pretrained weights | **From scratch** — after adaptation almost no tensor shapes survive; benchmark-equalized initialization. |

**Speaker notes:** The key structural point is the wrapper contract: only the backbone changes between benchmark models, so any difference is attributable to architecture.

---

## Slide 5 — Domain shift, part 3: a real port bug we found

**Title:** The relative-position-bias bug (why porting carefully matters)

**Bullets:**
- The official nnFormer computes its 3D relative-position index with strides `*= 3*ws-1` and `*= 2*ws-1` — valid only for **cubic** windows.
- With anisotropic windows this **collides distinct (Δz, Δy, Δx) offsets onto the same bias-table entry** — the model silently loses the ability to distinguish depth offsets from in-plane offsets.
- Our port uses the standard Swin-3D strides `(2wh−1)(2ww−1)` and `(2ww−1)`, with bounds assertions and a unit test.
- This is the same *class* of defect that was found and fixed in the UNETR++ port (hardcoded medical token counts) — an independent confirmation that published 3D medical repos do not transfer safely.
- All 9 deviations from the original are tabulated in `$WORK/nnFormer/docs/PORT_LOG.md` (D1–D9).

**Speaker notes:** This is a good "methods rigor" slide — it shows the port was verified rather than assumed. Mention the 11-check CPU test suite (`tests/test_smoke.py`), which includes exact-reduction proofs for both contributions.

---

## Slide 6 — Ablation study, part 1: stock architecture screens

**Title:** Which nnFormer components matter? (t192, seed 42)

**All runs below are directories under `$WORK/nnFormer/outputs/`.**
**Labels: `ink_reg2.npy` (the re-registered label set) — deltas are only valid *within* this block.**

| Run (output dir) | Toggle vs. base | raw F0.5 | post F0.5 | Δraw vs base |
|---|---|---|---|---|
| `n_base_t192_s42` | — (reference) | 0.7780 | 0.7880 | — |
| `n_mean_t192_s42` | collapse: surfrel → mean | 0.7739 | 0.7953 | −0.0041 |
| `n_abs_t192_s42` | collapse: surfrel → learned-absolute | 0.7682 | 0.7847 | −0.0098 |
| `n_skipconv_t192_s42` | skip **attention** → conv skip | 0.7778 | 0.7850 | −0.0002 |
| `n_lvonly_t192_s42` | GV bottleneck → local windows | 0.7833 | 0.7910 | +0.0053 |
| `n_embedz2_t192_s42` | embed depth stride 1 → 2 | 0.7665 | 0.7951 | −0.0115 |
| `n_lr5e5_t192_s42` | LR 1e-4 → 5e-5 | 0.7785 | 0.7937 | +0.0005 |

- **Noise floor: single-seed |Δ| ≤ 0.008 is noise** (measured in the UNETR++ study, inherited).
- **Verdict: every stock toggle is inside or barely outside the noise band.** Not even removing skip attention — nnFormer's signature component — changes the result (−0.0002).

**Speaker notes:** State plainly that this replicates the UNETR++ finding on a second architecture: pipeline/data dominate, stock architecture toggles are largely interchangeable at this scale.

---

## Slide 7 — Ablation study, part 2: capacity × data density

**Title:** Does the 178M-parameter published model help? (t256, labels `ink_reg.npy`)

| Run (output dir) | Capacity | Train tiles | raw | post | best epoch | Overfit signature |
|---|---|---|---|---|---|---|
| `nnf_s_t256_s42` | S (19.9M) | 21,360 (stride 128) | 0.7677 | 0.7868 | 59 | no |
| `nnf_m_t256_s42` | M (79.2M) | 21,360 | 0.7613 | 0.7753 | 44 | mild |
| `nnf_f_t256_s42` | F (177.8M) | 21,360 | 0.7671 | 0.7806 | **37** | **YES** |
| `nnf_f_d64_t256_s42` | F (177.8M) | **84,523** (stride 64) | 0.7652 | 0.7807 | 53 | reduced |
| `nnf_s_d64_t256_s42` | S (19.9M) | **84,523** | 0.7747 | 0.7884 | 50 | no |
| `nnf_s_d64_t256_e90_s42` | S, 90 epochs | 84,523 | 0.7809 | 0.8102 | 54 | no |

**Findings:**
- **Capacity is a dead lever**: S / M / F spread = 0.006 raw, inside the noise floor, with the *smallest* model nominally best.
- **nnF-F shows the classic overfit signature**: peaked at epoch 37 while train loss kept falling a further 15%.
- **Dense stride-64 sampling** (4× tiles, same 4000 samples/epoch) both improved S and *mitigated* F's overfit (best epoch 37 → 53).
- **Epoch budget:** 90 epochs gave +0.0062 raw over the same-seed 60-epoch run — inside noise. 60 epochs confirmed adequate, matching the UNETR++ precedent.

**Speaker notes:** This is the "150M parameters on 9 segments" answer the plan asked for. The overfit diagnosis is evidence-based (train/val divergence), not asserted.

---

## Slide 8 — Ablation study, part 3: our two contributions + prior art

**Title:** Contributions — SR-PB, CSSA, and a vFFC prior-art cell (t192, labels `ink_reg.npy`)

| Run (output dir) | What it adds | raw | post | Δraw vs stock **3-seed mean (0.7657)** |
|---|---|---|---|---|
| `n_base_t192_r1_s42` | stock reference, seed 42 | 0.7581 | 0.7747 | — |
| `n_base_t192_r1_s43` | stock reference, seed 43 | 0.7687 | 0.7919 | — |
| `n_base_t192_r1_s44` | stock reference, seed 44 | 0.7703 | 0.7844 | — |
| **stock band** | 3 seeds | **0.7657 ± 0.0066** | **0.7836 ± 0.0086** | — |
| `n_cssa_t192_r1_s42` | **C2 / CSSA** — decoder also attends into coarse encoder scales (zero-init gate) | 0.7618 | 0.7766 | −0.0039 |
| `n_srpb_t192_r1_s42` | **C1 / SR-PB** — surface-relative additive depth bias on attention logits | 0.7708 | 0.7796 | |
| `n_srpb_t192_r1_s43` | SR-PB, seed 43 | 0.7724 | 0.7852 | |
| `n_srpb_t192_r1_s44` | SR-PB, seed 44 | 0.7733 | 0.7844 | |
| **SR-PB band** | 3 seeds | **0.7722 ± 0.0013** | **0.7831 ± 0.0030** | **+0.0065** raw / **−0.0006** post — inside ±0.008 |
| `n_vffc_t192_r1_s42` | **prior art** — GV-MSA → 3× vFFC spectral blocks (Quattrini et al. 2023) | **0.7844** | **0.7992** | **+0.0187** (n=1) |
| `n_base_t128_r1_s42` / `n_cssa_t128_r1_s42` | CSSA 2×2 low rung | 0.7518 / 0.7539 | 0.7627 / 0.7642 | (2×2 cells) |

**Design discipline (state this explicitly):**
- Both contributions are **exact no-ops at initialization** — verified bitwise (max |Δ| = 0.0) in `tests/test_smoke.py`, with a second test proving the branch is *live* once its parameters are non-zero. So any measured delta is attributable to the component, not to a changed initialization.
- SR-PB implementation note: naively re-anchoring the *pairwise* relative-position index on ŝ is a **no-op** (Δz = z_i − z_j is shift-invariant, so a common ŝ cancels). The working form is a **key-side additive** bias b(z − ŝ).
- **Why the Δ column uses the band, not seed 42:** stock seed 42 (0.7581) turned out to be the **weakest** stock seed (s43 0.7687, s44 0.7703). Any delta measured against it alone is inflated by ~+0.008 — exactly the noise floor. The CSSA 2×2 on slide 10 is a paired seed-42 design *by construction* and is reported as such.

**Speaker notes:** Flag the vFFC caveat here (it carries 65.0M params vs 20M stock) — full discussion on slide 16.

---

## Slide 9 — Results, part 1: the benchmark row

**Title:** nnFormer benchmark number (t256, 3 seeds, labels `ink_reg.npy`)

| Configuration | Seeds | raw F0.5 | post F0.5 | best seed (post) |
|---|---|---|---|---|
| **stock** `nnf_s_d64_t256_s{42,43,44}` | 3 | 0.7825 ± 0.0117 | 0.7981 ± 0.0123 | 0.8120 |
| **c1 = stock + SR-PB** `nnf_c1_t256_s{42,43,44}` | 3 | 0.7801 ± **0.0037** | **0.8005** ± 0.0102 | **0.8122** |
| *UNETR++ `c1_t256` (reference)* | *3* | *0.8111 ± 0.0039* | *0.8190 ± 0.0062* | *0.8261* |

**With 8-fold dihedral TTA** (`$WORK/nnFormer/outputs/tta/`):

| Run | raw (TTA) | post (TTA) |
|---|---|---|
| `nnf_c1_t256_s42` | 0.7906 | **0.8133** |
| `nnf_s_d64_t256_s44` | 0.8006 | 0.8126 |

**Read:**
- Best nnFormer post-filter + TTA = **0.8133**; UNETR++ = 0.8190 → **gap ≈ 0.006–0.019** depending on whether you compare best seeds or 3-seed means.
- **SR-PB cuts raw seed variance ~3×** (±0.0037 vs ±0.0117) while leaving the mean statistically unchanged — a stability property worth reporting.

**Speaker notes:** Be precise: the 3-seed post means are 0.8005 (nnFormer) vs 0.8190 (UNETR++) = −0.0185. Best-seed comparison is much closer. Report both, do not cherry-pick.

---

## Slide 10 — Results, part 2: contribution verdicts against pre-registered gates

**Title:** What passed, what didn't — decided by rules fixed *before* the runs

| Contribution | Pre-registered gate | Measured | Verdict |
|---|---|---|---|
| **C1 / SR-PB** | δ=0 expected neutral; real readout = depth-offset stress tails at \|δ\|≥4 | **Band vs band: +0.0065 raw / −0.0006 post** (stock 0.7657 ± 0.0066 vs SR-PB 0.7722 ± 0.0013). Stress mean **+0.035**, tails **+0.096 / +0.077** — *measured against base seed 42 only* | **NEUTRAL on F0.5** (as pre-registered). What survives: seed variance ↓5× at t192, ↓3× at t256; Inkalyzer depth-broadening (slide 14). Robustness claim still rests on an n=1 baseline — see slide 16 |
| **C2 / CSSA** | must close ≥ half the stock t128→t192 gap (and ≥ +0.02) | closed **32%** (+0.0020 at t128); Δ at t192 +0.0037 — both inside noise | **FAIL on F0.5** — *but mechanism confirmed*, see slide 15 |
| **Prior art / vFFC** | expected neutral vs GV-MSA | **+0.0263 raw / +0.0245 post** | **Unexpected positive — confounded**, see slide 16 |

**Note on the SR-PB verdict (state this aloud):** an earlier reading of **+0.013…+0.015** compared SR-PB's 3-seed band against stock **seed 42 alone**, which turned out to be the weakest stock seed. Band vs band, the accuracy effect is inside the noise floor. This is the same seed-42 artifact the UNETR++ study documented for DA-EPA, recurring on a second architecture — and the pre-registered band-vs-band rule caught it again.

**Depth-offset stress (frozen-τ F0.5), from `$WORK/nnFormer/outputs/zshift_stress/zshift_stress.json`** — *baseline row is seed 42 only; SR-PB rows are two seeds*:

| Run | δ=−8 | −4 | 0 | +4 | +8 | mean |
|---|---|---|---|---|---|---|
| `n_base_t192_r1_s42` | 0.332 | 0.719 | 0.758 | 0.677 | 0.299 | 0.599 |
| `n_srpb_t192_r1_s42` | 0.428 | 0.735 | 0.771 | 0.700 | 0.376 | 0.633 |
| `n_srpb_t192_r1_s43` | 0.401 | 0.742 | 0.772 | 0.710 | 0.401 | 0.641 |
| `n_cssa_t192_r1_s42` | 0.388 | 0.720 | 0.762 | 0.704 | 0.385 | 0.627 |

**Speaker notes:** The gates were applied *mechanically* by `$WORK/nnFormer/scripts/collect_results.py`, which writes `docs/RESULTS.md`. Stress values are reproduced across two SR-PB seeds — the effect is not a single-seed artifact.

---

## Slide 11 — Visual results, model 1 of 2

**Title:** Best model — `nnf_c1_t256_s42` (post F0.5 = 0.8122, +TTA 0.8133)

- **Validation segment** 20241113090990, τ = 0.85 — 4-panel: probability | thresholded | ground truth | TP-green / FP-red / FN-blue overlay.
  **[ADD FIGURE]** `$WORK/nnFormer/outputs/nnf_c1_t256_s42/figs/20241113090990_tau0.85.png`
- **Test segment** (flatboi, no labels — qualitative only), τ = 0.85.
  **[ADD FIGURE]** `$WORK/nnFormer/outputs/nnf_c1_t256_s42/figs/20251115002740-auto_grown_20251115002740308_0_flatboi_tau0.85.png`
- Config: capacity S (19.9M), tile 256, dense stride-64 manifest (84,523 tiles), **SR-PB on**, u1 masked loss, z_flip off.

**Speaker notes:** This is the configuration that combines everything adopted: best capacity + dense sampling + the one contribution that passed its gate. Note the test segment is scored qualitatively only — no labels exist, and no threshold was ever selected on it.

---

## Slide 12 — Visual results, model 2 of 2

**Title:** Best stock model — `nnf_s_d64_t256_s44` (post F0.5 = 0.8120, +TTA 0.8126)

- **Validation segment** 20241113090990, τ = 0.85.
  **[ADD FIGURE]** `$WORK/nnFormer/outputs/nnf_s_d64_t256_s44/figs/20241113090990_tau0.85.png`
- **Test segment** (flatboi), τ = 0.85.
  **[ADD FIGURE]** `$WORK/nnFormer/outputs/nnf_s_d64_t256_s44/figs/20251115002740-auto_grown_20251115002740308_0_flatboi_tau0.85.png`
- Identical to slide 11's config **minus SR-PB** — so the two slides are a clean visual A/B of the contribution at deployment scale.

**Optional extra:** training curves for both, showing convergence and no overfit:
`.../nnf_c1_t256_s42/figs/train_curves.png` and `.../nnf_s_d64_t256_s44/figs/train_curves.png`

**Speaker notes:** Present 11 and 12 side by side if the slide format allows — the point is that the two best models are visually near-identical, which is itself the honest message (0.8122 vs 0.8120).

---

## Slide 13 — Inkalyzer, part 1: where the model looks in depth

**Title:** Depth attribution — the trust result replicates

**[ADD FIGURE]** `$WORK/nnFormer/probes/inkalyzer_compare/depth_profiles.png`

**Method:** Inkalyzer (input-space Information-Bottleneck attribution) learns a per-depth-layer gate that preserves the model's own prediction with minimal information — the converged gate at layer z = how much the model needs real data at that depth. 24 ink-rich validation tiles per model.

| Run | peak z | mass within ±2 | normalized entropy |
|---|---|---|---|
| `n_base_t192_r1_s42` | 32 | 0.501 | 0.792 |
| `n_srpb_t192_r1_s42` | **33** | **0.444** | **0.828** |
| `n_cssa_t192_r1_s42` | 32 | 0.496 | 0.799 |
| `nnf_s_d64_t256_s42` | 32 | 0.504 | 0.784 |
| `nnf_c1_t256_s42` | 33 | 0.483 | 0.792 |

**Finding 1 — trust result, now architecture-independent:** all five nnFormer models produce a single unimodal bump peaking at **z32–33** with ≈0 at the window edges. All 15 UNETR++ models peaked at z29–34. Two structurally different transformers agreeing makes this a statement about **the data**, not about one model — and it validates the `[16,48)` depth window twice over.

**Speaker notes:** Caveat to state aloud: cross-*tile* entropy comparisons are a method artifact (larger tiles average more heterogeneous pixels and admit flatter gates), so compare t192 rows to t192 rows only.

---

## Slide 14 — Inkalyzer, part 2: SR-PB's mechanism

**Title:** SR-PB is *visible* in the depth profile — a new result

**Bullets:**
- `n_srpb_t192_r1_s42` is the outlier on every concentration axis: **highest entropy (0.828 vs 0.792)**, **lowest concentration (mass ±2 = 0.444 vs 0.501)**, peak shifted one layer deeper, and visibly elevated shallow mass at z20–27 in the figure.
- Interpretation: **SR-PB makes the model read a *wider* depth band.** A model integrating evidence over more depth degrades more gracefully when the depth window shifts — which is exactly the measured robustness (+0.096 / +0.077 in the stress tails, slide 10).
- **Why this matters for the thesis:** in the UNETR++ study the equivalent surface-relative mechanism was **invisible** to this metric (`abl_surfrel`: entropy 0.899 = 0.899, peak z32 = z32), and that had to be recorded as a limitation of the method. nnFormer's SR-PB **does** move the profile — so the limitation is now resolved, and we have a mechanism statement, not just a delta.
- **Internal consistency check:** CSSA leaves the depth profile essentially untouched (entropy 0.799 vs 0.792) — correct, because CSSA is a *spatial* mechanism; its signature appears in the ERF instead (slide 15). Each contribution registers on its own axis and not the other.

**Optional figure:** per-model 3D saliency, e.g. `$WORK/nnFormer/outputs/n_srpb_t192_r1_s42/inkalyzer/attr3d_tile0.png` — **qualitative only** (16-px attribution windows).

---

## Slide 15 — Interpretation: what the ablations mean

**Title:** Why nnFormer trails UNETR++ — a mechanistic answer

**1. Effective receptive field is the binding constraint** (`$WORK/nnFormer/probes/erf/erf_stats.json`):

| Model | tile | r50 | r90 | **r90 / tile** |
|---|---|---|---|---|
| `n_base_t192_r1_s42` | 192 | 63 | 97 | **51%** |
| `n_cssa_t192_r1_s42` | 192 | **75** | **107** | 56% |
| `nnf_s_d64_t256_s42` | 256 | 59 | 115 | **45%** |
| `nnf_c1_t256_s42` | 256 | 62 | 124 | 48% |
| *UNETR++ (all rungs)* | *64–256* | — | *35 / 68 / 105 / 134* | *~53% constant* |

- UNETR++ consumed a **constant ~53%** of whatever tile it was given — context scaled, performance scaled (+0.055 for t128→t192).
- nnFormer's windowed attention **bounds** the receptive field and the fraction **shrinks** with tile size (51% → 45%).
- **Consequence:** nnFormer's own tile ladder is nearly flat — stock t128→t192 = **+0.0063** (vs UNETR++'s +0.055). It cannot cash in the context that made UNETR++ strong.
- **Headline sentence:** *local windowed attention caps context utilization; EPA's paired channel/spatial attention does not.*

**2. CSSA: mechanism confirmed, metric neutral.** CSSA measurably **grew** the ERF (r50 63→75, +19%) and improved robustness (stress mean +0.028) and recall (0.529→0.601), but F0.5 moved only +0.004. The honest reading is "the intervention did what it was designed to do and did not convert" — not "it failed". Note the 2×2 was also underpowered by construction, because nnFormer's t128→t192 gap was only +0.0063 to begin with.

**3. Error anatomy** (`$WORK/nnFormer/probes/error_anatomy/`) — where the missing recall lives:

| Run | recall | dead-label | boundary ≤4px | low-contrast | **unexplained** |
|---|---|---|---|---|---|
| `nnf_c1_t256_s42` | 0.588 | 0.009 | 0.137 | 0.133 | **0.721** |
| `nnf_s_d64_t256_s42` | 0.527 | 0.008 | 0.130 | 0.124 | 0.738 |
| `n_base_t192_r1_s42` | 0.537 | 0.020 | 0.127 | 0.111 | 0.742 |
| *UNETR++ `t256_lean_s42`* | *0.578* | *0.022* | *0.135* | *0.127* | *0.716* |

The failure *profile* is nearly identical across the two architectures — the residual problem is shared, not model-specific.

---

## Slide 16 — The final runs, the revised verdict, and what remains open

**Title:** The last three runs, what they changed, and what remains open

**Completed (SLURM array `4006120`, A40, 2026-08-14; analysis `4006121` completed):**

| Run (output dir) | Result | What it changed |
|---|---|---|
| `n_base_t192_r1_s43` | 0.7687 raw / 0.7919 post | Completed the stock t192 band → stock = **0.7657 ± 0.0066** |
| `n_base_t192_r1_s44` | 0.7703 raw / 0.7844 post | **Revealed seed 42 (0.7581) as the weakest stock seed → SR-PB's accuracy verdict revised from "+0.013" to "neutral"** |
| `n_srpb_aux_t192_r1_s42` | **0.7607** raw / 0.7747 post | SR-PB + supervised surface estimate — *below* both the stock band and plain SR-PB. **Supervising the surface hurts**, replicating UNETR++'s aux-ŝ result (centre −0.050 there). Gate was met (+0.042/+0.025 tails); hypothesis refuted; closed. |

**Open / honest caveats:**
0. **SR-PB's robustness curve is n=1 on the baseline side.** The stress table on slide 10 compares SR-PB (2 seeds) against `n_base_t192_r1_s42` **only** — the weak seed. Tail gaps (+0.096/+0.077) are large relative to the 0.0066 seed spread, so the effect *probably* survives, but two inference-only stress runs on base seeds 43/44 are required before it is claimed.
1. **vFFC's result is confounded — twice.** `n_vffc_t192_r1_s42` gave the best t192 number in the study (+0.0263 raw), but it carries **65.0M parameters vs 20M** stock, and it is a **single seed** — which, after what seed 42 just did to the SR-PB verdict, is not enough on its own. Against the stock *band* the delta is +0.0187 (still outside noise). The capacity-flatness of slide 7 only partially covers the parameter gap (that experiment varied *attention width*, not operator class). Needed before any claim: seeds 43/44 **and** a parameter-matched control.
2. **`n_sads` (stroke-aware deep supervision) not implemented.** It requires multi-scale decoder outputs plus a multi-scale loss in the *shared* engine that UNETR++'s frozen results also depend on. It is the plan's lowest-value item and is excluded from the cross-model headline by construction. Recommendation: report the DS question as settled at the *protocol* level (DS off for all models) rather than empirically.
3. **Label set caveat.** The t192 stock screens on slide 6 use `ink_reg2.npy`; everything else uses `ink_reg.npy`. **Never mix deltas across the two label versions** — they are different yardsticks.

---

## Slide 17 — Cross-model comparison: do UNETR++ and nnFormer agree?

**Title:** Two architectures, the same conclusions

| Question | UNETR++ (model 1) | nnFormer (model 2) | Agree? |
|---|---|---|---|
| Do stock architecture toggles matter? | No — DA-EPA, per-pixel warp, warp-stab, loss shapes all ≈0 at deployment scale | No — skip-attn −0.0002, GV +0.0053, LR +0.0005, collapse variants ≤ \|0.010\| | **YES** |
| Does capacity help? | t320 (112M) overfit on 13.3k tiles | S 19.9M ≈ M 79.2M ≈ F 177.8M, F overfits | **YES** |
| Where in depth does the model look? | Unimodal bump, peak z29–34, ≈0 at edges (15 models) | Unimodal bump, peak z32–33, ≈0 at edges (5 models) | **YES** |
| Is fine "crackle" texture the signal? | No — 2–8px bands worth ≈0 | No — 2–4px band worth −0.003 | **YES** |
| Does tile size / context dominate? | **Yes** — +0.16 total, +0.055 for t128→t192 | **No** — only +0.0063 for t128→t192 | **NO** |
| Is the residual error the same? | 71.6% "unexplained", recall 0.578 | 72.1% "unexplained", recall 0.588 | **YES** |

**Speaker notes:** Five of six agree. The one disagreement is the important one, and it is *explained* by the ERF measurement on slide 15 — not left as a mystery.

---

## Slide 18 — Where the benchmark stands

**Title:** Summary and next steps

**Where nnFormer landed:**
- Best nnFormer: **0.8133** post-filter + TTA (`nnf_c1_t256_s42`); 3-seed band **0.8005 ± 0.0102**.
- UNETR++: **0.8190 ± 0.0062** post (best single 0.8261).
- **nnFormer does not beat UNETR++**, and the reason is measured rather than assumed: **bounded effective receptive field** (45–51% of tile, shrinking with scale, vs UNETR++'s constant ~53%).

**What the nnFormer chapter contributes:**
1. **SR-PB** — a surface-relative attention bias that is **neutral on F0.5 band-vs-band** (+0.0065, inside noise) but cuts seed variance **5× at t192 / 3× at t256** and is **visible in depth attribution** (it widens the depth band the model reads), resolving an open limitation of the XAI method. Its depth-offset robustness (+0.035 mean vs base seed 42) is promising but awaits a band-vs-band stress replication.
1b. **A methodological result**: the seed-42 artifact documented in the UNETR++ study **recurred on a second architecture** — reported SR-PB deltas of +0.013–0.015 collapsed to +0.0065 once the baseline had its own 3-seed band. The pre-registered rule caught it both times.
2. **CSSA** — a mechanism-confirmed / metric-neutral result: the ERF genuinely grew (+19% r50), recall rose, F0.5 did not follow.
3. **A cross-architecture replication** of the study's central claim: data, scale and protocol dominate architecture.
4. **A port-level finding**: the published relative-position-bias indexing is unsafe for anisotropic windows.

**Immediate next steps (none block the chapter):**
1. Two inference-only stress runs on `n_base_t192_r1_s43/s44` → make SR-PB's robustness claim band-vs-band.
2. vFFC seeds 43/44 + a parameter-matched control → settle the study's best unexplained number, or document it as open.
3. Upstream report to Fei Wu — label findings from the re-registration round (`$WORK/nnFormer/rereg/summary.md`).
4. **Begin benchmark model 3: SegFormer3D.**

---

## Appendix — where everything lives (for the author)

| Artifact | Path (relative to `$WORK/nnFormer/`, `$WORK=/home/woody/iwi5/iwi5401h`) |
|---|---|
| Port + contributions source | `src/nnformer_ink/` (`contrib.py` = SR-PB, CSSA, vFFC) |
| Unit tests (11 checks) | `tests/test_smoke.py` |
| Port deviations D1–D9, verdict log | `docs/PORT_LOG.md` |
| Auto-generated results + gate verdicts | `docs/RESULTS.md`, `docs/results.json` |
| Efficiency table | `docs/bench_table.md` |
| All run outputs | `outputs/<run_name>/{summary.json, postprocess.json, train_log.csv, figs/}` |
| TTA | `outputs/tta/` |
| Depth-offset stress | `outputs/zshift_stress/zshift_stress.json` |
| ERF / frequency / anatomy / Inkalyzer | `probes/{erf,freqband,error_anatomy,inkalyzer_compare}/` |
| Label re-registration round | `rereg/summary.md`, `rereg/figs/` |
| Plan of record | `~/plan(md_files)/nnformer_plan.md` |
