# SegFormer3D for Ink Detection — Presentation Script

> **INSTRUCTIONS FOR THE SLIDE-GENERATING LLM**
> - This file is a **complete script**: each `## Slide N` block gives the slide title, the bullet
>   content to render, speaker notes, and (where relevant) an exact figure path.
> - **This deck contains NO results by design.** It presents the model, the domain shift, the
>   contributions and the ablation design only. Where a slide names a run (e.g. `sf_osr_t192_s42`),
>   that is a **pointer to an output directory** the author will open live — do **not** invent,
>   infer, or add any accuracy numbers, deltas, or verdicts anywhere in the deck.
> - Numbers that DO appear are architectural facts (parameter counts, grid sizes, FLOPs, tile
>   sizes). Every one is measured and traceable to `$WORK/segFormer/`. Do not change or round them.
> - `**[ADD FIGURE]**` entries come in **two kinds**, and they are handled differently:
>   - **A file path** (`$WORK/segFormer/outputs/...`): leave an empty image placeholder and put the
>     exact path beneath it as a small muted caption. The author inserts these manually.
>     **Never redraw or simulate these — they are real experimental output.**
>   - **A description ending "make this as a shape diagram" / "Build as shapes"**: you SHOULD build
>     it natively from PowerPoint shapes, arrows and text boxes, in the template's colours.
> - Beyond those, you may also **turn prose into a diagram or a native table wherever that
>   communicates better than a bullet list** — this is encouraged. What you may not do is invent
>   content or data that is not in the script.
> - Run names are **real output directory names**: `$WORK/segFormer/outputs/<run_name>/`, where
>   `$WORK=/home/woody/iwi5/iwi5401h`. Keep them verbatim so the author can open each one.
> - Tone: MSc thesis progress presentation, technical audience (Pattern Recognition Lab).
> - **16 slides total.** Keep bullets short on the slide; put the prose in speaker notes.

**Project:** MSc thesis — Benchmarking 3D medical segmentation transformers for ink detection in
carbonized Herculaneum papyrus (Scroll 5 / PHerc 0172).
**Benchmark position:** Model 1 = UNETR++ (complete, frozen) · Model 2 = nnFormer (complete, frozen)
· **Model 3 = SegFormer3D (this deck)**.

---

## Slide 1 — Where we are, and what this model is for

**Title:** Model 3 of 3 — the efficiency contender

**Bullets:**
- Benchmark contract: identical data, split, labels, recipe, inference, post-processing, metrics
  and probes. **Only the backbone changes.**
- UNETR++ ✅ → nnFormer ✅ → **SegFormer3D** (in progress).
- SegFormer3D (Perera et al., CVPRW 2024): **~4.5M parameters** — the smallest model in the
  benchmark by a wide margin.
- The question it exists to answer: **does a tiny model lose anything on this task?**
- Three properties make it the *only* model that can run certain experiments:
  1. attention is **global**, not windowed;
  2. **no positional encoding** anywhere;
  3. parameter count is **completely independent of tile size**.

**Speaker notes:** Frame the chapter as the closing argument of the thesis. The central claim so
far is that data, scale and protocol dominate architecture. The sharpest test of that claim is a
model an order of magnitude smaller. Property 3 is what lets this chapter run the tile ladder
without the capacity confound that made chapter 1's largest rung uninterpretable.

---

## Slide 2 — The family tree (why the architecture looks like this)

**Title:** ViT → PVT → SegFormer (MiT) → SegFormer3D

**Bullets:**
- **ViT (2020):** image → patches → tokens + positional encoding → plain transformer.
  Problems: single scale, and attention cost is O(N²) in the token count.
- **PVT (2021):** adds a **4-stage pyramid** (multi-scale), and **spatial-reduction attention** —
  shrink the keys/values, keep the queries at full resolution.
- **SegFormer / MiT (2021):** adds **overlapping** patch embedding (no seams between patches), and
  **Mix-FFN** — a depthwise convolution inside the feed-forward block that *replaces* positional
  encoding. Decoder is deliberately minimal ("all-MLP").
- **SegFormer3D (2024):** the same design in 3D, for volumetric medical segmentation.

**Speaker notes:** Each step fixes a problem the previous one created. Emphasise spatial-reduction
attention — it is where our main contribution attaches, two slides later.

**[ADD FIGURE]** a simple 4-box lineage diagram (ViT → PVT → SegFormer → SegFormer3D), one line of
"what it added" under each box. No file — make this as a shape diagram.

---

## Slide 3 — Architecture overview

**Title:** Four encoder stages, one all-MLP decoder

**Bullets:**
- **Encoder:** 4 stages. Each = overlapping patch embedding → *n* transformer blocks → LayerNorm.
  Resolution drops, channels widen: `[32, 64, 160, 256]` channels, `[1, 2, 5, 8]` heads,
  `[2,2,2,2]` blocks.
- **Each block** (pre-norm, residual): sequence-reduced self-attention → Mix-FFN.
- **Decoder (all-MLP):** project every stage to 256 channels → resize all to the finest grid →
  concatenate → one 1×1×1 convolution to fuse.
- **Final path:** learned upsample back to full resolution + a full-resolution convolutional skip
  from the raw input.
- Output of the backbone is a **feature volume**, not a prediction (see slide 8).

**[ADD FIGURE]** architecture block diagram — encoder stages with grid sizes, decoder fusion, final
path. Build as shapes; grid sizes to use are on slide 4.

---

## Slide 4 — The key mechanism: sequence-reduced attention

**Title:** Compress the keys, not the queries

**Bullets:**
- Plain attention compares every token to every other: at our stage 1, **131,072 tokens** → an
  attention matrix of ~17 billion entries per head. Impossible.
- SegFormer3D's fix: keys and values come from a **strided convolution** over the token grid;
  queries stay at full resolution.
  - `Conv3d(C, C, kernel=r, stride=r)` → cost becomes `N × M` instead of `N × N`.
  - Our default reduction `(1,4,4)` → **8,192 keys** instead of 131,072.
- Three consequences, and everything else follows from them:
  1. the reduction is a **learned** summary, not pooling;
  2. the **key grid is decoupled from the query grid** — nothing requires the keys to summarise the
     same spatial extent as the queries;
  3. attention remains **global** — every query still sees the whole tile.
- Point 2 is the opening our main contribution uses (slide 12).
- Point 3 is the contrast with nnFormer, whose windowed attention bounded how far information
  could travel.

**Speaker notes:** This is the slide to spend time on. If the audience understands only one
mechanism from this deck, it should be this one.

---

## Slide 5 — Mix-FFN: a network with no positional encoding

**Title:** Position comes from one convolution's zero padding

**Bullets:**
- Mix-FFN = `Linear(C→4C)` → **3×3×3 depthwise convolution** → GELU → `Linear(4C→C)`.
- There is **no positional encoding anywhere in this network**.
- The published argument: a *padded* convolution already leaks absolute position (a token at the
  tile border sees zeros; one in the middle does not), so an explicit encoding is unnecessary —
  and removing it means the model does not break when input resolution changes between training
  and inference.
- **Why this matters for us:** the argument assumes translation invariance on every axis.
  - In-plane: true — our tiles are cut on a blind grid.
  - In depth: **false** — the ink sits at a physically fixed depth index.
- This becomes contribution **C2** (slide 13).

---

## Slide 6 — The all-MLP decoder

**Title:** After the encoder, no spatial reasoning at all

**Bullets:**
- Per stage: `Linear(C_i → 256)` → trilinear resize to the finest grid → concatenate (1024 ch) →
  `Conv3d(1024→256, kernel 1)` + BatchNorm + ReLU.
- **Every operation is pointwise in space.** A 1×1×1 convolution mixes channels at a location,
  never across locations. The only spatial movement is fixed interpolation.
- The published justification: the transformer encoder's receptive field is already large, so the
  decoder needs to do no further spatial reasoning.
- Our thesis **measures** receptive fields, so we can check that premise rather than assume it.
- This becomes contribution **C4** (slide 14).

---

## Slide 7 — The three published claims = the spine of this chapter

**Title:** We audit the architecture's own claims

**Bullets:**

| # | The paper claims | It assumes | Ink violates it because |
|---|---|---|---|
| 1 | compressed keys lose nothing | volumes are redundant and near-isotropic | our volume is 32 × t × t; only ~6 depth layers carry signal |
| 2 | Mix-FFN removes the need for positional encoding | all axes are translation-invariant | in-plane yes — **in depth no** |
| 3 | an all-MLP decoder suffices (large receptive field) | the encoder really does see far enough | we can measure receptive fields directly |

- The chapter is therefore **an audit, not a shopping list**: one test per claim, each in a setting
  that violates the assumption behind it — plus one new mechanism that attacks the thesis's
  dominant lever (context).
- This is the answer to *"why these four contributions and not four arbitrary tricks?"*

---

## Slide 8 — The domain shift, part 1: what the data actually is

**Title:** Why medical assumptions do not transfer

**Bullets:**
- A **segment** = a virtually unrolled sheet of papyrus: **65 depth layers** of greyscale CT.
- We keep a **32-layer window `[16,48)`**; the ink signal lives at **depth 27–33** — roughly
  **6 of 32 layers** carry information.
- **Labels are 2D**: a human traced letters once, in-plane. There is no per-layer label.
  → 3D volume in, **2D binary map out**.
- Strokes ≈ **30–50 px**; letters ≈ **300–1300 px** → tile size is the dominant variable.
- **The input is anisotropic.** Two in-plane axes are translation-invariant; the depth axis is
  physically privileged. *Any design that treats all three axes alike is making an assumption this
  data violates.*

**[ADD FIGURE]** `$WORK/segFormer/outputs/sf_base_t192_s42/figs/20241113090990_tau0.85.png`
(validation panel: probability map, prediction, ground truth, overlay — shown here only to
illustrate the task, not to discuss performance)

---

## Slide 9 — The domain shift, part 2: the architectural changes

**Title:** Six decisions, each tied to a measured fact

**Bullets:**

| Medical assumption | What we changed |
|---|---|
| 4-channel MRI in, 3-class mask out | backbone emits a **feature volume** `[B,16,32,t,t]`; shared depth-collapse + head produce the 2D logit |
| isotropic 128³ stem (depth 32→8→4→2→**1**) | **depth-preserving stem**: kernel `(3,7,7)`, stride `(1,4,4)` → depth stays 32, halved only inside the hierarchy (32→16→8→4) |
| isotropic key reduction (`r³`) | **per-axis** reduction, default `(1,4,4)` / `(1,2,2)` — spend the budget in-plane, protect depth |
| whole-volume training crops | sliding-window inference at stride t/2, Gaussian-blended |
| repo recipe (wd 0.01, 800 epochs, batch 2, Dice) | **benchmark protocol** — identical for all three models |
| per-modality z-scoring | raw `/255` (every normalisation variant was measured harmful) |

- Both stem and reduction choices are **ablated, not asserted** — runs `sf_stemiso_t192_s42`,
  `sf_sr444_t192_s42`, `sf_sr244_t192_s42`.

---

## Slide 10 — The shared layer: what makes this a benchmark

**Title:** Only the backbone changes

**Bullets:**
- Every model in the benchmark obeys one interface:
  `input [B,1,32,t,t] → backbone → [B,16,32,t,t] → DepthCollapse → [B,16,t,t] → Head → [B,1,t,t]`
- **DepthCollapse and Head are byte-identical shared code** across UNETR++, nnFormer and
  SegFormer3D — the backbone never produces the prediction.
- Verification that this is real, not claimed: the training log reports **4,439,985** trainable
  parameters; the backbone has **4,437,568**; the difference of exactly **2,417** is the shared
  collapse (17 + 63) and head (2,320 + 17).
- Integration: `models/segformer3d` is a **symlink** into this workspace; one new branch in the
  shared wrapper; everything else untouched.

**Speaker notes:** This slide pre-empts "how do you know the comparison is fair?" The parameter
arithmetic is a proof, not an assurance.

---

## Slide 11 — Porting the published code (what had to be fixed)

**Title:** The reference implementation cannot be used as published

**Bullets:**
- It recovers the 3D token grid by taking a **cube root of the token count** and reshaping to
  `(n,n,n)` — valid for 128³ medical volumes, invalid for `32 × t × t`.
- For our five tile sizes with the stock stem:

| tile | stage-1 grid | tokens | perfect cube? | outcome |
|---|---|---|---|---|
| 64 / 128 / 192 / 320 | (8,·,·) | 2,048 / 8,192 / 18,432 / 51,200 | no | **crash** |
| **256** | **(8,64,64)** | **32,768 = 32³** | **yes** | **runs — and is silently wrong** |

- At t256 — **our deployment tile, the only one in the ladder where this happens** — the reshape
  succeeds and reinterprets an (8,64,64) grid as (32,32,32). Every later operation mixes tokens
  across depth rows. **No exception is raised.**
- Also: the file **does not import** on our PyTorch version; the final upsample is isotropic (would
  return depth 128 under our stem); the key-reduction convolution silently truncates a
  non-dividing grid.
- Our port threads the true `(D,H,W)` grid through every block and asserts it at build and forward
  time. A unit test asserts the cube-root arithmetic above so the trap cannot be reintroduced.

**Speaker notes:** This is a thesis-reportable finding, not housekeeping: the same class of defect
was found in both previous ports. Published 3D medical repositories do not transfer safely.

---

## Slide 12 — Contribution C1: Peripheral Downsampled Keys (centrepiece)

**Title:** C1 — buy context through the key path, not the tile

**Bullets:**
- **Motivation:** context is the thesis's dominant lever, but every previous way of buying it meant
  **growing the tile**, which also grows queries, decoder cost, memory and (for UNETR++) parameters.
- **The opening:** the keys are already a low-resolution learned summary whose extent is decoupled
  from the queries (slide 4). Nothing requires them to summarise *the same region*.
- **Mechanism:** `out = attention(q,k,v) + γ ⊙ attention(q, k_peripheral, v_peripheral)`
  - dataset reads a **2× wider crop**, downsamples it 2×, passes it as **input channel 1**
    (same tensor rank, same voxel count → training loop, loss, collapse, head untouched);
  - a tiny **context stem** encodes it **directly at the reduced key grid** — the halo is never
    represented at full resolution;
  - **separate softmax**, gated by a zero-initialised per-head γ.
- **Critical detail:** concatenating the peripheral keys into the existing softmax would **not** be
  a no-op — a zero-weighted key still enters the denominator. A separate γ-gated branch is required,
  and a unit test asserts bitwise equality at initialisation.
- **Only possible on this model:** UNETR++'s compression has no slot for foreign tokens;
  nnFormer's relative-position bias is undefined across extents.
- **Mandatory same-cost control** — `sf_pdkctl_t128_s42`: identical shapes, parameters and compute,
  but the "halo" is a downsampled copy of the tile itself → **no new information**. Without it a
  gain could be read as extra capacity rather than extra context.
- **Runs:** `sf_pdk_t128_s42` · `sf_pdk_t192_s42` · `sf_pdkctl_t128_s42`

---

## Slide 13 — Contribution C2: Depth-Coordinate Mix-FFN

**Title:** C2 — test the positional claim where its premise fails

**Bullets:**
- Injects an explicit **depth coordinate** at exactly the point that constitutes the architecture's
  entire positional apparatus (the Mix-FFN depthwise convolution).
- `h ← h + γ ⊙ P[·]`, with two frames:
  - **absolute** — table indexed by the token's depth index;
  - **surface** — table sampled at `d − ŝ`, where `ŝ` is a soft-argmax surface estimate
    (float32 island, NaN-guarded).
- **Deliberate initialisation asymmetry:** γ is zero, but the table is **not**. Zero-initialising
  both would leave the branch dead at both ends with no gradient — the failure mode from chapter 1,
  where a gate initialised at 0.1 finished training at 0.1010.
- **Accuracy is not the primary readout.** Pre-registered instead:
  1. the **learned table plotted against the independent depth-attribution probe**;
  2. a **sign asymmetry** under induced depth offset — the absolute frame should degrade, the
     surface frame should not. That is a mechanistic result independent of any accuracy movement.
- **Runs:** `sf_dcm_abs_t192_s42` · `sf_dcm_surf_t192_s42`

---

## Slide 14 — Contributions C3 and C4, and what was rejected

**Title:** C3 and C4 — the efficiency claim and the decoder claim

**Bullets:**
- **C3 — Overlapping Sequence Reduction** (`sf_osr_t192_s42`): the architecture advertises an
  *overlapping* patch embedding, then uses a *non-overlapping* key reduction, exactly where most of
  its compute lives. We widen the reduction kernel (`2r`, stride `r`, padding `r/2`) so key blocks
  overlap. Output grid unchanged; extra taps initialised to exactly zero → exact reduction.
- **C4 — Decoder Spatial Mixing** (`sf_dsm_t192_s42`): one zero-gated depthwise convolution on the
  fused decoder map. Primary readout is a **substitution test** — does the post-processing filter's
  gain shrink? — not accuracy.
- **Considered and rejected** (these appear in the thesis as verdicts):
  - *surface-weighted key reduction* — the geometry family, third instance; measured neutral twice,
    supervising it hurt on both architectures. Inherited free instead via the shared collapse.
  - *anything targeting "confidence on faint ink"* — the decision threshold is swept every epoch,
    so any monotone rescaling of the logits is **invisible by construction**. This retires a whole
    family analytically, before spending any GPU time.
  - deep supervision · new losses · photometric augmentation · morphology · ensembling.
- **House rule, all four contributions:** every one is an **exact no-op at initialisation** behind a
  default-off flag — bitwise identical to stock at step 0, unit-tested. That is what makes each
  ablation a genuine single toggle.

---

## Slide 15 — The ablation study

**Title:** One key changed per run — the config diff *is* the ablation

**Bullets:**
- Reference cell: `sf_base_t192_s42`. Decision rule: a single-seed difference of **≤ 0.008 is
  noise**, and a neutral result is itself a finding.

| group | what it isolates | runs (exact output directory names) |
|---|---|---|
| Reference | the cell every other run is a one-key delta from | `sf_base_t192_s42` |
| Adaptation | stem · key-reduction preset ×2 · added norms · conv skip · final path · decoder width · depth-vs-width | `sf_stemiso_t192_s42` · `sf_sr444_t192_s42` · `sf_sr244_t192_s42` · `sf_nonorm_t192_s42` · `sf_noskip_t192_s42` · `sf_finlow_t192_s42` · `sf_dec128_t192_s42` · `sf_deep_t192_s42` |
| Shared layer | depth collapse | `sf_mean_t192_s42` |
| Recipe | the two hyper-parameters a new backbone earns | `sf_lr5e5_t192_s42` · `sf_wd1e2_t192_s42` |
| Contributions | C1 + its same-cost control · C2 ×2 frames · C3 · C4 | `sf_pdk_t128_s42` · `sf_pdk_t192_s42` · `sf_pdkctl_t128_s42` · `sf_dcm_abs_t192_s42` · `sf_dcm_surf_t192_s42` · `sf_osr_t192_s42` · `sf_dsm_t192_s42` |
| Tile ladder | context scale, **no capacity confound** | `sf_base_t64_s42` · `sf_base_t128_s42` · `sf_base_t192_s42` · `sf_base_t256_s42` · `sf_base_t320_s42` |
| Capacity × context | 4.4M / 20.0M / 79.4M across three tiles | `sf_m_t192_s42` · `sf_m_t256_s42` · `sf_m_t320_s42` · `sf_l_t256_s42` |
| Density | sampling stride | `sf_s_d64_t256_s42` · `sf_base_d96_t320_s42` |
| Prior-art control | 2.5D depth-as-channels (Kaggle 2023 formulation) | `sf_25d_t256_s42` |

- All runs live in `$WORK/segFormer/outputs/<run_name>/` — open any of them for curves and panels.

**[ADD FIGURE]** `$WORK/segFormer/outputs/sf_base_t192_s42/figs/train_curves.png`
(training curves of the reference cell — shown as an example of the per-run artifacts, not for
discussion of values)

---

## Slide 16 — Implementation status and what the design buys

**Title:** Implementation, and the two experiments only this model can run

**Bullets:**
- **Implemented and verified** (`$WORK/segFormer/`):
  - port with the explicit-grid rewrite; **50 unit checks**, including a bitwise exact-reduction
    proof for each of the four contributions;
  - a **faithfulness control**: rebuilding the reference configuration reproduces the released
    file's own parameter count (**4,491,809**) exactly;
  - capacity rungs built to land on the other models' sizes: **4,437,568** / **19,966,880** /
    **79,353,344** (≈ nnFormer-S and ≈ UNETR++ at t256);
  - shared-engine changes for C1 are **default-off with a byte-identity proof** against the
    pre-change code, so chapters 1 and 2 stay reproducible.
- **30 runs queued on A40**, chained behind a sanity job (unit tests, efficiency bench, memory
  probe, 2-epoch pipeline check).
- **Two experiments neither predecessor could run:**
  1. the **tile ladder without a capacity confound** — parameters here are completely
     tile-invariant, so the largest rung is interpretable for the first time;
  2. **resolution transfer** — score one checkpoint at 128/192/256/320/384 without retraining,
     testing the source paper's own resolution-robustness claim. Possible only because there are no
     tile-dependent parameters.
- **Still to run:** the 3-seed final band, and the shared analysis layer (receptive field, depth
  attribution, frequency, error anatomy, depth-offset stress, resolution transfer).

**Speaker notes:** Close on the framing: whatever the numbers turn out to be, the chapter's payload
is the efficiency frontier plus a predictive account of *which architectures can use more context*.
