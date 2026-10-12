# nnFormer for Ink Detection — A Complete Tutorial

**Purpose:** teach you the model, the port, the contributions, the ablations, the probes, and every term used in the results — from first principles, using only what was actually built and measured in this project.

**Everything here is traceable.** Code paths are relative to `$WORK/nnFormer/` where `$WORK=/home/woody/iwi5/iwi5401h`. Numbers come from `outputs/<run>/summary.json`, `postprocess.json`, and `probes/`.

**How to read this:** Parts 0–2 are background (skip Part 1 if you're comfortable with attention). Part 3 is the domain mapping. Part 4 is our contributions. Part 5 explains each ablation. Part 6 is results. Part 7 explains the probes. Part 8 is a glossary you can jump to at any time.

---

# PART 0 — The problem, the data, and the rules of measurement

You must understand the *setup* before the architecture, because almost every design decision in the port is a response to something in this section.

## 0.1 What the data actually is

A **scroll segment** is a sheet of carbonized papyrus that was virtually unrolled and flattened. After processing you get, per segment:

- `vol.npy` — a stack of **65 depth layers**, each a 2D image. Think of it as a very thin 3D slab: you are looking *through* the papyrus surface. Shape `[65, H, W]`.
- `ink.npy` / `ink_reg.npy` — a **2D** binary mask: where the ink is. Note: **2D, not 3D**. The annotation says "there is ink at this (x,y)", not "at this (x,y,z)".
- `mask.npy` — which pixels are valid papyrus (vs. air/holes).

Nine labelled segments: 8 for training, **1 for validation** (`20241113090990`), plus an unlabelled test segment (`flatboi`) used only for qualitative pictures.

**The depth window.** We use layers `[16, 48)` — 32 layers centred on layer 32. Why: the ink signal physically lives in a narrow band around the papyrus surface, measured at **z27–33**. Layers far above are air; far below are deeper sheets. (Part 7.2 shows the XAI evidence that all models really do concentrate there.)

**Why this is hard.** The ink is carbon-based, sitting on carbonized papyrus. In X-ray CT they have nearly the same density — the ink is almost invisible by brightness alone. What the models actually detect is a subtle *structural* difference in the substrate.

## 0.2 The input/output contract

Every model in this benchmark obeys the same contract:

```
input : [B, 1, D=32, t, t]     a 3D tile: 32 depth layers × t×t pixels
output: [B, 1, t, t]           a 2D ink probability map
```

So **3D in, 2D out**. This mismatch is the single most important thing about the port — medical segmentation networks are 3D-in-3D-out. Part 3 explains how we bridge it.

`t` is the **tile size** (64, 128, 192 or 256). We cut big segments into tiles because a whole segment (e.g. 10580×22000) doesn't fit in GPU memory.

## 0.3 The metric: F0.5, and why it is not accuracy

**Precision (P)** = of the pixels I called ink, what fraction really is ink?
**Recall (R)** = of the pixels that really are ink, what fraction did I find?

These trade off. The **F-beta** score combines them:

```
F_β = (1 + β²) · P · R / (β² · P + R)
```

- β = 1 → precision and recall weighted equally (the familiar F1).
- **β = 0.5 → our metric.** β² = 0.25, which makes **precision matter 4× more than recall**.

**Why β=0.5?** This is the Vesuvius Challenge convention. Reading ancient text is a task where a false letter is worse than a missing letter — you'd rather leave a gap than hallucinate a character that changes the meaning of a sentence.

**Consequence you'll see everywhere in the results:** our models operate at high precision (~0.85–0.89) and modest recall (~0.52–0.66). They are *deliberately* conservative. When we later evaluate at a lower threshold, recall jumps massively (measured: 0.53 → 0.79) while F0.5 drops — that's the metric doing what it was designed to do, not a broken model.

## 0.4 τ (tau) and "the sweep"

The network outputs a **probability** per pixel, in [0,1]. To get a binary mask you pick a **threshold τ**: pixel is ink if `prob ≥ τ`.

Different τ give different P/R and different F0.5. So:

- **τ sweep** = after every training epoch, evaluate F0.5 at τ ∈ {0.10, 0.15, …, 0.90} on the validation segment, and record the best.
- **Checkpoint selection** uses that swept F0.5 (`select_on: sweep`).

**Why this matters enormously:** in the earlier UNETR++ study, evaluating at a fixed τ instead of sweeping under-reported one model by **0.163 F0.5** — the same weights, a completely different reported number. Sweeping is non-negotiable, and *no run may ever be compared to another at a fixed τ*.

## 0.5 Post-processing: the component min-size filter

After thresholding you have connected blobs. **8-connected component filter**: find each connected blob (8-connectivity = pixels touching including diagonally), delete any blob smaller than `min_size`.

**Why it works:** real letters are large (40k–280k pixels). False-positive noise tends to be small isolated islands. Deleting small blobs raises precision cheaply.

Both τ and `min_size` are selected on the validation segment. Hence every result table has:
- **raw** = best swept F0.5, no filter
- **post** = best F0.5 after the (τ, min_size) filter

`post` is always ≥ `raw` in practice (measured gains ~+0.01 to +0.02).

## 0.6 Seeds, the noise floor, and why single runs lie

Train the same config twice with different random seeds and you get different numbers. In this project the **noise floor is ±0.008 F0.5** — measured, not assumed.

**The binding rule:**
- A single-seed difference of |Δ| ≤ 0.008 means **nothing**. It cannot be reported as a finding.
- A single-seed screen may only **kill** an idea or **promote** it to a 3-seed test.
- Anything claimed in the thesis gets **3 seeds (42/43/44)**, reported as mean ± standard deviation.

**Where this rule came from:** in the UNETR++ study, one component appeared to give +0.05. Re-run across 3 seeds, the effect vanished — it had been a seed artifact. That episode is why this discipline exists, and you will see it applied repeatedly below.

## 0.7 The rest of the fixed protocol

Held constant for every model in the benchmark, so only the architecture differs:

| Setting | Value |
|---|---|
| Optimizer | AdamW, LR 1e-4, weight decay 3e-5 |
| Schedule | 3 warm-up epochs, then cosine decay to ~0 over 60 epochs |
| Loss | `dice_bce_masked` (see 0.8) |
| Augmentation | hflip, vflip, rot90 (p=0.5); z_shift ±2 (p=0.3); **z_flip OFF** |
| Precision | bf16 mixed precision |
| Batch | 4 at t192/t256, `samples_per_epoch` = 4000 → 1000 optimizer steps/epoch |
| Sampling | 1:1 positive/negative tile oversampling (a tile is "positive" if ≥5% ink) |
| Inference | sliding window, stride = t/2, Gaussian-blended overlaps |

**z_flip OFF** means we never mirror the depth axis — flipping "up" and "down" through a papyrus surface creates a physically impossible sample. (Measured: turning it off gained +0.009.)

**z_shift ±2** randomly shifts the depth window by up to 2 layers — teaches mild robustness to depth misalignment.

## 0.8 The loss function

```
dice_bce_masked = 0.5 · BCE + 0.5 · Dice,  computed only on "trusted" pixels
```

- **BCE (binary cross-entropy)**: standard per-pixel classification loss.
- **Dice loss**: `1 − 2|P∩G| / (|P|+|G|)` — an overlap measure. Robust to class imbalance, which matters because only ~13% of pixels are ink.
- **"masked" / the u1 trimap**: a per-pixel map `unc.npy` marks each pixel as *confident ink*, *confident background*, or **uncertain**. Uncertain pixels are **excluded from the loss entirely**.

**What is uncertain?** A ±4px band around every label boundary. Hand annotations have fuzzy edges, and the labels are known to be slightly misregistered. Rather than train the model on pixels whose labels we don't trust, we simply don't train on them. (Measured gain: +0.012.)

---

# PART 1 — Attention from first principles

Skip if you know transformers. Everything here is needed to understand Part 2.

## 1.1 What attention computes

Take a set of **tokens** (here: small 3D patches of the volume), each a vector. Attention lets every token gather information from other tokens, weighted by relevance.

Each token produces three vectors via learned linear layers:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer?"
- **Value (V)** — "what I actually pass on."

For token *i*, compare its query to every token's key:

```
score(i,j) = Q_i · K_j / sqrt(d)          d = per-head dimension
attn(i,·)  = softmax over j of score(i,·)   → weights summing to 1
output_i   = Σ_j attn(i,j) · V_j
```

The `sqrt(d)` scaling keeps scores in a sane range so softmax doesn't saturate. In our code this is `self.scale = (dim // num_heads) ** -0.5` (`attention.py`).

**Softmax** turns arbitrary real scores into a probability distribution: `softmax(x)_j = exp(x_j) / Σ_k exp(x_k)`. So each token's output is a weighted average of all values, with weights it chose itself.

## 1.2 Multi-head attention

Instead of one attention, split the channels into **h heads**, each doing its own attention with dimension `d = C/h`, then concatenate. Different heads can specialise (one on depth relationships, another on stroke edges, etc.).

In our port `head_dim = 32` throughout, so e.g. `dim=512 → 16 heads`.

## 1.3 The quadratic cost problem, and windows

Attention compares **every token to every token**: with N tokens the cost is O(N²).

Our stage-1 grid at t256 is (32, 64, 64) = **131,072 tokens**. N² ≈ 1.7×10¹⁰ per head. Impossible.

**Solution — windowed attention (from Swin Transformer):** partition the volume into small non-overlapping **windows** and run attention *only inside each window*. With window (4,8,8) = 256 tokens, cost per window is 256² and there are 512 windows — linear in volume size instead of quadratic.

This is what "**LV-MSA**" means: Local Volume Multi-head Self-Attention.

**The cost of this trick:** a token can only see other tokens in its own window. Information cannot cross window boundaries in a single layer. That limitation turns out to be *the* explanation for this model's benchmark result (Part 6.5).

## 1.4 Shifted windows

If windows were always in the same place, the boundaries would be permanent walls. **Shifted-window attention** alternates: block 1 uses the normal grid; block 2 shifts the whole partition by half a window (here (2,4,4)) so previously-separated tokens now share a window.

Implementation detail: shifting is done with `torch.roll`, which wraps around the volume edges. Tokens that wrap are not genuine neighbours, so an **attention mask** (`shifted_window_mask` in `windows.py`) adds −100 to those scores, making their softmax weight ≈0.

In our code, `SwinBlock3D(..., shifted=(i % 2 == 1))` — every second block shifts.

## 1.5 Relative position bias

Plain attention is **permutation-invariant** — it has no idea *where* tokens are. We must inject geometry.

**Relative position bias**: for every pair of positions inside a window, add a learned scalar (per head) to the attention score, indexed by their *relative* offset (Δz, Δy, Δx):

```
attn(i,j) = Q_i·K_j/√d + bias[relative_offset(i,j)]
```

For a window (wd, wh, ww) there are `(2wd−1)(2wh−1)(2ww−1)` possible offsets, so the learned table has that many rows × num_heads columns.

**The bug we found here** (see Part 3.5): the official repo's index arithmetic is only correct for cubic windows, and silently collides distinct offsets when windows are anisotropic — which ours are.

## 1.6 Cross-attention (Q from one place, K/V from another)

If Q comes from tensor A and K/V from tensor B, then A is *querying* B — "A, go look things up in B". This is **cross-attention**, and it's how nnFormer's skip connections work (Part 2.5).

---

# PART 2 — nnFormer's blocks, one at a time

nnFormer (Zhou et al., IEEE TIP 2023) is a U-shaped encoder–decoder. Here is every component, in the order data flows through it. Code: `src/nnformer_ink/`.

## 2.0 The overall shape

```
input [B,1,32,t,t]
   │
   ├─ Convolutional embedding                        → stage-1 tokens
   │
   ├─ ENCODER stage 1: 2× LV-MSA blocks   ──skip──┐  grid (32, t/4, t/4)
   │      ↓ patch merging (halve each axis)        │
   ├─ ENCODER stage 2: 2× LV-MSA blocks   ──skip──┤  grid (16, t/8, t/8)
   │      ↓ patch merging                          │
   ├─ ENCODER stage 3: 2× LV-MSA blocks   ──skip──┤  grid (8, t/16, t/16)
   │      ↓ patch merging                          │
   ├─ ENCODER stage 4: 3× GV-MSA blocks  = BOTTLENECK  grid (4, t/32, t/32)
   │      ↓                                        │
   ├─ DECODER stage 3: patch expand + skip attention ◄┘
   ├─ DECODER stage 2: patch expand + skip attention ◄
   ├─ DECODER stage 1: patch expand + skip attention ◄
   │
   └─ Final expand → [B,16,32,t,t] feature volume → (shared DepthCollapse + Head) → [B,1,t,t]
```

Concrete grids at **t256**: S1 (32,64,64) → S2 (16,32,32) → S3 (8,16,16) → S4 (4,8,8).

**Why a U-shape?** The encoder progressively shrinks resolution while growing channels — it trades "where" for "what". The decoder rebuilds resolution. **Skip connections** carry high-resolution detail from encoder to decoder, because the bottleneck alone is too coarse to place a stroke edge precisely.

## 2.1 Convolutional embedding (`PatchEmbed3D`, `sampling.py`)

Most transformers "patchify" with a single big-stride linear projection. nnFormer instead uses **four 3×3×3 convolutions** (arranged as two `_Project` blocks, each conv → GELU → LayerNorm → conv):

```
[B,1,32,256,256] → [B,C/2,...] → [B,C,32,64,64]
```

**Why convolutions?** Small convolutions capture fine local texture *before* anything gets tokenised. For ink — where the signal is subtle local structure — that's exactly the right inductive bias.

**GELU** = Gaussian Error Linear Unit, a smooth activation ≈ `x·Φ(x)`; standard in transformers.
**LayerNorm** = normalises each token's feature vector to zero mean / unit variance, then applies a learned scale and shift. Stabilises training.

Our version generalises the strides per axis so the depth axis can be left untouched (Part 3.2).

## 2.2 LV-MSA block (`SwinBlock3D`, `blocks.py`)

The workhorse. One block does:

```
x = x + DropPath( WindowAttention( LayerNorm(x) ) )       ← attention sub-layer
x = x + DropPath( MLP( LayerNorm(x) ) )                   ← feed-forward sub-layer
```

- **Residual connections** (`x + …`): the block learns a *correction* to its input, letting gradients flow through deep stacks.
- **Pre-norm** (LayerNorm *before* each sub-layer): the modern, more stable arrangement.
- **MLP**: two linear layers with GELU between, expanding to `4×dim` then back. This is where per-token nonlinear processing happens — attention mixes tokens, the MLP transforms them.
- **DropPath (stochastic depth)**: during training, randomly drop the *entire* residual branch for some samples, so the block is skipped. A regulariser for deep networks. Our default rate is 0.1 (0.2 for the largest model).

Inside `WindowAttention`, the flow is: partition into windows → attention with relative position bias → merge windows back.

## 2.3 Patch merging (`PatchMerging3D`) — the downsampler

```
GELU → LayerNorm → Conv3d(dim → 2·dim, kernel 3, stride 2)
```

Halves each of D, H, W (so token count ÷8) and doubles the channels. This is the standard "trade resolution for semantics" step. Note nnFormer uses a **strided convolution**, not the concatenate-and-project trick of the original Swin.

## 2.4 GV-MSA — the bottleneck (Global Volume MSA)

At stage 4 the grid is tiny: (4,8,8) = 256 tokens at t256, (4,6,6) = 144 at t192. Small enough that we can afford **full attention over the entire grid** — every token sees every other token.

In our implementation this is expressed cleanly: the bottleneck "window" is set to the *whole grid*, so the same `SwinBlock3D` code performs global attention. That's what `gv_bottleneck: true` means.

**Purpose:** give the network one place where genuinely global context is available — e.g. to suppress an isolated false blob because nothing else in the tile looks like it.

## 2.5 Skip attention (`SwinBlockKV3D`) — nnFormer's signature

A normal U-Net concatenates the encoder skip onto the decoder feature and convolves. nnFormer instead makes the decoder **query** the encoder:

```
x_up  = upsample(coarse decoder features)     ← "what I think is here, semantically"
skip  = encoder features at this resolution   ← "the high-resolution evidence"

Q = x_up          (note: NO linear projection — used directly, faithful to the original)
K, V = Linear(skip)
x = (x_up + skip) + DropPath( Attention(Q, K, V) )
x = x + DropPath( MLP(LayerNorm(x)) )
```

**Intuition:** rather than blindly merging the encoder's features, the decoder *asks* the encoder for what it needs. "I believe there's a stroke edge around here — show me your best local evidence for it."

This is the component nnFormer's own paper reports as its largest gain in the medical setting. Our ablation tests whether that holds for ink (Part 5.4 — it doesn't).

## 2.6 Patch expanding (`PatchExpanding3D`) — the upsampler

```
LayerNorm → ConvTranspose3d(dim → dim/2, kernel 2, stride 2)
```

The inverse of patch merging: doubles each spatial axis, halves channels. A **transposed convolution** is a learned upsampling operation.

## 2.7 Final expand + the full-resolution skip (our addition)

`FinalExpand3D` uses a transposed convolution with the patch size as kernel/stride to jump from the stage-1 grid back to full resolution, producing a **16-channel feature volume** `[B,16,32,t,t]`.

We then add a **full-resolution convolutional stem** on the raw input, and fuse:

```
stem = Conv3d(1→8) → GELU → Conv3d(8→16) → GELU        (on the original input)
out  = Conv3d(32→16) → GELU → Conv3d(16→16)            (on [expanded ; stem])
```

**Why:** the original nnFormer uses deep supervision to keep fine detail; we removed deep supervision (Part 3.4), so this conv skip carries the finest detail instead. This mirrors exactly what the UNETR++ port did, keeping the two models comparable.

---

# PART 3 — Mapping the model to the ink problem

This is the "domain shift" — what had to change, and why. Every item was a decision with a reason.

## 3.1 3D output → 2D output

**The mismatch:** nnFormer predicts a 3D mask (a label per voxel). Our labels are 2D (a label per surface pixel).

**The solution — collapse the depth axis:**

```
backbone → [B,16,D,t,t]  (feature volume)
DepthCollapse           → [B,16,t,t]   (depth axis removed)
Head                    → [B,1,t,t]    (logits)
```

`DepthCollapse` is a **shared** module — the exact same code UNETR++ uses. Its mode here is **surface_relative**:

1. Estimate a per-tile surface depth `ŝ` by a **soft-argmax**: predict a score per depth layer, softmax it, take the weighted average of layer indices. (Soft-argmax is a differentiable stand-in for "which layer is the peak?".)
2. Learn a weighting curve `φ_rel` over *relative* depth `r = d − ŝ`, initialised as a Gaussian centred at r=0.
3. Collapse by that weighting.

**Meaning:** "combine the depth layers, weighting most heavily the ones near the estimated surface."

`Head` = `Conv2d(3×3) → GELU → Conv2d(1×1)` → one logit per pixel. A **logit** is a pre-sigmoid score in (−∞,∞); `sigmoid(logit)` gives the probability.

**Why this design matters for the benchmark:** because the collapse and head are *shared and identical* across models, any difference between UNETR++ and nnFormer is attributable to the backbone alone.

## 3.2 Depth-preserving embedding

**Medical:** the embedding halves depth (a CT scan has hundreds of slices; losing half is fine).

**Ink:** we have 32 layers, and the entire signal is in a band of ~6. Halving depth immediately would throw away the thing we're trying to detect.

**Our change:** patch size **(1,4,4)** — stride 1 on depth, 4 in-plane. Depth is preserved at full 32 into stage 1, and only reduced *inside* the hierarchy (32→16→8→4).

This required generalising the embedding's stride factorisation (`_two_factor` in `sampling.py`) so each axis can independently use strides (1,1), (2,1) or (2,2).

## 3.3 Anisotropic windows

**Medical:** cubic windows, e.g. (4,4,4), because anatomy is roughly isotropic.

**Ink:** depth (32) and in-plane (t/4 = 64 at t256) are wildly different scales, and they *mean* different things — a step in depth is "deeper into the papyrus", a step in-plane is "along the writing".

**Our change:** per-stage tuple windows **(4,8,8), (4,8,8), (4,4,4)**, chosen so they divide every stage grid exactly at t ∈ {64,128,192,256}. That gives **no padding paths** — the code asserts divisibility at build time and refuses to run otherwise.

**Why no padding?** Padding silently hides geometry errors. Failing loudly at construction is safer. (This caught a real bug — see Part 5.5.)

## 3.4 Deep supervision OFF — a fairness decision

**Deep supervision (DS)** = attaching extra loss terms at intermediate decoder resolutions, using downsampled ground truth, to give gradients a shorter path.

nnFormer publishes with DS on. UNETR++'s port ran with DS **off** (it's structurally awkward with 2D labels: strokes ~30–50px wide become 4–6px at 1/8 resolution — they largely dissolve).

**The decision:** DS **off for every benchmark model**, stated explicitly in the methods. If nnFormer had DS and UNETR++ didn't, any difference between them would partly reflect a *training-objective* difference rather than architecture — a confound that would invalidate the headline comparison.

## 3.5 The port bug we found

The official implementation builds its relative-position index with strides `*= 3*ws−1` and `*= 2*ws−1`. That arithmetic is only valid for cubic windows. With anisotropic windows it maps **different (Δz,Δy,Δx) offsets onto the same table entry** — the model loses the ability to distinguish "one step deeper" from "one step sideways".

Our port uses the standard Swin-3D strides `(2wh−1)(2ww−1)` and `(2ww−1)`, with assertions that the index range exactly fills the table, plus a unit test.

**Lesson:** the equivalent class of bug (hardcoded medical token counts) was found in the UNETR++ port too. Published 3D medical repos do not transfer safely to new domains — verifying them *is* part of the work.

## 3.6 Everything else held fixed

Training recipe, augmentation, loss, sliding-window inference, τ sweep, post-filter, seeds — all inherited unchanged (Part 0.7), so the benchmark compares architectures and nothing else.

---

# PART 4 — What we introduced (the contributions)

Three additions. Two are our own proposals; one is prior art we implemented for comparison.

## 4.0 The design rule that makes them measurable

**Every new component is an exact no-op at initialisation, behind a default-off flag.**

Meaning: with the flag on but before any training, the model produces **bitwise identical** output to the stock model. This is verified in `tests/test_smoke.py` (max |Δ| = 0.0), together with a second test proving the branch becomes *live* once its parameters are non-zero.

**Why this matters:** if adding a component also changed the initialisation or the computation graph, any measured difference could come from that instead of from the idea. Exact reduction means a measured delta is attributable to the component. (This convention is inherited from the UNETR++ study, where every component started as an identity.)

## 4.1 C1 — SR-PB (Surface-Relative Position Bias)

**The problem it addresses.** The relative position bias learns "what does a depth offset of Δz mean?" in *absolute* layer coordinates. But the papyrus surface is not at a fixed layer — it drifts between segments and, under deployment on a new scroll, could sit anywhere in the window. A bias tuned around layer 32 is in the wrong frame if the surface is at 25.

**A correction that had to be made first.** The obvious idea — "re-anchor each token's depth coordinate on the estimated surface ŝ before the bias lookup" — is provably a **no-op**. The bias is indexed by *differences*: Δz = z_i − z_j. Subtracting a common ŝ from both gives (z_i − ŝ) − (z_j − ŝ) = z_i − z_j. It cancels exactly. Implementing that would have produced a null result caused by an algebraic identity, not by the data.

**What SR-PB actually is** (`SurfaceDepthBias` in `contrib.py`): an **additive, key-side, per-token** bias:

```
1. depth descriptor per layer = concat(mean over plane, std over plane)
2. small linear layer → one score per depth layer → softmax → soft-argmax → ŝ
3. relative depth r_d = d − ŝ  for each layer d
4. learned table b(r), sampled at r_d by linear interpolation → per-head bias
5. attention score gets  + b(r_j)   for key j   (broadcast over all queries)
```

**Why "std" in the descriptor:** the ink cue is *textural energy*, not mean brightness — a layer can be equally bright but more structured. (Same reasoning as UNETR++'s depth branch.)

**Why key-side and additive:** it does not cancel. It says "keys lying in the ink band are intrinsically more worth attending to", in a frame that follows the estimated surface.

**Exact reduction:** the estimator is zero-initialised → softmax is uniform → ŝ = window centre; the table is zero-initialised → b ≡ 0 → attention is bitwise stock.

**Where it's applied:** in the shared attention module, so it acts in LV-MSA, GV-MSA *and* skip attention — one implementation, three mechanisms.

## 4.2 C2 — CSSA (Cross-Scale Skip Attention)

**The problem it addresses.** The single biggest lever in the UNETR++ study was *spatial context*: growing the tile from 64 to 256 was worth +0.16 F0.5, because a bigger window lets the model see whole letterforms rather than isolated stroke fragments. But buying context by enlarging the tile costs memory quadratically in tile area.

**The idea.** nnFormer's skip attention has the decoder query the encoder **at its own resolution** — fine detail only. It never queries the *coarse* stages, where a single token already covers a letterform-sized region. So: let it query those too.

**Implementation** (`CrossScaleKV` in `contrib.py`):

```
out = softmax(Q·K_localᵀ)·V_local  +  γ · softmax(Q·K_coarseᵀ)·V_coarse
```

The coarse branch has its **own softmax** and is scaled by a **per-head gate γ, zero-initialised**. At step 0, γ=0 → exactly stock skip attention. Coarse stages hold few tokens (144 at t192, 256 at t256), so the added cost is small — which is the whole point: letterform context at a fraction of the cost of enlarging the tile.

**The trained γ is itself a result.** A gate that stays at zero is how "this had nothing to add" reveals itself.

## 4.3 Prior art — vFFC (Quattrini et al. 2023, arXiv:2308.05070)

A published ink-detection method whose contribution is a **volumetric Fast Fourier Convolution** block. We implemented it as an alternative bottleneck to compare *mechanism classes*.

**What a Fourier convolution does.** A convolution has a small kernel and therefore a small receptive field. The **Fourier transform** decomposes a signal into frequencies; crucially, *every* output frequency depends on *all* input positions. So doing a cheap 1×1×1 convolution **in the frequency domain** gives you a global operation for near-local cost.

**The block** (`SpectralTransform3D`, `VFFC3D`, `VFFCResBlock`):

```
channels split in two:
  LOCAL branch  : two 3×3×3 convolutions
  GLOBAL branch : 3×3×3 conv, then ST-3D:
        rFFT3 → stack(real, imag) on channels → 1×1×1 conv + BatchNorm + ReLU
        → inverse rFFT3
branches cross-summed, normalised, concatenated
```

**Engineering note:** `torch.fft` has no bf16 kernel, so the spectral transform runs as an **fp32 island** — cast in, transform, cast back, with `nan_to_num` on exit.

**Its role here:** a three-way bottleneck comparison — {local windows, global attention, spectral convolution} — asking whether the *class* of global-context mechanism matters.

---

# PART 5 — The ablation studies: what each one asks

An **ablation** removes or swaps exactly one thing and measures the change. The logic: if removing X doesn't hurt, X wasn't doing anything.

**Reading the deltas:** Δ is (variant − reference) in raw F0.5. |Δ| ≤ 0.008 = noise = "neutral".

⚠ **Critical caveat:** the runs in 5.1–5.7 used label file `ink_reg2.npy`; everything from 5.8 onward used `ink_reg.npy`. These are two different label versions = two different yardsticks. **Never compare a number across the two groups.**

## 5.1 `n_base_t192_s42` — the reference

Stock nnFormer-S at t192. Everything else in its group is measured against it. Result: raw **0.7780**, post 0.7880.

## 5.2 `n_mean_t192_s42` — is the surface-relative collapse worth it?

**Swap:** `DepthCollapse` from *surface_relative* → plain **mean** over depth.

**Where it comes from:** in the UNETR++ study, the surface-relative collapse was the *only* architectural component that survived ablation (−0.009 when removed at t192). Does that transfer to nnFormer?

**Result:** raw 0.7739 (Δ −0.0041) → **neutral**. The geometry prior does *not* reproduce as a clear win in this architecture.

## 5.3 `n_abs_t192_s42` — prior vs. frame

**Swap:** collapse → *learned absolute* depth weighting (a learned curve over absolute layer index, with no surface estimate).

**What it separates:** "is the benefit from *learning* a depth weighting at all, or specifically from expressing it *relative to the surface*?"

**Result:** raw 0.7682 (Δ −0.0098) → the worst of the three collapses, slightly outside noise.

## 5.4 `n_skipconv_t192_s42` — does nnFormer's signature component matter?

**Swap:** skip **attention** → plain concatenate + linear fuse (a conventional U-Net skip).

**Where it comes from:** this is the component nnFormer's paper credits with its largest medical gain. It is also the direct analogue of UNETR++'s "signature component" ablation, which found that component neutral.

**Result:** raw 0.7778 (Δ **−0.0002**) → **completely neutral.** nnFormer's headline mechanism does nothing measurable for ink. This is one of the most informative results in the study.

## 5.5 `n_lvonly_t192_s42` — does global bottleneck context matter?

**Swap:** GV-MSA (global) → local windows at the bottleneck.

**Pre-registered expectation:** removing global context should *not* help.

**Result:** raw 0.7833 (Δ **+0.0053**) → neutral; removing it if anything nudged upward.

**A real bug this caught:** the LV window (4,4,4) does not divide the t192 bottleneck grid (4,6,6). The build **failed loudly** rather than silently padding. Fixed by deriving the largest grid-dividing local window (→ (4,3,3) at t192). This is the no-padding design working as intended.

## 5.6 `n_embedz2_t192_s42` — how much does depth preservation matter?

**Swap:** patch size (1,4,4) → **(2,4,4)**, i.e. halve depth at the embedding like the medical original.

**Where it comes from:** the direct analogue of UNETR++'s depth-preserving-embedding ablation.

**Result:** raw 0.7665 (Δ −0.0115) → depth-halving *hurts*, mildly outside noise. Confirms the Part 3.2 reasoning.

**A cautionary history:** in an earlier round on the *other* label set this same toggle appeared to be a clear **winner** (+0.012). Re-run on cleaner labels it reversed sign. It was a label-noise artifact — exactly the kind of thing the 3-seed / re-run discipline exists to catch.

## 5.7 `n_lr5e5_t192_s42` — is the new backbone happy at this learning rate?

**Swap:** LR 1e-4 → 5e-5. The one recipe parameter a new architecture is allowed to question.

**Result:** raw 0.7785 (Δ +0.0005) → neutral. The inherited recipe transfers.

## 5.8 The capacity × density study (t256, `ink_reg.npy`)

**The question:** the published nnFormer is ~150M parameters trained on far more data than our 9 segments. Does that capacity help, hurt, or do nothing here?

Three capacity rungs (S 19.9M / M 79.1M / F 177.8M) × two data densities:
- **stride 128** → 21,360 training tiles
- **stride 64** ("d64") → **84,523** tiles (4× more, heavily overlapping)

Note `samples_per_epoch` stays 4000 in both, so the *optimizer budget is identical* — density buys **pool diversity**, not more training.

| Run | Result |
|---|---|
| `nnf_s_t256_s42` | 0.7677 raw / 0.7868 post, best epoch 59 |
| `nnf_m_t256_s42` | 0.7613 / 0.7753, best epoch 44 |
| `nnf_f_t256_s42` | 0.7671 / 0.7806, best epoch **37**, **overfit signature** |
| `nnf_f_d64_t256_s42` | 0.7652 / 0.7807, best epoch 53 (overfit reduced) |
| `nnf_s_d64_t256_s42` | 0.7747 / 0.7884, best epoch 50 |

**The overfit signature** (a specific, checkable pattern, not a vibe): validation peaks early **and** training loss keeps falling substantially afterwards. For nnF-F: best epoch 37, then train loss dropped a further 15% while validation *declined*. That is memorisation.

**Verdicts:**
- Capacity spread = 0.006 → **inside the noise floor**. The 178M model does not beat the 20M model. More parameters buy nothing on 9 segments.
- Dense sampling both improved the small model and **mitigated** the big model's overfit (best epoch 37→53) — evidence the mitigation works as designed.

## 5.9 `nnf_s_d64_t256_e90_s42` — is 60 epochs enough?

**Motivation:** `nnf_s_t256_s42` peaked at epoch **59 of 60** — validation was still rising when the budget ran out. That's a legitimate worry.

**Test:** same config, 90 epochs (patience scaled 20/60 → 30/90).

**Result:** raw 0.7809 vs 0.7747 same-seed = **+0.0062, inside noise**, best epoch 54 of 90. So the model converges before 90; 60 is adequate. Matches the UNETR++ finding (there, 90 epochs was actively *worse* than 60).

## 5.10 The contribution screens (t192, `ink_reg.npy`)

Reference: the **stock 3-seed band** — `n_base_t192_r1_s{42,43,44}` = 0.7581 / 0.7687 / 0.7703 raw → **0.7657 ± 0.0066** raw, 0.7836 ± 0.0086 post.

| Run | Adds | raw | Δraw vs stock band mean |
|---|---|---|---|
| `n_cssa_t192_r1_s42` | CSSA | 0.7618 | −0.0039 |
| `n_srpb_t192_r1_s42` | SR-PB | 0.7708 | |
| `n_srpb_t192_r1_s43` | SR-PB, seed 43 | 0.7724 | |
| `n_srpb_t192_r1_s44` | SR-PB, seed 44 | 0.7733 | |
| SR-PB **band** | 3 seeds | **0.7722 ± 0.0013** | **+0.0065** (inside ±0.008) |
| `n_vffc_t192_r1_s42` | vFFC bottleneck | **0.7844** | **+0.0187** (n=1) |

**A teaching moment — read this carefully.** For several days the SR-PB result was reported as **+0.013 to +0.015**, because all three SR-PB seeds were being compared against stock **seed 42 alone** (0.7581). When stock seeds 43 and 44 finished (0.7687, 0.7703), seed 42 turned out to be the *weakest* stock seed by a margin of ~0.011 — larger than the noise floor. Against the proper band, SR-PB's accuracy delta is **+0.0065: noise**. This is exactly the failure the ±0.008 / 3-seed discipline of Part 0.6 exists to catch, and it is the same artifact the UNETR++ study documented for DA-EPA. **Rule you should internalise: a band is only meaningful against another band.**

## 5.11 The CSSA 2×2 — a pre-registered payoff test

CSSA's claim is specific and falsifiable: *if it supplies letterform context architecturally, then CSSA at t128 should approach stock at t192.*

Four cells, all at batch 4 so steps-per-epoch is identical (removing an optimizer-budget confound):

| | t128 | t192 |
|---|---|---|
| stock | `n_base_t128_r1_s42` 0.7518 | `n_base_t192_r1_s42` 0.7581 |
| CSSA | `n_cssa_t128_r1_s42` 0.7539 | `n_cssa_t192_r1_s42` 0.7618 |

**Pre-registered success criterion:** CSSA@t128 closes ≥ half the stock t128→t192 gap.
**Measured:** the stock gap is only **+0.0063**; CSSA closed +0.0020 = **32%**. → **FAIL**.

**But note the design flaw exposed:** the test assumed a large gap to close (UNETR++'s was +0.055). nnFormer's is 0.0063 — so the experiment had almost no dynamic range. The honest conclusion is *"underpowered because nnFormer barely benefits from tile scale at all"* — which is itself the more interesting finding (Part 6.5).

## 5.12 `nnf_c1_t256_s{42,43,44}` — the final combination

Best capacity (S) + best density (d64) + the one contribution that passed its gate (SR-PB), at deployment scale, 3 seeds. This is the reported nnFormer configuration.

---

# PART 6 — Results, explained

## 6.1 The benchmark row

| Configuration | Seeds | raw F0.5 | post F0.5 | best seed |
|---|---|---|---|---|
| stock `nnf_s_d64_t256` | 3 | 0.7825 ± 0.0117 | 0.7981 ± 0.0123 | 0.8120 |
| **`nnf_c1_t256` (+SR-PB)** | 3 | 0.7801 ± **0.0037** | **0.8005** ± 0.0102 | **0.8122** |
| *UNETR++ `c1_t256`* | *3* | *0.8111 ± 0.0039* | *0.8190 ± 0.0062* | *0.8261* |

With TTA: `nnf_c1_t256_s42` post = **0.8133**.

**Reading it honestly:** nnFormer's 3-seed post mean is 0.8005 vs UNETR++'s 0.8190 → **−0.0185**. Best-seed-to-best-seed the gap is smaller. Report both; don't cherry-pick.

**A subtle but real finding:** SR-PB leaves the *mean* essentially unchanged at t256 but **cuts raw seed variance ~3×** (±0.0037 vs ±0.0117). Stability is a legitimate property — a model you can train once and trust is worth something.

## 6.2 SR-PB — neutral on accuracy, real on stability, and a lesson in seed discipline

Three pieces of evidence, read carefully:

**(a) Accuracy — neutral.** SR-PB 3-seed band 0.7722 ± 0.0013 vs stock 3-seed band 0.7657 ± 0.0066 → **+0.0065 raw, inside the noise floor**; post −0.0006. (An earlier "+0.013" was an artifact of comparing against the weakest stock seed alone — see 5.10.) **What is real is the variance**: SR-PB's seed spread is 5× tighter at t192 and ~3× tighter at t256 (±0.0037 vs ±0.0117). A model you can train once and trust is a legitimate property to report.

**(b) Robustness — the depth-offset stress test.** Evaluate the *same trained model* with the inference depth window deliberately shifted by δ layers. This simulates deploying on a scroll whose surface sits at a different depth.

| δ (layers) | −8 | −4 | 0 | +4 | +8 | mean |
|---|---|---|---|---|---|---|
| stock | 0.332 | 0.719 | 0.758 | 0.677 | 0.299 | 0.599 |
| SR-PB (s42) | 0.428 | 0.735 | 0.771 | 0.700 | 0.376 | 0.633 |
| SR-PB (s43) | 0.401 | 0.742 | 0.772 | 0.710 | 0.401 | 0.641 |

SR-PB is above the baseline **at every offset**, mean +0.035, with the largest gains in the tails.

**The caveat that applies here too:** the baseline row is `n_base_t192_r1_s42` — the weak seed — and it is **n=1**. The tail gaps (+0.096, +0.077) are much larger than the 0.0066 seed spread at δ=0, so the robustness effect *probably* survives replication, but it is not yet a band-vs-band claim. Two inference-only stress runs on base seeds 43/44 would settle it.

**Why it would matter if it holds:** in the UNETR++ study, the equivalent geometry intervention could only *redistribute* — supervising the surface estimate bought +0.13 in the tails but **cost −0.050 at the centre**, for a mean of 0.596 vs 0.596: identical to three decimals. A robustness gain with *no* centre cost would be a genuine improvement over the prior approach. We also tested the supervised variant on nnFormer directly (`n_srpb_aux_t192_r1_s42`): **0.7607 raw — below both stock and plain SR-PB**. Supervising the surface estimate hurts here too, replicating the UNETR++ finding.

**(c) Mechanism.** See Part 7.2 — the depth-attribution probe shows *why*.

## 6.3 CSSA — mechanism confirmed, metric neutral

- F0.5: +0.0037 (t192), +0.0020 (t128) → inside noise.
- **But the ERF grew measurably**: r50 63→75 px (+19%), r90 97→107.
- Robustness improved (stress mean +0.028), recall rose (0.529→0.601) at precision cost.

**Interpretation:** the intervention *did what it was designed to do* — more context genuinely reached fine-resolution decoding — but it didn't convert into the precision-weighted metric. That is a much more informative outcome than a plain null, and the ERF measurement is the evidence.

## 6.4 vFFC — the unplanned result, and its caveat

`n_vffc_t192_r1_s42`: 0.7844 raw / 0.7992 post = **+0.0263 raw**, the best t192 number in the study, far outside noise.

**Why it cannot yet be claimed:** it carries **65.0M parameters vs 20M** stock. The capacity study (5.8) partially defuses this — capacity was flat there — but that experiment varied *attention width*, not operator class. **A parameter-matched control is required** before this is a mechanism finding rather than a capacity effect.

**Second caveat:** the block was implemented from a written summary of the paper, not the authors' code. Details the summary didn't specify (channel split ratio, exact BatchNorm placement) are our reconstruction and should be checked against the paper and `github.com/aimagelab/vffc`.

## 6.5 Why nnFormer trails UNETR++ — the mechanistic answer

This is the most valuable interpretive result, and it comes from the ERF probe:

| Model | tile | r90 | **r90 / tile** |
|---|---|---|---|
| `n_base_t192_r1_s42` | 192 | 97 | **51%** |
| `nnf_s_d64_t256_s42` | 256 | 115 | **45%** |
| `nnf_c1_t256_s42` | 256 | 124 | 48% |
| *UNETR++ (all rungs)* | *64→256* | *35/68/105/134* | *~53% constant* |

**UNETR++** consumed a constant ~53% of whatever tile it was given. Give it more context, it uses more — which is why its tile ladder paid +0.055 from t128→t192.

**nnFormer's** fraction is lower *and shrinks* with tile size (51% → 45%). Its windowed local attention **bounds** how far information can propagate. Consequence: its own tile ladder is nearly flat (+0.0063 for t128→t192). It cannot cash in the context that made UNETR++ strong.

**The sentence for the thesis:** *local windowed attention caps context utilisation; EPA's paired channel/spatial attention does not.*

This also explains CSSA's outcome: CSSA raised the ERF, but the binding constraint on final accuracy lies elsewhere.

## 6.6 Cross-model agreement

| Question | UNETR++ | nnFormer | Agree? |
|---|---|---|---|
| Do stock architecture toggles matter? | No | No | ✔ |
| Does capacity help? | No (t320 overfit) | No (S≈M≈F, F overfits) | ✔ |
| Where in depth does the model look? | z29–34 bump, ~0 at edges | z32–33 bump, ~0 at edges | ✔ |
| Is fine crackle texture the signal? | No | No | ✔ |
| Does tile size dominate? | **Yes** (+0.16) | **No** (+0.0063 for t128→t192) | ✘ |
| Is the residual error profile the same? | 71.6% unexplained, R 0.578 | 72.1% unexplained, R 0.588 | ✔ |

Five of six agree — strong evidence those five are properties of *the task*, not of one architecture. The one disagreement is explained by 6.5.

---

# PART 7 — The probes, explained properly

These are **model-agnostic diagnostics** run after training. They answer "what did the model learn?", which the F0.5 number cannot.

## 7.1 ERF — Effective Receptive Field

**Definition — the theoretical receptive field** is the set of input pixels that *could* influence one output pixel, by architecture. The **effective** receptive field is the set that *actually does*, measured by gradient magnitude. It is typically far smaller than the theoretical one, and it is a property of the *trained* model.

**How it's measured** (`interp_erf.py`): take an ink-rich validation tile, pick the **centre output pixel**, and backpropagate to get `|∂ logit_centre / ∂ input|` for every input pixel. Average over 32 tiles. That gives a 2D map of influence.

**r50 / r90** = the radius (in pixels) of the circle around the centre containing **50%** / **90%** of the total gradient mass.

- Large r90 → the model genuinely uses far-away context.
- Small r90 → the model is effectively local, regardless of how big you make the tile.

**Why it's decisive here:** `r90 / tile` tells you whether the model can *exploit* a larger tile. nnFormer's ratio falls as tiles grow (51%→45%); UNETR++'s stays at ~53%. That single number explains the benchmark gap (6.5).

## 7.2 Inkalyzer — depth attribution via an information bottleneck

**The question:** which *depth layers* does the model actually rely on?

**The method — Information Bottleneck Attribution (IBA).** Put a controllable bottleneck on the input:

```
x̃ = g · x + (1 − g) · noise            g ∈ [0,1] per depth layer
```

`g_z = 1` → the model sees the real data at layer z. `g_z = 0` → that layer is replaced by noise (here, the depth-collapsed median). Now **optimise g** to satisfy two competing goals:

```
loss = dice_bce( f(x̃), f(x) )  +  β · mean(g)
       ^ keep the prediction unchanged      ^ pass as little information as possible
```

The converged `g_z` = **how much the model needs real information at depth z**. Run on 24 ink-rich tiles, average.

**Note:** IBA is *prediction-preserving*, not ground-truth-preserving — it explains what the model uses, which is exactly the right question here.

**Reported quantities:**
- **peak z** — the depth layer of maximum importance.
- **mass within ±2** — fraction of total importance within ±2 layers of the peak. Higher = more concentrated.
- **normalized entropy** — how *spread out* the importance profile is. 0 = all mass on one layer; 1 = perfectly flat. Higher = the model reads a wider depth band.

**Results:**

| Run | peak z | mass ±2 | entropy |
|---|---|---|---|
| `n_base_t192_r1_s42` | 32 | 0.501 | 0.792 |
| `n_srpb_t192_r1_s42` | **33** | **0.444** | **0.828** |
| `n_cssa_t192_r1_s42` | 32 | 0.496 | 0.799 |
| `nnf_s_d64_t256_s42` | 32 | 0.504 | 0.784 |
| `nnf_c1_t256_s42` | 33 | 0.483 | 0.792 |

**Finding 1 — the trust result.** Every model is a single unimodal bump peaking at z32–33 with ≈0 at the window edges. All 15 UNETR++ models peaked at z29–34. Two structurally different architectures agreeing makes this a statement about **the data**: the ink evidence really is in that band, and the `[16,48)` window is validated twice over.

**Finding 2 — SR-PB's mechanism.** SR-PB is the outlier on every axis: highest entropy, lowest concentration, peak one layer deeper, extra shallow mass. **It makes the model read a wider depth band.** And a model integrating over more depth degrades more gracefully when the window shifts — which is precisely the robustness measured in 6.2(b). The mechanism and the behaviour agree.

**Why that's notable:** in the UNETR++ study the equivalent surface-relative mechanism was **invisible** to this metric, which had to be recorded as a limitation. Here it's visible — the limitation is resolved.

**Finding 3 — an internal consistency check.** CSSA barely moves the depth profile (0.799 vs 0.792) — correct, because CSSA is a *spatial* mechanism, and its signature showed up in the ERF instead. Each contribution registers on its own axis and not the other. That's strong evidence both are doing what they claim.

**Caveat to always state:** cross-*tile* entropy comparisons are a **method artefact** — a larger tile averages more heterogeneous pixels into the per-depth gate and naturally admits flatter solutions. Compare t192 to t192 only.

## 7.3 Frequency occlusion — what spatial scale carries the signal?

**Method:** band-stop filter the input (remove one spatial frequency band), re-run inference, measure the F0.5 drop. A large drop = that scale mattered.

**Result for `nnf_s_d64_t256_s42`** (control 0.7747):

| band removed | F0.5 | drop |
|---|---|---|
| 2–4 px | 0.7720 | −0.003 |
| 4–8 px | 0.7496 | −0.025 |
| 8–16 px | 0.6023 | −0.172 |
| 16–32 px | 0.6670 | −0.108 |
| 32–64 px | 0.6726 | −0.102 |
| 64–128 px | 0.6188 | −0.156 |
| 128–256 px | 0.5846 | −0.190 |
| ≥256 px | 0.0191 | **−0.755** |

**Interpretation:** removing the finest texture (2–4 px, i.e. "crackle") costs essentially **nothing**. The model is not reading microscopic surface texture. It depends on mid-scale structure (8–16 px, stroke width) and, overwhelmingly, on the coarse/global component (≥256 px) — without which it collapses entirely.

**Why this matters for the thesis:** it revises the physics story. The intuition that models detect ink via fine "crackle" is not what the trained model does; it reads stroke-and-letterform-scale morphology plus regional context. UNETR++'s probe gave the same qualitative answer.

## 7.4 Error anatomy — where the missing recall lives

**Method:** take all the false negatives (real ink the model missed) and sort them into explanatory buckets, in priority order:

1. **dead-label components** — connected label components whose *mean predicted probability* is < 0.07. No model has ever seen anything there → likely annotation errors.
2. **boundary band ≤4 px** — FN within 4 px of a stroke edge → label-precision noise, not real blindness.
3. **low-contrast** — FN in components whose ink-vs-surroundings intensity contrast is < 3.0 (measured on the central depth slab) → physically faint ink.
4. **unexplained** — everything else. **This is the genuine attainable headroom.**

**Results:**

| Run | recall | dead | boundary | low-contrast | **unexplained** |
|---|---|---|---|---|---|
| `nnf_c1_t256_s42` | 0.588 | 0.009 | 0.137 | 0.133 | **0.721** |
| `nnf_s_d64_t256_s42` | 0.527 | 0.008 | 0.130 | 0.124 | 0.738 |
| `n_base_t192_r1_s42` | 0.537 | 0.020 | 0.127 | 0.111 | 0.742 |
| *UNETR++ `t256_lean_s42`* | *0.578* | *0.022* | *0.135* | *0.127* | *0.716* |

**Interpretation:** ~72% of misses are not explained by label noise or faintness — that's real headroom. And the *profile* is nearly identical between architectures, meaning the residual problem is a property of the task, not of the model.

## 7.5 Depth-offset stress test

Covered in 6.2(b). Mechanically: re-run inference with `center_layer` shifted by δ, score two ways — **frozen τ** (the operating point chosen at δ=0, i.e. what you'd actually deploy) and **re-swept τ** (an upper bound if you could re-tune). We report frozen-τ, because that's the deployment-relevant number.

## 7.6 TTA — Test-Time Augmentation

Run inference **8 times** on each tile — the identity plus the 7 other members of the **dihedral group D4** (horizontal flip, vertical flip, both, and the four 90° rotations) — invert each transform on the output, and average the probability maps.

**Why it works:** the model's errors are partly orientation-dependent; averaging over orientations cancels some of them. It costs 8× inference and zero training. Measured gain here: `nnf_c1_t256_s42` post 0.8122 → **0.8133**.

---

# PART 8 — Glossary

**Ablation** — removing/swapping exactly one component to measure its contribution.

**bf16 (bfloat16)** — 16-bit float with the same exponent range as fp32. Halves memory; the wide exponent avoids the overflow that plain fp16 suffers at large tile sizes.

**BCE** — binary cross-entropy, the standard per-pixel classification loss.

**Cross-attention** — attention where Q comes from one tensor and K/V from another; "A queries B".

**Dice loss** — `1 − 2|P∩G|/(|P|+|G|)`; an overlap loss robust to class imbalance.

**DropPath (stochastic depth)** — randomly skip an entire residual branch during training; a regulariser.

**ERF (Effective Receptive Field)** — the input region that measurably influences one output pixel, quantified by gradient magnitude. Reported as r50/r90 = radius containing 50%/90% of gradient mass. See 7.1.

**F0.5 / F-beta** — precision–recall combination with β=0.5, weighting precision 4× over recall. See 0.3.

**GELU** — smooth activation function, ≈ `x·Φ(x)`.

**GV-MSA** — Global Volume Multi-head Self-Attention: full attention over the whole (small) bottleneck grid.

**IBA (Information Bottleneck Attribution)** — the XAI method behind Inkalyzer; learns per-layer information gates that preserve the prediction while minimising information passed. See 7.2.

**IoU (Intersection over Union)** — `|P∩G| / |P∪G|`; reported alongside F0.5 but never used for selection.

**LayerNorm** — normalises each token's feature vector; the transformer-standard normalisation.

**Logit** — pre-sigmoid score in (−∞,∞); `sigmoid(logit)` = probability.

**LV-MSA** — Local Volume MSA: attention restricted to small 3D windows. See 1.3.

**Manifest** — a CSV listing every training tile as (segment, y, x, ink_frac, mask_frac). "Dense"/"d64" = stride 64 instead of 128, giving 4× more, overlapping tiles.

**min-size filter** — post-processing that deletes connected components smaller than a threshold. See 0.5.

**MLP** — the per-token feed-forward sub-layer of a transformer block (expand ×4, GELU, project back).

**Noise floor (±0.008)** — the measured seed-to-seed variability of F0.5. Differences below it are meaningless. See 0.6.

**Overfit signature** — validation peaks early **and** training loss keeps falling afterwards. The checkable definition of memorisation used in 5.8.

**Patch merging / expanding** — the strided-conv downsampler and transposed-conv upsampler between stages.

**Post / raw** — `raw` = best swept F0.5; `post` = after the (τ, min_size) component filter.

**Q / K / V** — query, key, value: the three projections of attention. See 1.1.

**Relative position bias** — a learned per-head scalar added to attention scores, indexed by the relative offset between two tokens. See 1.5.

**Residual connection** — `x + f(x)`; lets a block learn a correction and keeps gradients flowing.

**Seed band** — mean ± standard deviation over seeds 42/43/44. Required for any claim.

**Shifted windows** — alternating the window partition by half a window so information crosses boundaries. See 1.4.

**Skip attention** — nnFormer's decoder→encoder cross-attention replacing concatenated skips. See 2.5.

**Soft-argmax** — a differentiable "which index is the peak?": softmax the scores, then take the weighted average of indices.

**Softmax** — converts scores to a probability distribution.

**SR-PB** — our contribution C1: surface-relative additive depth bias on attention logits. See 4.1.

**CSSA** — our contribution C2: cross-scale skip attention with a zero-init gate. See 4.2.

**Stride (tiling)** — the step between tile origins. Smaller stride = more overlapping tiles.

**τ (tau)** — the probability threshold turning a probability map into a binary mask. See 0.4.

**TTA** — test-time augmentation; average predictions over the 8 dihedral transforms. See 7.6.

**Trimap / u1** — per-pixel map marking confident-ink / confident-background / **uncertain**; uncertain pixels are excluded from the loss. See 0.8.

**vFFC** — volumetric Fast Fourier Convolution (Quattrini et al. 2023); global context via frequency-domain convolution. See 4.3.

**z_flip / z_shift** — depth-axis augmentations. z_flip (mirroring depth) is physically impossible and is OFF; z_shift (±2 layers) is on.

---

# Appendix — where to look

| What | Where |
|---|---|
| Model source | `src/nnformer_ink/{backbone,blocks,attention,windows,sampling,contrib}.py` |
| Unit tests (11 checks, incl. exact-reduction proofs) | `tests/test_smoke.py` |
| Port deviations from the original (D1–D9) | `docs/PORT_LOG.md` |
| Auto-generated results + pre-registered gate verdicts | `docs/RESULTS.md` |
| Efficiency (params/GFLOPs/ms/memory) | `docs/bench_table.md` |
| Per-run outputs | `outputs/<run>/{summary.json, postprocess.json, train_log.csv, figs/}` |
| Probes | `probes/{erf,freqband,error_anatomy,inkalyzer_compare}/`, `outputs/zshift_stress/`, `outputs/tta/` |
| Presentation script | `docs/presentation.md` |
| Plan of record | `~/plan(md_files)/nnformer_plan.md` |
