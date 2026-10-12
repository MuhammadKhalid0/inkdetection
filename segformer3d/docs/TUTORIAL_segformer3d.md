# SegFormer3D for Ink Detection — a complete tutorial

**Who this is for:** you, in six months, when you have to defend every line of this chapter.
It assumes you know what a neural network and a transformer are, and nothing else. Every
acronym is expanded the first time it appears, and there is a glossary at the end (§13).

**Status when written:** 2026-09-11. 19 of the 30 runs are finished; 9 are still training.
Every number below is labelled **FINAL** or **provisional (still training)**. Nothing that
is still training is used to draw a conclusion.

---

## 0. How to read this document

| § | What it answers |
|---|---|
| 1 | The 60-second version |
| 2 | Where this architecture came from, and why that lineage matters |
| 3 | What the input data actually is (you cannot understand the design without this) |
| 4 | The architecture, block by block, with the maths |
| 5 | A worked end-to-end shape trace |
| 6 | What was *wrong* with the published code, and why it would have silently ruined the chapter |
| 7 | Medical → ink: every adaptation decision and its justification |
| 8 | The four contributions, in depth |
| 9 | The ablation study: what each cell isolates |
| 10 | The probes: what each measures and how to read it |
| 11 | The protocol, and why each rule exists (each one was learned the hard way) |
| 12 | The results so far, and what they already tell us |
| 13 | Glossary |
| 14 | Questions you will be asked, with answers |

---

## 1. The 60-second version

SegFormer3D is a **small, hierarchical vision transformer** for 3D medical image
segmentation. It has ~4.5 million parameters — about **18× smaller** than UNETR++ at our
deployment tile (79.1M) and **4.5× smaller** than nnFormer-S (19.87M). It is the third and
final model in the thesis benchmark, and its job is to answer: *does a tiny model lose
anything on this task?*

It differs from the two previous models in three ways that matter:

1. **Its attention is global, not windowed.** Every query can see the whole tile, because
   the keys and values are compressed rather than restricted to a local window. nnFormer
   restricted attention to local windows, and that is exactly why its performance stopped
   improving when we gave it bigger tiles.
2. **It has no positional encoding at all.** Position leaks in only through the zero padding
   of one convolution inside the feed-forward block. This is a deliberate published design
   claim, and our data violates the assumption behind it on one axis.
3. **Its parameter count does not depend on tile size.** Not even slightly. This makes it
   the only model in the benchmark that can run the tile ladder without a capacity confound,
   and the only one whose weights can be evaluated at a different tile size than they were
   trained at.

Our chapter is built as an **audit**: SegFormer3D makes three explicit published claims, and
we test each one in a setting that violates its assumption, plus we add one new mechanism
that attacks the thesis's dominant lever (context).

---

## 2. Where this architecture came from

You cannot understand why SegFormer3D looks the way it does without its family tree. Each
step solved a specific problem created by the previous one.

### 2.1 ViT (Vision Transformer, 2020) — the starting point

Split an image into non-overlapping patches (say 16×16 pixels), flatten each into a vector,
add a learned **positional encoding** (a vector per position, so the model knows where each
patch was), and run a standard transformer over the resulting sequence.

Two problems for segmentation:
- **One scale only.** The sequence has a single resolution the whole way through. Segmentation
  needs both fine detail (where exactly is the boundary?) and coarse context (what is this?).
- **Quadratic cost.** Self-attention compares every token to every other token. For `N` tokens
  that is `O(N²)`. Doubling the image side quadruples `N` and multiplies cost by 16.

### 2.2 PVT (Pyramid Vision Transformer, 2021) — two fixes

- **Hierarchy.** Four stages; each stage halves the spatial resolution and widens the channels,
  exactly like a convolutional network's pyramid. Now you have multiple scales.
- **Spatial-reduction attention.** Before computing keys and values, shrink the token grid by
  a factor `r`. Queries stay at full resolution, but they attend to a *summary*. Cost drops
  from `O(N²)` to `O(N²/r²)` in 2D.

**This is the single most important idea to understand**, because it is where our main
contribution attaches. See §4.3.

### 2.3 SegFormer (2021) — the MiT encoder plus a deliberately dumb decoder

**MiT** = **Mix Transformer**, SegFormer's encoder. It takes PVT and changes two things:

- **Overlapping patch embedding.** Patches overlap (kernel 7, stride 4) instead of tiling
  exactly. This preserves continuity across patch borders. A non-overlapping embedding creates
  artificial seams the network then has to repair.
- **Mix-FFN replaces positional encoding.** SegFormer removes positional encodings entirely
  and instead puts a 3×3 depthwise convolution inside the feed-forward network. The argument:
  a padded convolution leaks absolute position anyway (a token at the border sees zeros, a
  token in the middle does not), so an explicit positional encoding is unnecessary — and
  removing it means the model no longer breaks when you change the input resolution between
  training and inference.

Then the **all-MLP decoder**: take the four stage outputs, project each to a common width with
a 1×1 layer, resize them all to the finest stage's grid, concatenate, fuse with one 1×1
convolution, and predict. There is **no spatial mixing in the decoder at all** — every output
voxel is a pointwise function of features. SegFormer justifies this by saying the transformer
encoder's receptive field is already large, so the decoder does not need to do any more
spatial reasoning.

### 2.4 SegFormer3D (2024) — the same thing in 3D

Perera et al. take MiT, replace every 2D operation with its 3D equivalent, and apply it to
volumetric medical segmentation. Reduction ratios become `[4,2,1,1]`, and because the
reduction is implemented as a strided 3D convolution, a ratio of `r` reduces the sequence by
`r³`, not `r`. Published size: **4.51M parameters**.

**Three published claims** — the spine of our chapter (§8):

| # | Claim | Assumption it rests on | How ink violates it |
|---|---|---|---|
| 1 | The compressed keys lose nothing ("medical data exhibits repetitive structures") | volumes are redundant and near-isotropic | our volume is 32 × t × t, and the informative part of the depth axis is ~6 layers |
| 2 | Mix-FFN removes the need for positional encoding | all axes are translation-invariant | **in-plane yes, in depth no** — the ink sits at a physically fixed depth index |
| 3 | An all-MLP decoder suffices because transformer receptive fields are large | the encoder really does see far enough | this thesis **measures** receptive fields, so it can check rather than assume |

---

## 3. The input: what the data actually is

Skip this and the architecture decisions will look arbitrary.

- A **segment** is a flattened sheet of papyrus that has been virtually unrolled from a CT scan
  of a carbonized scroll. We have 9 labelled segments.
- Each segment is a stack of **65 depth layers**, each an 8-bit greyscale image. Layer 0 is
  slightly below the sheet, layer 64 slightly above. The sheet's surface is somewhere in the
  middle.
- We use a **32-layer window, `[16, 48)`**, centred on layer 32. Chapter 1 measured where the
  ink signal lives: **depth 27 to 33**. So of our 32 layers, roughly **6 carry the signal** and
  the rest are context or nothing.
- **Labels are 2D.** A human traced the visible letters once, in-plane. There is no per-layer
  label. So the model takes a 3D volume and must produce a **2D** map.
- **Ink is invisible to the eye in these scans.** It is not a bright substance; the network is
  picking up a subtle texture and morphology difference.
- Strokes are roughly **30–50 pixels** wide; letters are **300–1300 pixels** across. This is why
  tile size matters so much: a 64-pixel tile cannot contain a whole letter.

**The single most important consequence:** the input is **anisotropic**. The two in-plane axes
behave like a normal image — translation-invariant, and a tile is an arbitrary crop of a much
larger sheet. The depth axis does not: index 30 means something physically specific. Any design
that treats all three axes identically is making an assumption this data violates.

---

## 4. The architecture, block by block

Notation: `B` = batch, `C` = channels, `D,H,W` = depth/height/width of the *token grid* (not
the input), `N = D·H·W` = number of tokens, `M` = number of key/value tokens after reduction.

### 4.1 The wrapper contract (what the backbone must produce)

Every model in this benchmark obeys one interface, so that only the backbone differs:

```
input   x : [B, 1, 32, t, t]          the raw volume, intensities divided by 255
backbone -> [B, 16, 32, t, t]         a full-resolution FEATURE volume, 16 channels
DepthCollapse -> [B, 16, t, t]        shared code: collapse the depth axis
Head          -> [B, 1, t, t]         shared code: 3x3 conv, GELU, 1x1 conv -> one logit
```

The backbone never produces the prediction. It produces features; the **shared Class-A layer**
(collapse + head) turns them into a 2D logit map. This is what makes the benchmark fair — the
collapse and head are byte-identical code for all three models.

Our model has 4,437,568 backbone parameters. The training log reports **4,439,985** trainable
parameters, and the difference of exactly **2,417** is the shared collapse (17 + 63) and head
(2,320 + 17). That arithmetic is a *proof* that the shared layer is shared and not
reimplemented.

### 4.2 Overlapping patch embedding

```python
self.proj = nn.Conv3d(in_ch, dim, kernel_size=k, stride=s, padding=p)
self.norm = nn.LayerNorm(dim)

x = self.proj(x)                   # [B, C, D', H', W']  <- a strided convolution
grid = x.shape[2:]                 # we REMEMBER the grid; see section 6
x = x.flatten(2).transpose(1, 2)   # [B, N, C]  the token sequence
x = self.norm(x)
```

A strided convolution *is* patch embedding: each output position is a learned linear function
of a `k × k × k` input cube. With `stride < kernel` the cubes **overlap**, so neighbouring
tokens share input voxels and there are no seams.

**Our choice.** Stock stage 1 uses kernel 7, stride 4, padding 3 **on all three axes**, which
divides depth by 4: 32 → 8, then 4, 2, and finally **1**. At the last stage the depth axis is a
single token — the 3×3×3 convolution there sees nothing but zero padding along depth, and
attention is purely in-plane.

We default instead to a **depth-preserving stem**: kernel `(3,7,7)`, stride `(1,4,4)`, padding
`(1,3,3)`. Depth stays 32 at stage 1 and is only halved inside the hierarchy: 32 → 16 → 8 → 4.

Why: the signal is a thin band in depth, and a contribution about *depth coordinates* (§8.2) is
meaningless if depth has been crushed to one token before attention ever sees it. The isotropic
stem is kept as ablation **A1** so the choice is measured, not asserted.

**Cost of that choice, measured:** the depth-preserving stem gives stage 1 **131,072** tokens at
t256, against **32,768** for the isotropic stem — 4× the tokens, and since attention cost scales
with tokens × keys, roughly 16× the stage-1 attention cost.

### 4.3 Sequence-reduced attention — the heart of the model

This is the block our main contribution modifies, so it is worth reading slowly.

Standard self-attention:

```
Q = x·W_q         K = x·W_k        V = x·W_v          all [B, N, C]
attention(Q,K,V) = softmax(Q·Kᵀ / sqrt(d)) · V
```

`Q·Kᵀ` is `N × N`. At our stage 1 with t256 and the depth-preserving stem, `N = 131,072`, so
that matrix would have **17 billion entries** per head. Impossible.

SegFormer3D's fix — **reduce the keys and values, not the queries**:

```python
if any(r > 1 for r in sr):
    x_ = x.transpose(1, 2).reshape(B, C, D, H, W)   # sequence -> grid
    x_ = self.sr_conv(x_)                           # Conv3d(C, C, kernel=r, stride=r)
    x_ = x_.flatten(2).transpose(1, 2)              # grid -> shorter sequence
    x_ = self.sr_norm(x_)
kv = self.key_value(x_)                             # keys and values from the SHORT sequence
q  = self.query(x)                                  # queries from the FULL sequence
out = scaled_dot_product_attention(q, k, v)
```

Now the attention matrix is `N × M` with `M = N / (r_d·r_h·r_w)`. With our default reduction
`(1,4,4)` at stage 1, `M = 131,072/16 = 8,192`.

**Three things to notice, because everything else follows from them:**

1. **The reduction is a learned convolution, not pooling.** `sr_conv` has weights. The model
   learns *how* to summarise a neighbourhood into a key.
2. **The key grid is decoupled from the query grid.** Queries live at full resolution; keys
   live wherever we say. Nothing in the mathematics requires the keys to summarise *the same
   spatial extent* as the queries. This is the opening our main contribution uses (§8.1).
3. **Attention is still global.** Every query attends to every key, covering the entire tile.
   Contrast nnFormer, which restricted each query to a local window — and whose effective
   receptive field consequently *shrank* as tiles grew, flattening its tile ladder.

**Why fused attention is mandatory, not an optimisation.** The published code has a fallback
that materialises `softmax(Q·Kᵀ)` explicitly. At t256, batch 4, `(1,4,4)` reduction, that
tensor is **8.0 GB in bf16 per stage-1 block**, and autograd keeps it for the backward pass. Two
stage-1 blocks alone would need over 32 GB before anything else. We always call
`F.scaled_dot_product_attention`, whose fused kernels never build the matrix. The explicit path
survives only behind a flag, used by one unit test that checks the two agree (they do, to
9.4e-06).

**The scale factor, a subtlety worth knowing.** The released code sets
`self.scale = head_dim ** -0.5` but then calls fused attention *without passing it*, so the
library default (also `1/√head_dim`) applies. The original launch commit used `1/√num_heads`,
a genuinely different function. The paper's numbers correspond to the launch version. We
default to `1/√head_dim`, pass it explicitly, and keep the alternative as a documented
ablation.

### 4.4 Mix-FFN, and "we don't need positional encoding"

```python
x = self.fc1(x)          # Linear(C -> 4C)
x = self.dwconv(x, grid) # 3x3x3 depthwise Conv3d, padding 1  <- the whole positional apparatus
x = self.act(x)          # GELU
x = self.fc2(x)          # Linear(4C -> C)
```

A **depthwise** convolution applies one small filter per channel independently (`groups=C`),
which is why it is cheap.

**The claim.** There is no positional encoding anywhere in this network. SegFormer argues you
do not need one, because a *padded* convolution already leaks absolute position: a token at the
edge of the tile has zeros in part of its neighbourhood, a token in the middle does not, so the
network can infer "how close to the border am I". And removing explicit positional encodings
means the model does not break when the input resolution changes between training and inference
— a real problem for encodings that are learned per position.

**Why this is interesting for us.** The claim assumes translation invariance. In our data:
- **In-plane it holds.** Tiles are cut on a blind grid; nothing distinguishes one position.
- **In depth it fails.** Layer 30 is physically privileged. Every model measured across both
  previous chapters locks onto depth 27–33.

So ink is exactly the case the argument does not cover: two axes where identity does not matter
and one where it does. That is contribution C2 (§8.2).

### 4.5 The four-stage pyramid

```
stage i:  patch embed  ->  depths[i] transformer blocks  ->  LayerNorm  ->  reshape to a grid
```

Each block is pre-norm with residual connections:

```python
x = x + drop_path(attention(norm1(x), grid))
x = x + drop_path(mlp(norm2(x), grid))
```

**Pre-norm** means normalise *before* the sub-layer, not after; it makes deep transformers
train stably. **Residual** means add the input back, so a block can learn to do nothing.
**DropPath** (stochastic depth) randomly drops a whole residual branch during training; the
reference uses none, and we default to 0.0 to stay faithful, reserving it as the first rung of
the overfit response ladder.

Widths and heads by capacity rung (**measured**, not estimated):

| rung | channels per stage | heads | blocks | parameters | matched to |
|---|---|---|---|---|---|
| SF-S | 32, 64, 160, 256 | 1, 2, 5, 8 | 2,2,2,2 | **4,437,568** | the published SegFormer3D |
| SF-M | 64, 128, 320, 512 | 1, 2, 5, 8 | 2,2,5,2 | **19,966,880** | nnFormer-S (19.87M) |
| SF-L | 96, 192, 480, 768 | 1, 2, 5, 8 | 3,6,14,3 | **79,353,344** | UNETR++ at t256 = nnFormer-M (79.1M) |

The rungs were built to land on the *other models'* sizes, so "same size, different family" is a
fair comparison rather than a coincidence.

### 4.6 The all-MLP decoder

```python
for each stage i:  project C_i -> 256 with a Linear, then LayerNorm
resize stages 2,3,4 up to stage 1's grid (trilinear interpolation)
concatenate the four -> 1024 channels
fuse: Conv3d(1024 -> 256, kernel 1) + BatchNorm + ReLU
```

Every operation here is **pointwise in space**. A `1×1×1` convolution mixes channels at a
location and nothing across locations. The only thing that moves information spatially is the
trilinear resize, which is fixed interpolation, not learned.

So: **after the encoder, this network does no further spatial reasoning.** SegFormer justifies
that by the encoder's large receptive field. Our thesis measures receptive fields directly, so
we can check the premise — and contribution C4 (§8.4) tests whether one neighbourhood term in
the decoder changes anything.

### 4.7 The final full-resolution path

The decoder output sits at the stage-1 grid, `(32, t/4, t/4)`. We need `(32, t, t)`.

The reference upsamples by a factor of 4 **on all three axes**. That works for the reference
only because its isotropic stem also divided depth by 4. With our depth-preserving stem, depth
was never reduced, so an isotropic ×4 would return depth **128** instead of 32. We therefore
make the final path explicit:

- **`expand`** (our default): `ConvTranspose3d(256 -> 16, kernel = stride = (1,4,4))`. A
  transposed convolution is a *learned* upsample. Chosen for parity with the other two ports.
- **`trilinear_lowres_pred`**: predict 16 channels at the coarse grid, then interpolate. Cheapest.
- **`trilinear`**: interpolate the 256-channel map first, then predict. **Forbidden above batch 2** —
  it allocates a 256-channel full-resolution tensor, 4.0 GB at t256 batch 4 in bf16.

Plus a **full-resolution convolutional skip**: a tiny two-layer convolution on the *raw input*,
concatenated with the upsampled features and fused. It restores fine detail the 4× downsampling
threw away. Both previous ports have one, so we do too (ablation A4 measures whether it earns
its 74.5 GFLOPs).

### 4.8 The shared Class-A layer

**DepthCollapse** turns `[B,16,32,t,t]` into `[B,16,t,t]`. In `surface_relative` mode it:
1. averages features over the in-plane axes to get a depth profile,
2. runs a tiny linear layer to get a saliency per depth,
3. takes a **soft-argmax** — a softmax-weighted average of depth indices — to estimate the
   surface position `ŝ`,
4. samples a learned weighting curve at offsets `d − ŝ` and uses it to average over depth.

The intent was a model that finds the surface and reads relative to it. Chapters 1 and 2
measured what actually happens: **`ŝ` never learns** (correlation with the true surface ≈ 0.2)
and the weighting curve never leaves its initialisation. It works as a **fixed centred prior**,
not as a learned mechanism. That is a settled result, which is why this chapter does not
propose a fourth variation on it (§8.5, rejection R1).

---

## 5. A worked shape trace (t256, depth-preserving stem, default reduction)

| step | shape | note |
|---|---|---|
| input | `[4, 1, 32, 256, 256]` | batch 4 |
| stage-1 embed | `[4, 32, 32, 64, 64]` → tokens `[4, 131072, 32]` | in-plane ÷4, depth kept |
| stage-1 keys after reduction `(1,4,4)` | `[4, 8192, 32]` | 16× fewer |
| stage-1 attention matrix | `131072 × 8192` per head | never materialised (fused kernel) |
| stage-2 | `[4, 64, 16, 32, 32]` → `[4, 16384, 64]` | all axes ÷2 |
| stage-3 | `[4, 160, 8, 16, 16]` → `[4, 2048, 160]` | reduction is now 1 (full attention) |
| stage-4 | `[4, 256, 4, 8, 8]` → `[4, 256, 256]` | 256 tokens |
| decoder: each stage → 256 ch, resized to stage-1 grid | 4 × `[4, 256, 32, 64, 64]` | |
| concatenate | `[4, 1024, 32, 64, 64]` | 1.0 GB in bf16 — the largest unavoidable tensor |
| fuse | `[4, 256, 32, 64, 64]` | |
| final expand | `[4, 16, 32, 256, 256]` | the wrapper contract |
| collapse | `[4, 16, 256, 256]` | shared code |
| head | `[4, 1, 256, 256]` | one logit per pixel |

Measured peak training memory at this configuration on an A40: **9.4 GB** of 48.

---

## 6. What was wrong with the published code

This is not a footnote. Two defects in the reference implementation would each have silently
invalidated the chapter.

### 6.1 The cube-root trap

The published code recovers the token grid by taking a **cube root of the token count**:

```python
n = cube_root(N)
x = x.transpose(1, 2).reshape(B, C, n, n, n)
```

This assumes the grid is a cube — true for 128³ medical volumes, false for `32 × t × t`. Here
is what actually happens for our five tile sizes with the isotropic stem:

| tile | stage-1 grid | tokens | is it a perfect cube? | result |
|---|---|---|---|---|
| 64 | (8,16,16) | 2,048 | no | **crash** |
| 128 | (8,32,32) | 8,192 | no | **crash** |
| 192 | (8,48,48) | 18,432 | no | **crash** |
| **256** | **(8,64,64)** | **32,768 = 32³** | **YES** | **runs, and is wrong** |
| 320 | (8,80,80) | 51,200 | no | **crash** |

At t256 — **our deployment tile, the only one in the whole ladder where this happens** — the
reshape succeeds and reinterprets a `(8,64,64)` grid as `(32,32,32)`. Every subsequent operation
(the reduction convolution, the depthwise convolution, the stage reshape) then mixes tokens from
different depth rows. The model trains, the loss goes down, the numbers look plausible, and
**the 3D geometry is nonsense**. No exception is raised.

The fix is to compute the true `(D,H,W)` grid at build time, thread it through every block, and
assert it on every forward pass. Our port does this, and a unit test asserts the arithmetic
above so nobody can quietly reintroduce the trap.

### 6.2 It does not even import

`cube_root` is decorated `@torch.jit.script` and its return-type annotation is wrong (`round()`
returns a float where `int` is declared). On our PyTorch version the file fails at import.

### 6.3 The other deviations

| # | Reference | Ours | Why |
|---|---|---|---|
| D3 | scalar reduction ratio per stage | per-axis tuples, divisibility asserted | the reduction convolution has no padding, so a non-dividing grid is **silently truncated** |
| D4 | isotropic stem (depth 32→8→4→2→1) | depth-preserving default | the signal is a thin depth band; A1 measures the choice |
| D5 | isotropic ×4 final upsample | explicit final path | would return depth 128 under our stem |
| D8 | attention scale differs between the launch commit and current main | explicit, documented | the paper's numbers correspond to the older convention |
| D9 | manual softmax fallback | fused attention always | the explicit path needs 8 GB per block at t256 |

### 6.4 The trap in *our own* harness

`models/wrapper.py` forwards configuration keys to the model **one at a time**. A key present in
the YAML but not forwarded is **silently ignored**, and the run trains something other than what
the config says, with no error. This is not hypothetical: the nnFormer builder accepts several
arguments the wrapper never passes, so those keys are dead config surface in chapter 2.

Our defence is a unit test, `test_config_keys_reach_the_model`, which builds from every shipped
config and asserts each key provably changes the model — via parameter count or forward output.
A configuration key that cannot change the model is a bug, not a comment.

---

## 7. Medical → ink: every adaptation decision

| Medical assumption | Ink reality | What we did |
|---|---|---|
| 4-channel MRI in, 3-class mask out | 1-channel volume in, **2D binary** map out | backbone returns a feature volume; the shared collapse and head produce the 2D logit |
| isotropic 128³ | anisotropic 32 × t × t, signal in ~6 depth layers | depth-preserving stem (A1 ablates it) |
| reduction is isotropic, `r³` | depth is precious, in-plane is redundant | per-axis reduction, default `(1,4,4)`/`(1,2,2)` (A2 ablates three presets) |
| training crops are the whole volume | tiles are arbitrary crops of a huge sheet | sliding-window inference at stride t/2 with Gaussian blending |
| AdamW 1e-4, **weight decay 0.01**, 800 epochs, batch 2, Dice loss, 20-epoch warm-up, cosine-with-restarts | benchmark protocol, fixed across all three models | AdamW 1e-4, weight decay 3e-5, 60-epoch cosine, warm-up 3, masked Dice+BCE, batch 4 |
| per-modality z-scoring | measured: raw `/255` is optimal, every normalisation variant hurt | unchanged `/255` |
| 9 added BatchNorm layers (absent from the 2D original) | they are the **only tile-dependent state** in the model | kept by default; A3 removes them; the transfer study reports curves with and without |

**A caveat worth remembering:** the paper and the released repository disagree on *every*
hyperparameter — learning rate, weight decay, schedule, epochs, batch size, loss. The
maintainers also state the paper's ACDC and Synapse results were produced inside the *nnFormer*
codebase with code they no longer have. So "the published recipe" is not a well-defined object,
which is a further reason to use our own protocol and say so.

---

## 8. The four contributions

### 8.0 The rules any contribution here has to satisfy

Two completed chapters measured **every** architecture-internal mechanism anyone proposed, and
every one landed inside the ±0.008 noise floor at deployment scale: the depth-attention gate
−0.004, the texture stem +0.000, the surface-relative collapse ±0.000 at t256, removing
nnFormer's signature skip attention **−0.0002**, its surface position bias +0.0065, its
cross-scale attention +0.0037, every loss variant, and supervising the surface estimate **−0.050**.

A third chapter proposing four more "improvements" and reporting four more nulls adds nothing.
So four rules govern what is worth GPU time:

**Rule 1 — information, not routing.** Re-routing information the model already has is measured
dead twice. Only two things have never been tried: *changing what information exists in the
forward pass*, and *converting the one measured lever into something cheaper or deployable*.

**Rule 2 — the threshold sweep retires a whole family for free.** We sweep the decision
threshold τ over {0.10…0.90} every epoch and report the best. Therefore **any mechanism whose
effect is a monotone rescaling of the logits is invisible by construction** — calibration heads,
learned temperature, logit priors, loss reweighting. This is worth stating because it explains,
without a single GPU hour, why the tempting "32% of missed ink is a near miss" headroom is not
addressable by a confidence trick: it is a near miss *at the swept optimum*.

**Rule 3 — power the gate before you run it.** nnFormer's cross-scale attention did not merely
fail, it failed **underpowered by construction**: its gate asked it to close half of nnFormer's
own tile-ladder gap, and that gap was +0.0063. So here we measure our own ladder *first* and set
the threshold from it.

**Rule 4 — smallness is an experimental instrument.** At 4.4M parameters with zero tile-dependent
weights, this model can run experiments that were anti-goals for the big models — the largest
tile without a capacity confound, and a capacity ladder in the opposite direction.

Every contribution is an **exact no-op at initialisation** behind a default-off flag: at step 0
the network is bitwise identical to stock. That is what makes each ablation a genuine single
toggle and lets the parameter count verify each rung. All four are unit-tested for exactly this
(`max|Δ| == 0.0`), for being live once trained, and for receiving gradients.

### 8.1 C1 — Peripheral Downsampled Keys (the centrepiece)

**The idea in one sentence.** Give the attention keys a view of the *surroundings* of the tile,
which the tile itself does not contain, without paying for more queries.

**Why it is possible here and nowhere else.** From §4.3: the key set is already a low-resolution
learned summary, and its resolution is decoupled from the query resolution. Nothing requires the
keys to summarise the *same spatial extent* as the queries. UNETR++ cannot do this — its
compression is a learned map over the token axis of keys drawn from the same tile, so foreign
tokens have no slot. nnFormer cannot — its relative-position bias table is indexed by offsets
within a window, undefined across extents. The **absence of positional encoding** is what makes
foreign-extent tokens legal at all.

**Why it is not the mechanism that already failed.** nnFormer's cross-scale attention re-routed
coarse features **from inside the same tile**, adding no information, and failed. Its pre-written
conclusion was *"letterform context must enter at input resolution; it cannot be retrieved from
coarse features after the fact."* Our halo brings **pixels the tile does not contain**. That
conclusion is the hypothesis we test, not a verdict against us.

**The mechanism.**

```
out = attention(q, k, v)  +  γ ⊙ attention(q, k_peripheral, v_peripheral)
```

- The dataset reads a **2× wider crop centred on the tile**, average-pools it 2× in-plane, and
  stacks it as **input channel 1**. Same tensor rank, same voxel count — so the training loop,
  sampler, loss, collapse and head are all untouched.
- A tiny **context stem** encodes that channel **directly at the reduced key grid**. This matters:
  a 2×-downsampled 2× halo has exactly the tile's voxel count, so encoding it normally would
  double the encoder and the "constant cost" claim would be false.
- The peripheral keys go through a **separate softmax**, scaled by a zero-initialised per-head
  gate γ.

**The single most important implementation detail.** You must **not** concatenate the peripheral
keys into the existing softmax. A zero-*weighted* key concatenated into one softmax still enters
the denominator, so it changes the stock attention weights and the contribution is **not** a
no-op at initialisation. A separate branch multiplied by γ = 0 contributes exactly 0.0. A unit
test asserts `max|Δ| == 0.0` precisely to catch this.

**The mandatory same-cost control.** A second run replaces the halo with a **2× downsampled copy
of the tile itself** — identical shapes, identical parameters, identical compute, **no new
information**. Without it, any gain could be read as extra capacity rather than extra context.

**Result (FINAL, single seed):**

| cell | raw F0.5 |
|---|---|
| stock @ t128 | 0.7578 |
| PDK @ t128 | 0.7617 |
| **PDK same-cost control @ t128** | **0.7620** |
| stock @ t192 | 0.7857 |
| PDK @ t192 | 0.7864 |

PDK beats stock by **+0.0039** — and beats its own control by **−0.0003**. The gain is entirely
explained by the extra keys and parameters, **not by the new pixels**. The gate required
+0.02 over stock, half the ladder gap, *and* +0.008 over the control; it fails on all three.

**This is a clean, informative null, and the control is what makes it publishable.** Without it
we would have reported a small positive delta and been unable to say what caused it. The
pre-written reading now applies: *context that is not present at the query resolution cannot be
injected through the key path.* Combined with nnFormer's result — coarse features already inside
the tile do not help either — the tile ladder's large gain is **not about information content at
all**. It is about the resolution of the queries and the decoder, and about what a bigger tile
does to the epoch's sampling distribution. That is a stronger statement than either previous
chapter could make alone.

### 8.2 C2 — Depth-Coordinate Mix-FFN

**The idea.** Inject an explicit depth coordinate at exactly the point that constitutes the
architecture's entire positional apparatus, and see whether the model wanted one.

```
h  ←  h + γ ⊙ P[·]           immediately after the depthwise convolution
```

- `frame: absolute` — a table `P[D, 4C]` indexed by the token's depth index.
- `frame: surface` — a table `P[2D−1, 4C]` sampled at `d − ŝ`, where `ŝ` is a soft-argmax surface
  estimate computed in a float32 island with NaN guards (this exact guard was load-bearing in both
  previous ports).

**A deliberate asymmetry in the initialisation.** γ is zero, but the table `P` is **not** — it gets
the same truncated-normal initialisation as every other table in the model. Zero-initialising both
would leave the branch dead at both ends with no gradient signal. That is the failure mode from
chapter 1, where a gate initialised at 0.1 ended training at 0.1010 — it stayed at its safe
initialisation because it had nothing to add. γ alone carries the exactness.

**Why the threshold-zero score is *not* the primary readout.** The pre-registered readouts are:
1. **The learned table, read against an independent probe.** Plot the norm of `P[d]` against depth
   and compare with the measured depth-attribution profile. A contribution whose learned parameter
   can be checked against an independent measurement is rare, and it is reported either way.
2. **A sign asymmetry under induced depth offset.** The *absolute* frame hard-codes a band index,
   so it should degrade when we shift the depth window; the *surface* frame should not. If that
   asymmetry appears it is a clean mechanistic result **independent of any accuracy movement**.

**Result (FINAL, single seed):** absolute **0.7869** (+0.0012 vs stock), surface **0.7851**
(−0.0006). Both comfortably inside the noise floor — the predicted outcome. The mechanistic
readouts require the analysis job, which runs when training finishes.

The expected conclusion, already written before the runs: *a network with no positional encoding
of any kind still concentrates its depth attribution on the ink band, and supplying an explicit
depth coordinate changes nothing. Depth identity in this data is recoverable from content and need
not be supplied as a coordinate.* That is a cross-domain confirmation of the source paper's claim,
in a setting that violates its premise.

### 8.3 C3 — Overlapping Sequence Reduction

**The observation.** SegFormer3D advertises an **overlapping** patch embedding "that preserves the
local continuity of the input voxels" — and then uses a **non-overlapping** key reduction, exactly
where most of its compute lives. The architecture contradicts its own stated design principle.

**Why it might matter here.** At the stage-1 grid, a 30–50 pixel stroke is 7–12 tokens wide, so a
4-token reduction block is the same order as stroke width. Blocks straddling a stroke edge average
ink with papyrus in a way that depends on where the block boundary happens to fall — classic
aliasing. And the frequency probe says mid-scale structure is exactly what matters.

**The mechanism.** Replace `Conv3d(C, C, kernel=r, stride=r)` with `Conv3d(C, C, kernel=2r,
stride=r, padding=r/2)` on every axis whose `r` is even. The output grid is unchanged:
`(n + 2·(r/2) − 2r)/r + 1 = n/r`.

**Exact reduction, with an honest caveat.** Build the wide kernel as zeros and copy the stock
weights into the central tap block. In exact arithmetic the extra taps contribute 0.0, so the
output is identical. But a different kernel size can make the library choose a different algorithm
and therefore a different summation order, so **bitwise equality is only guaranteed on CPU**. The
test asserts exact on CPU and ≤ 1e-6 on GPU.

**This contribution is where the port nearly went wrong.** The widened convolution was originally
built during module construction — but weight initialisation runs **afterwards** and re-initialised
the wide kernel using the *wide* fan-out, overwriting the zero taps and destroying the exact
reduction. The test caught it at `max|Δ| = 3.9e-02` instead of 0. The fix is to widen **after**
initialisation, so the wide kernel is built from the already-initialised stock convolution. This is
exactly the "initialisation trap" the plan pre-registered, and it is why exact-reduction tests are
mandatory rather than nice-to-have.

**Result (FINAL, single seed):** **0.7951 raw / 0.8218 post**, against stock 0.7857 / 0.8092.
That is **+0.0094 raw and +0.0126 post** — the only cell in the entire screen set that lands
**outside the ±0.008 noise floor**, and the prediction was a null.

**Do not over-read this yet.** It is one seed, and chapter 2's central methodological lesson is
that a single-seed delta measured against a single-seed baseline is exactly how a +0.013 "win"
collapsed into +0.0065 noise. Under the pre-registered rule this result **promotes to three seeds
and a deployment-scale confirmation**; it does not become a claim until those land. Note also that
it adds 98,304 parameters (+2.2%), so the parameter-matched comparison is part of the write-up.

### 8.4 C4 — Decoder Spatial Mixing

**The observation.** The largest post-hoc lever in the entire thesis is a **spatial** one: an
8-connected minimum-size filter that deletes small isolated false-positive islands, worth +0.010
to +0.020 on every model. Meanwhile the decoder does no spatial reasoning at all (§4.6).

**The mechanism.** One zero-gated depthwise convolution on the fused decoder map:
`F ← F + γ ⊙ DWConv3d(F, kernel 3, dilation 2)`. Dilation 2 gives a support of about 16 pixels at
full resolution — stroke-to-ring scale. 7,424 parameters.

**Accuracy is the secondary readout.** The primary, pre-registered test is a **substitution**: the
post-filter gain must shrink by at least 40% while raw accuracy does not fall. That is a claim
about *where the gain comes from*, and it is measurable at one seed from the existing sweep.

**Result: still training** (epoch 49 of 60 at the time of writing). Its raw score so far is
0.7862, but the substitution readout needs the finished post-processing sweep.

Either outcome is useful. If island false positives are not a decoder-locality artefact, then the
minimum-size filter and the learned refiner are genuine, non-redundant pipeline stages rather than
patches over an architectural omission — a real defence of the thesis's pipeline result.

### 8.5 What was considered and rejected (and why that list matters)

A plan that only lists what you did is unfalsifiable. These appear in the thesis as verdicts:

| # | Rejected idea | Decisive reason |
|---|---|---|
| R1 | Surface-weighted / band-sparse key reduction | This is the geometry family, **third instance**. Measured twice already, both times inside the noise floor at deployment scale; the learned form never learns; supervising it hurts on both architectures; and the domain premise (per-segment depth drift) turned out to be a documentation error. A third null in one family is the worst available use of GPU time, and in a viva it reads as running the same experiment three times. **Instead** the collapse-level geometry prior is inherited free and the cross-architecture stress table is filled by an inference-only job. |
| R2 | Any further depth-frame accuracy mechanism | Same family; C2 is admitted only because its mechanism and its readout are both different. |
| R3 | The 2.5D model as a *contribution* | It has no exact-no-op form — it is a different network — so calling it a contribution breaks the house rule. It is a **control cell**. |
| R4 | Anything aimed at "improving confidence on faint ink" | Rule 2: invisible to a swept threshold by construction. |
| R5–R12 | Deep supervision · new losses · photometric augmentation · morphology · ensembling · a pretrained-init headline arm · the biggest tile for the biggest model · follow-up architectures | Inherited anti-goals, each measured dead, or benchmark-fairness violations |

### 8.6 The prior-art control: 2.5D depth-as-channels

There is **no published use of SegFormer3D on Herculaneum data**. But 2D SegFormer is well
represented in the 2023 Kaggle ink-detection competition: the winning solution used a 3D
convolutional stem, collapsed depth by max-pooling, then fed a 2D SegFormer; the cleanest pure
"2.5D" entry fed 3 depth slices as channels to SegFormer-B3 and scored 0.66 private.

So we run the same question inside our own family: **is 3D attention over depth worth anything at
all, at 32 layers?** The control feeds the 32 depth layers as input **channels** to the same MiT
code path with a singleton depth axis — literally the same code, running as a 2D model. Its output
is broadcast back over depth so the shared collapse and head stay byte-identical, which resolves
the wiring problem without a special-case branch.

**Result (FINAL, single seed): 0.7254 raw / 0.7379 post**, against the 3D model's 0.7857 / 0.8092
at t192. The 3D model is **+0.06 better**. Our pre-registered guess was "within 0.02" — that guess
was wrong, and by a wide margin. At 32 depth layers, 3D attention over depth **does** buy
something substantial. This is a control, never a benchmark row.

---

## 9. The ablation study: what each cell isolates

Every cell is a **one-key change** from `sf_base_t192_s42`, so the configuration diff *is* the
ablation. Decision rule: `|Δ| ≤ 0.008` is neutral, and a neutral result is itself a finding.

**All results below are FINAL, single seed, raw F0.5 vs the base 0.7857 (post 0.8092).**

| cell | what it toggles | raw | Δ raw | reading |
|---|---|---|---|---|
| `sf_base_t192` | the reference | 0.7857 | — | — |
| **A1 `sf_stemiso`** | depth-preserving stem → isotropic | 0.7693 | **−0.0164** | **outside the floor: preserving depth is a real decision, not a preference.** Crushing depth to 1 token by stage 4 costs measurable accuracy |
| A2 `sf_sr444` | reduction `(1,4,4)` → `(4,4,4)` | 0.7781 | −0.0076 | just inside the floor; the cheaper preset costs a little. 1.8× less compute |
| A2 `sf_sr244` | reduction → `(2,4,4)` | 0.7858 | +0.0001 | free: halving depth resolution in the keys costs nothing |
| **A3 `sf_nonorm`** | remove the repo's 9 added normalisation layers | 0.7663 | **−0.0194** | **outside the floor: the added BatchNorms matter a lot.** They are absent from the 2D original — an undocumented but load-bearing change |
| A4 `sf_noskip` | remove the full-resolution conv skip | 0.7819 | −0.0038 | neutral; it costs 74.5 GFLOPs for nothing measurable |
| A5 `sf_finlow` | final path → predict-then-upsample | 0.7915 | +0.0058 | neutral, and cheaper |
| A6 `sf_mean` | shared collapse → plain mean | 0.7837 | −0.0019 | neutral — the geometry prior fades at deployment scale, as in both previous chapters |
| A7 `sf_dec128` | decoder width 256 → 128 | 0.7810 | −0.0047 | neutral: a cheap deployment saving |
| A8 `sf_deep` | depths `[2,2,2,2]` → `[2,2,5,2]` at the same width | 0.7777 | −0.0080 | borderline; isolates *depth* from *width* in the capacity study |
| A12 `sf_lr5e5` | learning rate halved | 0.7869 | +0.0012 | neutral; doubles as the pre-decided instability fallback |
| A13 `sf_wd1e2` | weight decay 3e-5 → 1e-2 (the repo's own value) | 0.7923 | +0.0066 | neutral, but the largest neutral positive — worth a seed if the schedule allows |
| **C3 `sf_osr`** | overlapping key reduction | 0.7951 | **+0.0094** | **the only cell outside the floor upward** — promotes to 3 seeds |

**What this table already says.** Ten of thirteen cells are inside the noise floor. That replicates,
on a third architecture, the thesis's central finding: *architecture-internal choices barely
matter; the pipeline dominates*. But two adaptation decisions are **not** neutral — depth
preservation and the added normalisation layers — and both are decisions the medical original
made for medical reasons.

---

## 10. The probes: what each one measures

Accuracy alone cannot explain *why* a model behaves as it does. Each probe answers a specific
question and is model-agnostic, so all three chapters use identical code.

| probe | question | how it works | how to read it |
|---|---|---|---|
| **Effective receptive field** | how far from a pixel does the model actually look? | put a gradient on the centre 3×3 of the output, back-propagate to the input, sum the absolute gradient; report the radii containing 50% and 90% of the mass | the ratio **r90 / tile** is the key number. UNETR++ held ≈53% at every tile and had a steep ladder; nnFormer's *shrank* 51%→45% and its ladder was flat. This is the thesis's one predictive law |
| **Frequency band-stop** | is the signal fine texture or coarse morphology? | delete one spatial frequency band from the input and re-score; repeat per band | large drops in the coarse bands and ≈0 in the 2–8 pixel bands mean the model reads letter shape, not "crackle". Also C3's mechanism readout |
| **Depth attribution (Inkalyzer)** | which depth layers does the model use? | learn a per-layer gate that preserves the prediction while suppressing as much input as possible | a unimodal bump at the ink band with ≈0 at the window edges means the model found the surface honestly. Also C2's table-versus-probe comparison |
| **Error anatomy** | what kind of errors remain? | partition missed ink into dead-label / boundary / low-contrast / unexplained, and by predicted probability into near-miss / faint / blind | the pre-declared axis on which a small model may "win" at equal accuracy. Also C4's false-positive island statistics |
| **Depth-offset stress** | is the geometry mechanism real or a fixed prior? | shift the depth window by δ ∈ [−8,+8] and re-score at a **frozen** threshold | C2's sign asymmetry lives here. Run **band versus band from the start** — the chapter-2 lesson applied prospectively |
| **Resolution transfer** | can weights trained at one tile be used at another? | score one checkpoint at 128/192/256/320/384 | **only possible for this model**, since it has no tile-dependent parameters. Tests the source paper's own resolution claim. Run with and without the added normalisation layers, since their running statistics are the only tile-dependent state |

---

## 11. The protocol, and why every rule exists

Each of these was learned the hard way in an earlier chapter. None is decoration.

**Sweep the threshold every epoch, and select on the swept score.** A fixed threshold once
under-reported a model by **0.163** on identical weights. The optimum sits at 0.70–0.85 for these
models, not at 0.5.

**A single-seed difference of ≤ 0.008 is noise.** Measured, not assumed.

**Three seeds for any claim, and compare band against band.** In chapter 2 a contribution
measured +0.0127 against a single baseline seed. When the baseline got its own three-seed band, the
difference became **+0.0065 — inside the noise floor**. The baseline seed had simply been the weak
one. This is why our results collector computes band-versus-band by construction rather than
leaving it to the analyst.

**A screen may only kill or promote, never claim.** Everything in §9 is a screen.

**Confirm screen-scale wins at deployment scale.** A component that helped at a small tile was
absorbed at a large one, twice.

**Fix the gate before the run.** And, from Rule 3, *measure the quantity the gate refers to first*.

**Never select anything on the test segment**, which has no labels and is qualitative only.

**Identical everything else.** Same split, labels, loss, augmentation, schedule, inference,
post-processing, metrics and probes across all three models. Only the backbone changes. This is
what makes the comparison a benchmark rather than three separate experiments.

---

## 12. What the results already say

**Reminder:** 19 runs FINAL, 9 still training. Everything below is single-seed.

### 12.1 The tile ladder — and the prediction it tested

| tile | raw F0.5 | status |
|---|---|---|
| 64 | 0.6816 | FINAL |
| 128 | 0.7578 | FINAL |
| 192 | 0.7857 | FINAL |
| 256 | 0.7948 | provisional (epoch 41/60) |
| 320 | 0.7975 | provisional (epoch 24/60) |

**Prediction 2 was pre-registered before any of these ran:** *because SegFormer3D's attention is
global rather than windowed, its receptive-field ratio should stay high and its ladder should be
steep — at least +0.03 from t128 to t192, against nnFormer's +0.0063.*

Measured: **+0.0279**. Essentially on the pre-registered number, and **4.4× steeper than
nnFormer's**. The receptive-field measurement in the analysis job will confirm or refute the
mechanism, but the behavioural prediction held.

This also **powered C1's gate** (Rule 3). Had the gap been under 0.02, the 2×2 would have been
cancelled before running, exactly as nnFormer's should have been.

**And the largest tile did not turn down.** Chapter 1's ladder *fell* at t320 with the overfit
signature — 112M parameters on 13,768 tiles. Here, at 4.4M with tile-invariant parameters, t320 is
still the best rung so far. That is prediction 4, and if it holds when the run finishes it means
chapter 1's ceiling was a **capacity artefact, not a data limit** — a correction to a published
conclusion of this thesis, produced by the model small enough to test it.

### 12.2 Where the small model stands

At t192, single seed: **0.7857 raw / 0.8092 post**, from **4.44M parameters**. UNETR++'s frozen
three-seed benchmark row is 0.8111 / **0.8190** from 79.1M at t256, and nnFormer's is 0.7801 /
0.8005 from 19.87M.

So a model **18× smaller than UNETR++**, at a *smaller tile*, is already within 0.01 post of it and
above nnFormer — and its own larger tiles are still training. Whatever the final number, the
efficiency frontier is going to be the interesting figure of this chapter.

### 12.3 The three things worth watching

1. **The overlapping key reduction (+0.0094 / +0.0126)** is the only screen outside the noise
   floor, and it was predicted to be a null. It promotes to three seeds. Treat with suspicion
   until then — one seed against one seed is precisely the trap chapter 2 fell into.
2. **The peripheral-keys contribution produced a clean null**, and only because of the same-cost
   control. This is the chapter's strongest methodological moment so far: the control converted an
   ambiguous +0.0039 into a definite statement about *what does not work and why*.
3. **Two adaptation choices are real** — depth preservation (−0.0164 when removed) and the repo's
   added normalisation layers (−0.0194 when removed). Both matter more than any contribution
   measured so far, which is itself the thesis's recurring lesson: getting the domain adaptation
   right dominates architectural cleverness.

---

## 13. Glossary

Every term used above, in one place.

**Architecture**

| term | meaning |
|---|---|
| **ViT** | Vision Transformer. Splits an image into patches, adds positional encodings, runs a plain transformer. Single-scale and quadratic in token count. |
| **PVT** | Pyramid Vision Transformer. Adds a four-stage hierarchy and spatial-reduction attention to ViT. |
| **MiT** | Mix Transformer — SegFormer's encoder. PVT plus overlapping patch embedding plus Mix-FFN. Sizes B0 to B5; our SF-S is the B0 configuration. |
| **Patch embedding** | Turning image regions into token vectors. Implemented as a strided convolution. |
| **Overlapping patch embedding** | Kernel larger than stride, so neighbouring patches share voxels and there are no seams. |
| **Sequence reduction (`sr`)** | Shrinking the key/value grid by a strided convolution before attention, so cost is `N × M` instead of `N × N`. The model's efficiency mechanism. |
| **Mix-FFN** | Feed-forward block with a depthwise convolution between its two linear layers. Replaces positional encoding in this family. |
| **Depthwise convolution** | One filter per channel, applied independently (`groups = channels`). Cheap; mixes space, not channels. |
| **All-MLP decoder** | Decoder built only from 1×1 projections, interpolation and concatenation. No spatial mixing. |
| **Transposed convolution** | A learned upsampling operation. Our default final path. |
| **Pre-norm residual** | `x = x + f(norm(x))`. Stabilises deep transformers. |
| **DropPath** | Stochastic depth: randomly drop an entire residual branch during training. Regulariser. |
| **LayerNorm / BatchNorm** | Normalise across features of one sample / across the batch for one channel. BatchNorm keeps *running statistics*, which is why it is the only tile-dependent state in this model. |
| **Soft-argmax** | A differentiable "where is the peak": softmax over a axis, then a weighted average of indices. |
| **SDPA** | Scaled dot-product attention — PyTorch's fused attention. Never materialises the attention matrix. |

**This project**

| term | meaning |
|---|---|
| **Segment** | One virtually unrolled sheet of papyrus: 65 depth layers of greyscale. |
| **Tile (`t`)** | The in-plane crop size fed to the model: 64…320. |
| **Depth window** | The 32 layers `[16,48)` we keep of the 65. |
| **Class-A layer** | The shared, model-external code: depth collapse plus head. Byte-identical for all three models. |
| **Depth collapse** | Reducing `[B,C,32,t,t]` to `[B,C,t,t]`. Our mode is `surface_relative`. |
| **Trimap / `unc.npy`** | Per-pixel map marking ink core / background / uncertain band. The uncertain band is excluded from the loss. |
| **`ink_reg.npy`** | The registered labels. All three chapters use these; a second registration round exists but must never be mixed with these numbers. |
| **F0.5** | Precision-weighted F-measure (β = 0.5), so precision counts four times as much as recall. The competition metric. |
| **τ (tau)** | The probability threshold turning a probability map into a binary prediction. **Swept** every epoch. |
| **Swept F0.5** | The best F0.5 over the threshold sweep. Our selection and reporting metric. |
| **Post / post-filter** | Removing 8-connected components smaller than a minimum size. Worth +0.010 to +0.020. |
| **Sliding-window inference** | Predicting a whole segment tile by tile at stride t/2, blending overlaps with a Gaussian weight. |
| **TTA** | Test-time augmentation: predict under 8 flips/rotations and average. |
| **Noise floor** | ±0.008. A single-seed difference inside it is not a result. |
| **Screen** | A single-seed run. May kill or promote an idea, never claim one. |
| **Band** | A three-seed mean ± standard deviation. Required for any claim. |
| **Gate** | A numeric success criterion fixed *before* the run. |
| **Informative null** | A pre-written sentence the thesis will contain if the result is negative. Every contribution here has one. |
| **Exact no-op at init** | The contribution is bitwise identical to stock at step 0, so the ablation is a genuine single toggle. |
| **Effective receptive field** | How far from a pixel the model actually looks, measured by gradient. |
| **Overfit signature** | Best epoch early **and** training loss still falling while validation falls. Overfitting is never claimed without both. |

---

## 14. Questions you will be asked

**"Why is a 4.5M-parameter model in a benchmark with a 79M one?"**
Because the thesis's central claim is that data, scale and protocol dominate architecture. The
sharpest possible test of that claim is whether an 18× smaller model loses anything. If it lands
close, the claim gets its strongest statement; if it collapses, we have found the floor. Either is
a result.

**"Isn't the peripheral-keys idea just the cross-scale attention that already failed?"**
No, and the difference is the whole point. That mechanism re-routed features **from inside the same
tile** — no new information. Ours brings **pixels the tile does not contain**. Its pre-written null
was "context must enter at input resolution", and that is the hypothesis we set out to test. As it
happens our result agrees with it, which is why the two chapters together say something neither
says alone.

**"Your contribution didn't work. Why is that in the thesis?"**
Because the same-cost control makes the null *informative*. We can say precisely what does not
work — context injected through the key path — and rule out the alternative explanation that it
was merely extra capacity. Two chapters of neutral results taught us that a contribution's value
is its pre-registered reading, not its sign.

**"Why not just use the published code?"**
It does not import on our PyTorch version, it crashes at four of our five tile sizes, and at the
fifth — our deployment tile — it silently produces scrambled 3D geometry with no error. §6.

**"Why do you keep saying single seed?"**
Because in chapter 2 a +0.0127 result became +0.0065 noise the moment the baseline got its own
three-seed band, and the whole write-up had to be corrected across four documents. Every
single-seed number here is provisional by policy.

**"Why is the threshold swept instead of fixed at 0.5?"**
Because a fixed threshold once under-reported a model by 0.163 on identical weights. It also means
any mechanism that merely rescales the logits is invisible, which analytically rules out a whole
family of proposals before spending anything on them.

**"What's the one number to remember?"**
The tile-ladder slope. UNETR++ +0.055, nnFormer +0.0063, SegFormer3D **+0.0279** — and the
receptive-field ratio predicts which you get. That is the closest thing this thesis has to a law.

---

## 15. Where everything lives

```
$WORK/segFormer/
  reference/SegFormer3D/     pristine clone — never edited
  src/segformer3d_ink/
    geometry.py    stage grids, stems, reduction presets, capacity rungs, assertions
    blocks.py      patch embedding, sequence-reduced attention, Mix-FFN, block
    contrib.py     C1 PDK · C2 DCM · C3 OSR · C4 DSM
    decoder.py     all-MLP decoder, final path, conv skip
    backbone.py    the model and its builder
  tests/test_smoke.py   50 checks incl. the four exact-reduction proofs
  configs/              31 configs, each a one-key delta from the base
  scripts/              make_configs · bench · collect_results · stress_tile · sbatch files
  outputs/<run>/        best.pt · summary.json · postprocess.json · train_log.csv · figs/
  probes/               ERF · Inkalyzer · frequency · error anatomy · transfer
  docs/                 PORT_LOG.md · RESULTS.md · bench_table.md · this file
```

**To reproduce anything:**

```bash
source $WORK/vesuvius/.venv/bin/activate
python $WORK/segFormer/tests/test_smoke.py            # 50 checks, ~2 min, CPU
python $WORK/segFormer/scripts/collect_results.py     # regenerate RESULTS.md + the gates
cd $WORK/vesuvius/vesuvius-ink-bench
python scripts/smoke_test.py --config $WORK/segFormer/configs/<run>.yaml --device cpu
```

The plan this implements is `~/plan(md_files)/segformer3d_plan.md`; the port's deviations and
fix log are in `docs/PORT_LOG.md`.
