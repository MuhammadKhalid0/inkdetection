# TUTORIAL — How this codebase is built and why

A guided tour of `vesuvius-ink-bench/` for the thesis author. Read top to bottom
once; after that use it as a map. This answers two questions:

1. **What is the structure of the code and why is it shaped this way?**
2. **Why are there new files instead of editing the UNETR++ repo — and did we use
   the repo at all?**

---

## Part 0 — The one design decision everything follows from

Your methodology (see `plan(md_files)/vesuvius_methodology_plan (1).md` §1 and the
implementation guide §0) makes one demand above all others:

> **The pipeline must be *identical* across every model and every ablation — same
> data, splits, seed, loss, inference, post-processing — so that a difference in
> F0.5 is caused by the *architecture*, not by the training setup.**

That is the whole game of a benchmark. If UNETR++ trained inside its repo, nnFormer
inside its repo, and SegFormer3D inside its repo, then when the scores differ you
could never say *why* — different augmentation, different LR schedule, different
sliding-window, different loss. The comparison would be worthless.

So the guiding rule of this codebase is:

> **One harness. The model is a plug-in. Everything around the model is written
> once and shared.**

Every structural choice below is a consequence of that rule.

---

## Part 1 — Point 2 first: why new files, and did we use the repo?

### We DID use the repo — we "lifted" its network, we did not rewrite it

We copied four files **verbatim** from your clone
(`/home/hpc/iwi5/iwi5401h/unetr_pp`) into `models/unetrpp/`:

| File | What it contains | Changed? |
|---|---|---|
| `layers.py` | `LayerNorm`, positional-encoding helpers | only touched the import lines |
| `dynunet_block.py` | `UnetResBlock`, `get_conv_layer`, `UnetOutBlock` (conv building blocks) | only the import lines |
| `model_components.py` | `UnetrPPEncoder`, `UnetrUpBlock` (the encoder/decoder) | only the import lines |
| `transformerblock.py` | **`EPA`** (Efficient Paired Attention) + `TransformerBlock` | only the import lines |

`EPA` — the actual contribution of UNETR++, the block your DA-EPA will later
extend — is **their exact code, unmodified**. So the answer to "did we utilize the
repo?" is: **yes, the entire network brain (encoder, decoder, EPA attention) is
theirs.** By line count, ~616 lines are lifted repo code.

The only edit inside those four files was rewriting imports from
`from unetr_pp.network_architecture.X import Y` to `from .X import Y`, so the files
work as a small self-contained package instead of needing the whole nnU-Net repo
installed. Nothing about the math changed.

### So why NOT just edit the repo in place?

Three concrete reasons — this is the part worth understanding.

**Reason 1 — The repos are training *frameworks*, and three incompatible ones.**
UNETR++ and nnFormer are built **on top of nnU-Net**. "Training" in those repos does
not mean "call `model(x)`." It means: register a `TaskXXX` dataset folder, run their
fingerprinting/planning step that writes a `plans.pkl`, let *their* trainer class own
the data loading, augmentation, sliding-window, loss, and checkpointing — all assuming
**3D medical volumes with 3D labels**. SegFormer3D is a completely different
`accelerate` + YAML + MONAI world. If you tried to run our task inside each repo you
would be writing three different data loaders, three augmentation configs, three loss
setups — i.e. you would rebuild the confound the benchmark is designed to remove.

**Reason 2 — Our task does not fit their assumption.** They expect **3D in → 3D mask
out**. We have **3D surface volume in → 2D ink mask out**, with only 8 training
segments and an ink-aware sampler. Their trainers cannot express that without deep
surgery.

**Reason 3 — Their inference machinery drags in heavy dependencies.** The repo's
`UNETR_PP` inherits `SegmentationNetwork`, which imports `batchgenerators`, nnU-Net
utilities, `pad_nd_image`, etc. We do not want any of that. We want a plain
`nn.Module` we can call.

This is exactly what your implementation guide §0 prescribes, in its own words:

> "**Do not run training inside the repos.** Lift only the network-definition code…
> wrap each network with a **common adapter**… drive everything from one YAML."

So the "new files" are not a rewrite of UNETR++. They are the **shared harness** the
methodology demands — the neutral test bench that every model plugs into. The model
itself is still theirs.

### The picture

```
        THEIR CODE (lifted, ~unchanged)            OUR CODE (the shared harness)
   ┌───────────────────────────────────┐   ┌──────────────────────────────────────┐
   │ models/unetrpp/                    │   │ models/unetr_pp_ink.py  (thin adapter)│
   │   layers.py                        │   │ models/shared.py  (stem/collapse/head)│
   │   dynunet_block.py                 │◄──┤ models/wrapper.py (unified forward)   │
   │   model_components.py (encoder/dec)│   │ data/  (prepare, tiling, dataset)     │
   │   transformerblock.py (EPA!)       │   │ engine/(train, infer, eval, losses)   │
   └───────────────────────────────────┘   └──────────────────────────────────────┘
        the "brain"                              everything around the brain
```

Later, nnFormer and SegFormer3D get the *same* treatment: lift their network files
into `models/nnformer/` and `models/segformer3d/`, wrap with the *same* adapter, and
they train through the *same* `engine/train.py`. That is what makes the benchmark fair.

---

## Part 2 — The structure, folder by folder

```
vesuvius-ink-bench/
├── configs/            # YAML: one file per experiment (A0 now, A0s…A2b later)
├── models/             # networks + the shared adapters
│   ├── unetrpp/        # LIFTED repo code + our unetr_pp_ink.py adapter
│   ├── shared.py       # TextureStem, DepthCollapse, Head  (Class-A shared layer)
│   └── wrapper.py      # build_model(cfg) -> UnifiedModel
├── data/               # everything about turning Vault files into training tiles
│   ├── segments.py     # the authoritative paths + train/val/test split
│   ├── prepare.py      # Vault .npy -> aligned arrays on $WORK
│   ├── tiling.py       # coordinates of 64×64 tiles -> manifests (CSV)
│   └── dataset.py      # torch Dataset + ink-aware sampler
├── engine/             # everything about the training/eval loop (model-agnostic)
│   ├── losses.py       # Dice+BCE (and Tversky+BCE for later)
│   ├── utils.py        # seed, LR schedule, checkpoint
│   ├── train.py        # the training loop
│   ├── infer.py        # sliding-window + Gaussian blend
│   └── eval.py         # F0.5 / Dice / precision / recall + threshold sweep
├── scripts/            # smoke_test.py (validate before GPU), slurm sbatch
└── outputs/            # checkpoints, logs, prediction maps
```

The split is deliberate and maps to a mental model:

- **`data/`** = "how do we get a labelled tile?" — knows about papyrus, depth
  layers, ink masks. Knows *nothing* about neural networks.
- **`models/`** = "what transforms a tile into a prediction?" — knows about tensors.
  Knows *nothing* about papyrus files.
- **`engine/`** = "how do we optimise and measure?" — knows about loops, losses,
  metrics. Treats the model as a black box `logits = model(x)`.

Because `engine/` never mentions "unetrpp", swapping in nnFormer later changes
**one line of config**, not the training code. That separation *is* the fairness
guarantee, written as folder boundaries.

---

## Part 3 — The single most important contract

Every model in this benchmark obeys one interface:

```python
forward(x: [B, 1, D, H, W])  ->  logits: [B, 1, H, W]
```

3D volume in, 2D ink map out. This is enforced by `models/wrapper.py`:

```python
class UnifiedModel(nn.Module):
    def forward(self, x):            # x: [B,1,D,H,W]
        if self.stem: x = self.stem(x)   # (A0s+) texture features — OFF for A0
        feat  = self.backbone(x)         # [B,C,D,H,W]  <-- the lifted UNETR++
        x2d   = self.collapse(feat)      # [B,C,H,W]    <-- depth -> 2D
        return self.head(x2d)            # [B,1,H,W]    <-- logits
```

Read those four lines carefully — they are the whole architecture of the harness:

1. **stem** — optional input enrichment (texture channels). A0 = off.
2. **backbone** — the lifted network. For A0 it is UNETR++’s encoder+decoder,
   returning a **feature volume**, not a mask (we bypass their 3D segmentation head).
3. **collapse** — squashes the depth axis `D` into 2D. A0 = mean. This is the "3D in,
   2D out" step your methodology §2.7 calls the depth→2D collapse.
4. **head** — a tiny 2D conv that outputs the 1-channel ink logit.

Every future ablation is a change to exactly one of these four slots, chosen by
config — never a fork of the training code. Example:
- **A0 → A0s**: turn `stem` on, switch `collapse` from `mean` to `learned`.
- **A1**: swap `EPA` inside the backbone for `DA_EPA`.
Nothing in `engine/` changes. That is the payoff of the structure.

---

## Part 4 — The one real piece of engineering in the model glue

`models/unetrpp/unetr_pp_ink.py` (our file, 150 lines) is the adapter between their
network and our harness. It exists to fix a genuine bug you would otherwise hit.

Their `UNETR_PP` was written for **128×128×64** medical volumes. Inside, the EPA
attention has these layers:

```python
self.E = self.F = nn.Linear(input_size, proj_size)         # input_size = token count
self.pos_embed  = nn.Parameter(torch.zeros(1, input_size, hidden_size))
```

`input_size` is the number of tokens at a stage = `D×H×W` at that resolution. In the
published class this number is **hardcoded to the medical sizes** (e.g. `32*32*32`).
Our input is **32×64×64**, so the correct token counts are different
(`[4096, 512, 64, 8]`). If we used their hardcoded values, the `Linear` and the
positional-embedding tensor would be the wrong shape and the model would crash — or
worse, silently mis-shape.

So `unetr_pp_ink.py` **recomputes** every stage resolution from the actual patch size
and feeds the correct `input_size`/`out_size` into the lifted encoder and decoders. It
also:
- makes the class a plain `nn.Module` (drops the nnU-Net `SegmentationNetwork` base),
- returns the **decoder feature volume** instead of a 3D mask (so our `collapse`+`head`
  can do the 2D readout),
- exposes a clean `build_unetrpp_ink(...)` used by the wrapper.

We verified the recomputation in `scripts/smoke_test.py`:
`stage_res [(16,16,16),(8,8,8),(4,4,4),(2,2,2)]`, `tokens [4096,512,64,8]`,
`bottleneck (2,2,2)`, and `(2,1,32,64,64) -> (2,1,64,64)`. This is why we could not
"just edit the repo": the repo assumes a fixed input size; our task changes it.

---

## Part 5 — Data path, end to end (concrete)

1. **`data/segments.py`** — the single source of truth for *where files are* and *the
   split*: Train = 8 segments, Val = `20241113090990`, Test = the `flatboi` segment
   (no labels → qualitative). Every other file imports the split from here, so it can
   never drift.

2. **`data/prepare.py`** — reads each raw `(65,H,W)` uint8 volume from **Vault**
   (slow, read once), the `inklabels.png` (→ binary), the `mask.png`, **crops all
   three to a common size** (the masks are 1 px smaller; one segment is 16 px off),
   and writes clean aligned arrays to **`$WORK`** (fast). It copies the volume
   *layer-by-layer* so even the 22000×10580 segment never blows up RAM.

3. **`data/tiling.py`** — slides a 64×64 window (stride 32) over each segment, keeps
   tiles that are inside the valid mask, records each tile’s **ink fraction**, and
   writes `train.csv` / `val.csv`. A tile is just a coordinate `(segment, y, x)` — we
   do **not** copy pixels here, keeping the manifest tiny.

4. **`data/dataset.py`** — at training time reads a tile’s pixels on demand from the
   memory-mapped `$WORK` arrays, applies the depth window (D=32 centred on layer 32)
   and augmentation, and hands `engine/` a `[1,D,64,64]` tensor + `[1,64,64]` label.
   Its `build_sampler` **oversamples ink-positive tiles** so the model doesn’t collapse
   to "predict no ink" (only ~10 % of pixels are ink).

---

## Part 6 — Training loop, in plain words (`engine/train.py`)

1. Build the dataset + ink-aware sampler; build the model from the YAML.
2. AdamW + warmup-cosine LR + mixed precision (all values from the config).
3. Each epoch: sample tiles, `logits = model(x)`, `loss = dice_bce(logits, y)`,
   backward, step.
4. **Validate the honest way**: run full **sliding-window inference on the entire
   validation segment** (`engine/infer.py`) and compute **F0.5 on the reconstructed
   segment** — not on random tiles. This matches how the metric is actually defined.
5. Keep the checkpoint with the best val F0.5; early-stop on patience.

The threshold `τ` is swept **on validation only** and the metric is computed **inside
the mask only** — both are fairness requirements from your methodology §2.7–§2.8.

---

## Part 7 — Why the smoke test exists

`scripts/smoke_test.py` builds the model, pushes a fake batch through, checks the
output is `[B,1,64,64]`, checks the loss is finite and gradients flow — all on CPU in
seconds. The point: **catch a shape/typo bug in 5 seconds on the login node instead of
waiting hours in the GPU queue only to crash on line 1.** Always run it after any model
change. "ALL CHECKS PASSED" = safe to submit.

---

## TL;DR

- **We used the UNETR++ repo fully**: its encoder, decoder, and the EPA attention block
  are lifted almost verbatim (~616 lines of their code). That is the model’s brain.
- **The new files are the shared harness**, not a rewrite. Your methodology *requires*
  one identical pipeline around every model; the repos are three incompatible nnU-Net /
  accelerate training frameworks that cannot provide that. So we lift the network and
  wrap it.
- **One contract** (`[B,1,D,H,W] -> [B,1,H,W]`) and **four swappable slots** (stem,
  backbone, collapse, head) turn every future ablation into a config flag instead of a
  code fork — which is exactly what keeps the benchmark honest.
```
