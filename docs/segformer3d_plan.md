# PLAN — Benchmark Model 3: SegFormer3D for Ink Detection (with contributions)

**Project:** MSc thesis, ink detection in carbonized papyrus (Scroll 5 / PHerc 0172, Unlost project)
**Written:** 2026-09-10 (rev. 1 — written after both prior chapters closed) · **Status:** ready for execution
**Prerequisite reading:** `UNTER_Plus_Plus_Final.md` (model-1 results + protocol), `nnformer_plan.md` (the structure this plan mirrors), `$WORK/nnFormer/docs/FINAL_SUMMARY.md` (model-2 verdicts), `plan.md` §0/§12 (ops rules + anti-goals), `vesuvius_methodology_plan (1).md` §5 (SegFormer3D source config), `data_facts.md`
**Role in the thesis:** third and final entry of the 3-model benchmark (UNETR++ ✅ → nnFormer ✅ → **SegFormer3D**), carrying **two SegFormer3D-native contributions** — PDK (peripheral downsampled keys — the chapter centerpiece, and the benchmark's only direct attack on its own master lever) and DCM (depth-coordinate Mix-FFN — the positional-encoding claim tested on an axis that violates its premise) — plus two gated cells that audit the model's remaining two published claims. The benchmark contract is: *identical data, splits, labels, recipe, inference, post-processing, metrics, probes; only the backbone changes; contributions are ablation-isolated on top.*

**The chapter's one-sentence spine:** *SegFormer3D makes three explicit design claims — the sequence-reduced K/V loses nothing, Mix-FFN removes the need for positional encoding, and an all-MLP decoder suffices because transformer ERFs are large. This chapter tests each claim in a domain that violates the assumption behind it, and adds one new mechanism that asks whether the thesis's dominant lever (context) can be bought inside the K/V budget the architecture already pays for.*

---

## 0. What the UNETR++ and nnFormer studies already decided for us (binding evidence)

Every design choice below is derived from a **measured** result of chapter 1 or chapter 2, not from the medical-segmentation literature. This section is the audit trail. `$WORK = /home/woody/iwi5/iwi5401h`.

| # | Measured evidence (chapter 1 = UNETR++, chapter 2 = nnFormer) | Consequence for SegFormer3D |
|---|---|---|
| E1 | **Context scale is the master lever**: UNETR++ lean ladder 0.639 → 0.719 → 0.774 → 0.804 raw at t64/128/192/256 (+0.080 / +0.055 / +0.029), saturating at t256; t320 turned **down** (0.8000 raw / 0.8032 post, −0.009/−0.015) with the overfit signature (best epoch 27, train loss 0.51→0.39 while val fell, 112.2M params on 13,768 tiles — note: chapter 1's report rounds this to "13.3k"; the manifest `train_reg_t320.csv` holds 13,769 lines including the header, so **13,768** is the on-disk truth and is what this chapter quotes) | Deploy at **t256**. But SegFormer3D is the **only** model in the benchmark that can run **t320 without the capacity confound** (its params are tile-invariant and it is 4.44M, not 112M) — so the t320 rung is *promoted from anti-goal to a pre-registered experiment* for this model only (§5 Phase S1b, §2.0 rule 4). Context is also the target of contribution **C1/PDK** (§2.1). |
| E2 | **Params↔tokens confound**: UNETR++ params grew 25M→79.1M→112.2M with tile (Linformer projections + PEs). nnFormer's were tile-invariant except its GV bottleneck bias table | SegFormer3D has **zero tile-dependent parameters** (measured: no relative-position tables, no positional encoding, no Linformer projections — verified on the prototype at t64…t320). Its ladder is the **only fully clean** one in the benchmark. State this as the benchmarking payoff and report measured params per rung anyway. It also makes **C1's same-cost control** and the **resolution-transfer study** (§5 Phase S4.7) possible at all. |
| E3 | Fixed loop + swept-τ selection is non-negotiable (fixed-τ under-reported `a2bs_reg` by 0.163 on the same weights; a broken cosine manufactured a fake +0.05 architecture win that a re-run deleted) | Identical training loop, 60-epoch cosine, warmup 3, patience 20, per-epoch τ-sweep, `select_on: sweep`. No exceptions. |
| E4 | **Noise floor: single-seed \|Δ\| ≤ 0.008 is noise.** Seed 42 was the weakest seed in **both** chapters — it deflated DA-EPA (chapter 1) and inflated SR-PB from a claimed +0.0127…+0.0152 to a measured **+0.0065** band-vs-band (chapter 2) | Same discipline, now with two precedents: seed-42 screens may only **kill or promote**, never claim. Every reported claim = 3-seed mean ± sd (42/43/44). **A contribution's band is compared only against the base's own band** — never against a single base seed. This is written into `scripts/collect_results.py` from day 1 (§9). |
| E5 | Adopted pipeline levers: registered labels (`ink_reg.npy`, +0.013 @t192), u1 ignore-band (`dice_bce_masked` over `unc.npy` trimaps, +0.012), z_flip OFF (+0.009/+0.011), 8-fold dihedral TTA (+0.0035), 8-connected component min-size filter (+0.010…+0.020 on every model) | SegFormer3D's config inherits **all of them from day 1**. The aim is to not regress below **0.8190 post** (UNETR++) / **0.8005 post** (nnFormer), so the model competes at full pipeline strength from the first reported run. |
| E6 | Class-A shared layer: texture stem **absorbed** at ≥ t192 (neutral; the frequency probe prices the 2–8 px band it engineers at ≈ 0); collapse = **`surface_relative` tile-level** (per-pixel + warp-stab measured neutral) | Stem stays OFF. Collapse = `surface_relative` tile-level, byte-identical shared code (`models/shared.py`). `DepthCollapse` computes its own ŝ from the feature volume, so it transfers to SegFormer3D with **zero changes** — which is also why the geometry family is inherited **free** here (§2.5, rejection R2). |
| E7 | **The geometry family is closed.** Unsupervised ŝ never learns (r ≈ 0.255/0.167/−0.210, MAE ≈ 6.1 layers; φ_rel stays at init); the prior still buys offset robustness; supervising it **hurts on both architectures** (UNETR++ aux-ŝ 0.7250 = −0.050; nnFormer `n_srpb_aux` 0.7607 < stock 0.7657 < SR-PB 0.7722) and is exactly zero-sum over δ∈[−8,8] (0.596 = 0.596). nnFormer's SR-PB was **neutral band-vs-band (+0.0065)**. The "segment 145701 peaks at z=42" premise was a **documentation error** (measured z=29; all train segments peak z28–30, val z30) | **Do not propose a third depth-frame accuracy contribution.** A fourth null in the same family would read, correctly, as running the same experiment three times. The cross-architecture geometry claim is completed here **inference-only** (`stress_zshift`, 9 val inferences per run, ~2 GPU-hours) on the inherited shared collapse prior — third architecture, honest scope (§5 Phase S4). The depth axis enters the *architecture* only through **C2/DCM**, which is a different mechanism (a coordinate inside the FFN feature stream, not a depth-frame readout) and whose primary readout is **not** δ=0 F0.5. |
| E8 | Loss engineering dead (Tversky/recall/precision arms ≤ noise, twice); every intensity normalization (photometric/CLAHE/z-score/detrend) harmful; raw `/255` correct | Loss = `dice_bce_masked` (u1). No photometric aug, no normalization variants, no loss reweighting. Anti-goals, inherited verbatim. |
| E9 | **Error anatomy** (`nnf_c1_t256_s42`, and chapter 1's twin): FN strata dead-label 0.9% / boundary ≤4px 13.7% / low-contrast 13.3% / **unexplained 72.1%**; by probability: near-miss (p .5–.75) 32.1%, weak (.25–.5) 23.8% + faint (.07–.25) 25.4% = **49.2% combined**, **blind (<.07) only 18.7%**. FP mass sits in **isolated islands** (nnFormer 190,433 px in 514 islands). ERF r90 ≈ **53% of tile at every UNETR++ rung** (35/68/105/134 px) | The benchmark readout is **per-stratum recall**, not only F0.5 — the pre-declared axis on which a small model may "win" at equal F0.5. The ERF/context pair is the direct motivation for **C1/PDK**; the FP-island statistic is the primary readout for the gated **C4/DSM** (§2.4). |
| E10 | **The ERF ratio is the one explanation that worked.** UNETR++ r90/tile ≈ 53% constant → steep ladder (+0.055 for t128→t192). nnFormer 51% → **45%, shrinking** (97 px @t192, 115 @t256) → **flat ladder (+0.0063)**. Thesis sentence: *"local windowed attention caps context utilisation; EPA's paired channel/spatial attention does not."* | This is now a **predictive law with two points**. SegFormer3D is the third and decisive point: it has *global* (sequence-reduced, not windowed) attention at every stage, so the law predicts a **high, non-shrinking r90/tile and therefore a steep ladder**. **This prediction is pre-registered in §0 and measured before C1 is run** — and it doubles as the power check that CSSA lacked (E11). |
| E11 | **CSSA failed underpowered by construction**: its gate asked it to close half of nnFormer's own t128→t192 gap, and that gap was **+0.0063**. The mechanism worked (ERF r50 63→75 px, +19%; recall .529→.601) but could not convert | **Rule: measure the model's own ladder before pre-registering any context gate** (§2.0 rule 3). C1/PDK's 2×2 is declared underpowered *before it is run* if SegFormer3D's stock t128→t192 gap is < 0.02, and in that case C1 is evaluated at deployment scale only. This is the single most important process fix this chapter can make. |
| E12 | **Capacity is a dead lever on this data**, twice: UNETR++ t320 (112.2M) overfit; nnFormer S 19.87M ≈ M 79.11M ≈ F 177.74M within 0.006 raw, with nnF-F showing the textbook overfit signature (best epoch 37, train loss −15% after the val peak). Dense stride-64 sampling mitigated it (best epoch 37→53) | SegFormer3D tests the claim from the **opposite direction**: it is the only model *small enough to scale up*. Measured presets **SF-S 4,437,568 / SF-M 19,966,880 / SF-L 79,353,344** were built to land on nnF-S (19.87M) and on UNETR++ t256 = nnF-M (79.1M), giving an **18× parameter range at fixed architecture** and a **capacity × context grid with no confound** (§5 Phase S2). If capacity is inert across 4.4M→79.4M *and* across three tiles, that is the strongest form the thesis's central claim can take. |
| E13 | **Ops, measured**: A100 wall times t128 ≈ 3 h, t192 ≈ 4.5–5.2 h, t256 (nnF-S, dense) ≈ 7.3–8.2 h, e90 ≈ 10.8 h; A40 ≈ 1.45× A100. The account hits `AssocGrpGRES` on a100 intermittently — **A40 is the reliable fallback**. A lost `#SBATCH --array` directive once silently ran only task 0 | Same ops rules (§3.4, §8). Budget from measured numbers, not estimates. **Always verify the `--array` directive survives any sed edit**, and read scheduler reason fields only after a full cycle. |

**Predictions to pre-register (write in the lab notebook before any GPU run):**

1. **Pipeline ≫ architecture (third replication).** The capacity-matched stock SegFormer3D lands **within ±0.02 of 0.8190 post** despite having 4.44M parameters against UNETR++'s 79.1M — i.e. an **18× parameter reduction costs less than the noise-to-pipeline gap**. (Wider band than nnFormer's ±0.01 because this is a 4.4M model; the point is the *ratio*, not the tie.)
2. **The ERF law holds (E10).** SegFormer3D's r90/tile ≥ 50% and **does not shrink** with tile (unlike nnFormer's 51%→45%), because its attention is global-with-reduced-keys rather than windowed. **Therefore its tile ladder is steep** — pre-registered as ≥ +0.03 for t128→t192, against nnFormer's +0.0063. *This is the chapter's falsifiable headline prediction and it is made before the ladder is run.*
3. **Capacity is inert across an 18× range** (SF-S 4.44M vs SF-M 19.97M within ±0.008 at each tile) and **shows no capacity × context interaction**; SF-L (79.35M) at t256 shows the overfit signature at baseline density.
4. **t320 without the capacity confound**: SF-S at t320 lands within ±0.008 of t256 with **no** overfit signature — i.e. UNETR++'s t320 turn was a **capacity artifact, not a data limit**. (If instead t320 also turns down at 4.44M with a dense manifest, the saturation is a genuine information/sampling limit — the stronger and more interesting result.)
5. **C1/PDK**: closes ≥ half of SegFormer3D's own stock t128→t192 gap at t128 — *conditional on that gap being ≥ 0.02*, checked first (E11). At t256 ≈ neutral (saturation there was data/capacity-bound on both prior models).
6. **C2/DCM**: **neutral at δ=0 for both frames** (\|Δ\| ≤ 0.008) — the expected outcome, and a positive cross-domain confirmation of Xie et al.'s positional-encoding-free claim. The falsifiable part is the **sign asymmetry on the stress curve**: the *absolute* frame should lose ≥ 0.02 at \|δ\| ≥ 6 (it hard-codes the band index) while the *surface* frame should not.
7. **C3/OSR and C4/DSM**: both neutral (\|Δ\| ≤ 0.008). After two chapters in which *every* internal mechanism landed inside the noise floor, the prior for a kernel-shape change and a decoder conv is a null; both are run for the **mechanism measurement** (frequency band-stop shift; post-filter-gain substitution), not for a win.
8. **2.5D control**: the depth-as-channels 2D MiT lands **within 0.02 of its 3D twin** at 60–100× less compute — i.e. 3D attention over 32 depth layers buys little inside this family.

---
## 1. Medical → ink adaptation map (every assumption, explicitly)

SegFormer3D [Perera24] is a hierarchical 4-stage MiT-style encoder with an all-MLP decoder, built for multi-modal 3D medical segmentation: 4-channel 128³ volumes in → 3-class mask out, **4.51M params / 17.5 GFLOPs as published** (the repo's own profiler notebook reports 12.8 GMac at input `(4,128,128,128)`; our re-measurement of the released file gives **4,525,251** params at `in_channels=4, num_classes=3` and **25.38 GFLOPs = 12.69 GMACs** — the paper's 17.5 GFLOPs figure is not reproducible from either the code or the notebook, and the thesis must quote **measured** numbers at the ink config, per the methodology's numbers caveat). Each medical assumption and its ink replacement:

| Medical assumption (source) | Ink reality | Adaptation |
|---|---|---|
| 4-modality MRI volume in → **3D** multi-class softmax mask out | Single-channel 3D surface volume in → **2D** binary ink map out | Backbone returns a full-resolution **feature volume** `[B,16,32,t,t]`; the shared Class-A `DepthCollapse(surface_relative)` + shared `Head` (3×3→GELU→1×1→1 logit) produce `[B,1,t,t]`. Identical wrapper contract as UNETR++ and nnFormer (`models/wrapper.py`), so `out_channels = 16 = model.feature_size`. |
| **Isotropic 128³ anatomy**; the stem strides `[4,2,2,2]` on all three axes | Input is **anisotropic `32 × t × t`** and the signal is a **thin band at z27–33** (Inkalyzer: every one of 20 models measured across both chapters peaks z29–34, ≈0 at window edges; mass within ±2 layers 0.444–0.504) | **Depth-preserving stem** is the default: stage-1 `Conv3d(1, 32, kernel=(3,7,7), stride=(1,4,4), padding=(1,3,3))`, then the stock `(3,3,3)/s2/p1` for stages 2–4 → grids **(32,t/4,t/4) → (16,t/8,t/8) → (8,t/16,t/16) → (4,t/32,t/32)**. The stock isotropic stem crushes depth **32→8→4→2→1**, at which point the stage-4 Mix-FFN 3×3×3 sees only zero padding along depth and stage-4 attention is purely in-plane. Both are built; `stem: iso` is ablation cell **A1** (§6.1), not the default. This also protects **C2/DCM**: a depth-coordinate mechanism is meaningless if depth is 1 token by stage 4. |
| Sequence reduction assumes "medical imaging data frequently exhibits repetitive structures, suggesting that it can be compressed" — applied **isotropically** (`sr_ratios [4,2,1,1]`, realised in code as `Conv3d(C,C,k=s=r)`, i.e. reduction by **r³ = 64×**, not r) | The informative part of the 32-layer axis is ~6 layers; in-plane, strokes are 30–50 px and letters 300–1300 px | Per-axis sr ratios, config-driven: **`(1,4,4)/(1,2,2)/1/1` is the default** (depth-preserving in the keys too), with `(4,4,4)/(2,2,2)` (stock-isotropic) and `(2,4,4)/(2,2,2)` as ablation cell **A2**. Measured cost at t256: 497.5 / 274.1 / 342.8 GFLOPs respectively — the preset choice is a **1.8× compute decision at ±131k params**, so A2 is run early and its winner becomes the deployment config. |
| Dataset-specific shapes; the released code assumes **cubic token counts** (`cube_root(N)` + `reshape(B,C,n,n,n)` in `SelfAttention`, `DWConv`, and the stage loop) | `t ∈ {64,128,192,256,320}`, D=32 | **Every `cube_root` site is replaced by the explicit per-stage `(D,H,W)` grid threaded through each block** (the NVlabs `blk(x,H,W)` pattern, in 3D), computed at build time and asserted at forward time. This is not optional: see §3.4 items 1–2 — the released file **crashes at t64/t128/t192/t320 and silently produces scrambled geometry at t256**. |
| Deep supervision: not used by SegFormer3D (it has none) — but nnU-Net-style DS is used by both competitors | 2D labels; DS was removed from UNETR++ as structurally incompatible and replaced by a full-res conv skip; kept OFF for nnFormer | **DS stays OFF for every benchmark model** (protocol decision already on the record). SegFormer3D natively has no DS, so for once the fair setting is also the native one — state this in the methods as the one place where the fairness constraint costs nothing. |
| Repo recipe: AdamW lr 1e-4 **wd 0.01**, 20-epoch warm-up, `cosine_annealing_wr` (T₀=400), **800 epochs**, batch 2, Dice loss, EMA off, grad-clip off, sliding-window `roi [128,128,128]`. (The *paper* instead states warm-up 4e-6→4e-4 then PolyLR, "AdamW with a learning rate of 3e-5", Dice-CE, batch 4, 1000 epochs — the paper and the released code disagree on **every** axis, and the maintainers state the paper's ACDC/Synapse runs were done inside the *nnFormer* codebase with code they no longer have) | Benchmark protocol, measured-dead loss/aug variants | AdamW **1e-4, wd 3e-5**, warmup 3 + cosine to **60 epochs**, `dice_bce_masked` (u1 trimaps), batch 4, bf16, grad-clip 1.0, aug hflip/vflip/rot90 + z_shift ±2, **z_flip OFF**. Deviation from both published recipes is stated in the thesis and justified by protocol identity (E3). One LR screen at 5e-5 (**A12**) and one weight-decay screen at 1e-2 (**A13**, the repo's own value — cheap, and it is the one hyper-parameter where the source differs by 300×). |
| Pretrained checkpoints: none used ("without the use of complicated pretraining efforts"); released BraTS weights exist | UNETR++ and nnFormer both ran from scratch | **From scratch** (benchmark-equalized initialization). SegFormer3D is, however, the **only** model in the benchmark whose released medical weights are *shape-compatible* with a legal ink config (the isotropic stem with `in_channels` re-inflated), so a single **gated, optional** init-arm cell exists (**A14**) — reported as a curiosity, never in the headline. |
| Intensity: per-modality z-score of non-zero voxels | Raw `/255` is the measured optimum (E8) | Unchanged `/255`; depth window `[16,48)`, `center_layer 32`, `n_layers 65`. |
| `BatchNorm3d` ×9 (8 after the Mix-FFN depthwise conv, 1 in `linear_fuse`) — added by the repo, **absent in the NVlabs reference** | Train/eval statistics are estimated on tiny grids (stage 4 at t64 is 4 voxels per sample) and, critically, **BN running statistics are the only tile-dependent state in the whole model** | Kept by default (fidelity to the released architecture), but they are the one thing that could break the tile-invariance claim, so: **A3** ablates them (NVlabs-exact, −10,240 params) and the **resolution-transfer study** (§5 Phase S4.7) reports the transfer curve **with and without** them. Under `torch.autocast(bf16)` batch-norm computes in fp32 and is safe; the model is **never** `.to(bfloat16)`-cast (that would store running stats in 8-bit mantissa). |

### 1.1 What stays *exactly* the SegFormer3D identity (the architecture under test)

1. **Overlapping convolutional patch embedding** (`Conv3d` + flatten + LayerNorm per stage) — kept; only the per-axis kernel/stride/padding change, and the isotropic original is an ablation cell.
2. **Sequence-reduced ("efficient") self-attention**: Q at the full stage grid, K/V produced by `Conv3d(C,C,k=s=r)` + LayerNorm, `sr_ratios [4,2,1,1]` — kept as the mechanism; only the per-axis ratio changes. **This is the mechanism C1/PDK extends and C3/OSR audits.**
3. **Mix-FFN** (`Linear(C→4C) → DWConv3d(3×3×3, groups=4C) → GELU → Linear(4C→C)`) with **no positional encoding anywhere in the model** — kept; **this is the mechanism C2/DCM audits.**
4. **Hierarchical 4-stage pyramid** with `depths [2,2,2,2]`, `num_heads [1,2,5,8]`, `mlp_ratios [4,4,4,4]`, head_dim 32 — kept; only widths/depths change across the capacity rungs.
5. **All-MLP decoder**: per-stage `Linear(C_i→256)` → trilinear resize to the stage-1 grid → concat → `Conv3d(1024→256,1×1×1)`+BN+ReLU → prediction conv — kept; **this is the mechanism C4/DSM audits.** The final path to full resolution is the one place parity with the other two ports is imposed (§3.1).

---
## 2. Contributions — an audit of three published claims, plus one new mechanism

Two chapters have now measured **every** architecture-internal mechanism anyone proposed, and every one landed inside the ±0.008 noise floor at deployment scale: DA-EPA −0.004 · texture stem +0.000 @t192 · surface-relative collapse −0.009 @t192 but **+0.008 raw / ±0.000 post @t256** · skip-attention removal **−0.0002** · SR-PB **+0.0065** band-vs-band · CSSA +0.0037 · every loss shape ≤ 0.008 · aux-ŝ **−0.050**. A third chapter that proposes four more "improvements" and reports four more nulls adds nothing. This section therefore states **the rules a contribution must satisfy to be worth GPU time here**, and then proposes contributions that satisfy them.

### 2.0 What two chapters imply for any contribution (four binding rules)

**Rule 1 — information, not routing.** A mechanism that *re-routes information the model already has* is measured dead twice: CSSA (pre-written null: *"letterform context must enter at input resolution; it cannot be retrieved from coarse features after the fact"*) and the texture stem (absorbed at ≥ t192; the frequency probe prices the 2–8 px band it engineers at ≈ 0). DA-EPA's gate ended at **0.1010** from an init of 0.1 — *the depth gate stayed at its safe init because it had nothing to add.* Only two things have never been tried: **changing what information exists in the forward pass**, and **converting the one measured lever into something cheaper or deployable**. C1 is the first; the resolution-transfer study (§5 S4.7) and the efficiency frontier (§6.7) are the second.

**Rule 2 — the τ-sweep retires an entire family, analytically and for free.** τ is swept over {0.10…0.90} every epoch and both selection and reporting use the swept optimum. Therefore **any mechanism whose effect is a monotone rescaling of the logits is invisible to this metric by construction**: calibration heads, learned temperature, logit priors, prior injection, and loss reweighting (already an anti-goal) cannot move swept F0.5 unless they change the *ranking* of pixels. This is worth stating in the thesis because it explains, without a single GPU hour, why the error anatomy's tempting "32% near-miss / 49% faint" headroom is **not** addressable by a confidence trick — the near-miss mass is near-miss *at the swept optimum*. Any proposal aimed at "improving confidence on faint ink" is rejected on this ground alone (§2.5, R4).

**Rule 3 — power the gate before you run it.** CSSA did not merely fail, it failed **underpowered by construction**: its gate asked it to close half of nnFormer's own t128→t192 gap, and that gap was **+0.0063** (UNETR++'s was +0.055). The fix, binding for this chapter: **measure SegFormer3D's own tile ladder and ERF first (Phase S1b), and only then decide whether the context experiment is worth running and at what threshold.** A gate whose target quantity has not been measured is not pre-registration, it is guessing.

**Rule 4 — the model's smallness is itself an experimental instrument.** UNETR++'s ladder ended in a capacity confound (25M→79.1M→112.2M with tile) and its t320 turn is therefore uninterpretable: data limit or overfit? nnFormer had tile-invariant params but could not use context at all. SegFormer3D at **4,437,568 parameters with zero tile-dependent state** can run the rungs neither predecessor could run cleanly. Experiments that were anti-goals for the big models (t320; capacity ladders) are **promoted to pre-registered experiments here**, and that promotion is itself a contribution of the chapter (§5 Phase S1b/S2, hypotheses H2–H4).

---

### 2.1 C1 — Peripheral Downsampled Keys (PDK): **ADOPT — the chapter centerpiece**

**Verdict: the only mechanism in the entire benchmark that can buy context without buying queries, implementable on SegFormer3D and on neither predecessor. It is the chapter's headline experiment and the thesis's only direct attack on its own dominant lever.**

**Motivation (measured).** Context scale is the master lever (E1: +0.16 across the UNETR++ ladder; left-word recall 0.10 → 0.31 → 0.51 → ~0.55 as the tile grows) and the residual error is under-confidence on faint ink (E9: 32% near-miss, 49% faint, only 18.7% blind). A faint stroke inside a small tile is a locally uniform texture with **no letterform available to disambiguate it**; letters are 300–1300 px wide, so even a t256 tile often contains less than one letter of context around a given stroke. Every previous attempt to supply that context did so by **growing the tile**, which simultaneously grows queries, decoder cost, activation memory, epoch composition, optimizer steps and (for UNETR++) parameters — the four confounds the ladder chapter had to state every time.

**Why this architecture and no other.** In SegFormer3D, Q is computed on the full stage grid while K/V are produced by `Conv3d(C,C,k=r,s=r)` on that grid: **the key set is already a low-resolution learned summary, and its resolution is decoupled from the query resolution.** Nothing in the architecture requires the summary to summarise *the same spatial extent as the queries*. UNETR++ cannot do this (its Linformer projection is a learned map on the token axis of K/V drawn from the same tile, so foreign tokens have no defined slot); nnFormer cannot (its relative-position-bias table is indexed by intra-window grid offsets, undefined across extents). The **absence of positional encoding** is what makes foreign-extent tokens legal here at all — the same property the source paper advertises for resolution robustness.

> **PDK = a second, γ-gated attention branch whose keys and values are computed from a 2× wider, 2× downsampled *halo* crop centred on the tile, produced directly at the stage's reduced K/V grid by a small context stem, and combined through its own softmax.**
> `out = SDPA(q, k, v) + γ_h ⊙ SDPA(q, k_p, v_p)`, with `γ_h ∈ R^{heads}` zero-initialised.

**Mechanism spec (exact).**

1. **Data carrier — the halo rides as input channel 1.** With `data.halo.enabled: true`, `InkDataset.__getitem__` reads a second window `vol[z0:z0+32, y−t/2 : y+3t/2, x−t/2 : x+3t/2]` (clamped to the segment, zero-filled outside), average-pools it **2× in-plane**, and stacks it as channel 1, so `x` becomes `[B,2,32,t,t]` — **the same tensor rank and the same voxel count as before**. Consequences: `engine/train.py`, `build_sampler`, the loss, the collapse, the head and every manifest are untouched. The geometric augmentations (hflip/vflip/rot90) and the z-window are applied **identically to both channels** (the halo must stay geometrically consistent with the tile); the label and trimap are unchanged. A `halo_mask_frac` field records how much of the halo fell outside the segment.
2. **Backbone routing.** `x[:, :1]` enters the stock path, so the stage-1 patch embed keeps `in_ch=1` and **its weights are shape-identical to the stock model**. `x[:, 1:]` enters a per-stage **context stem** `Conv3d(1, C_i, kernel=stride=(d_i, s_i, s_i))` that produces peripheral tokens **directly at (or below) the stage's reduced K/V grid** — the halo is *never* materialised at full resolution. This is essential: a 2×-downsampled 2t halo has exactly the tile's voxel count, so encoding it through the normal path would **double the encoder** and the "constant full-resolution cost" claim would be false.
3. **Attention.** In each block of the enabled stages, the peripheral tokens are projected by the block's **existing** `key_value` Linear (dims match by construction → **no new projection parameters**), then attended in a **separate softmax branch** scaled by `γ_h`. *Concatenating* the peripheral keys into the stock softmax would renormalise the stock attention weights and is therefore **not** an exact no-op — this is the single most important implementation detail in the contribution and it is unit-tested (§3.4 item 3).
4. **Default stages `pdk_stages: [2,3,4]`.** At t256 with the depth-preserving stem, stage-1 attention is **276.56 of 497.5 GFLOPs (57.1%)** while stage 2 is **35.30 GFLOPs (8.0%)** — peripheral keys are cheap exactly where letterform-scale semantics live. Sizing the peripheral grid to `M_p ≈ M` adds ≈ +25% to stage-2 attention ≈ **+2% of the model total**. Stage 1 is available (`pdk_stages: [1,2,3,4]`) but costs ≈ +55% of total FLOPs and is **not** the default.
5. **Missing-channel rule.** If `x.shape[1] == 1` the peripheral branch is skipped entirely (equivalent to γ=0). This keeps the **shared** `scripts/smoke_test.py` (which forwards `randn(2,1,D,T,T)`), `interp_erf.py` and `inkalyzer_run.py` working unmodified — and gives a free *"halo ablated at inference time"* cell on a halo-trained checkpoint.

**Exact no-op at init.** `γ_h = nn.Parameter(torch.zeros(num_heads))` per block ⇒ the second SDPA term is multiplied by exactly 0.0 ⇒ the block output, and hence the entire network, is **bitwise identical** to stock. Unit tests (mirroring `nnFormer/src/nnformer_ink/contrib.py`): `test_pdk_exact_reduction` asserts `max|Δ| == 0.0`; `test_pdk_live_when_trained` asserts `max|Δ| > 1e-6` after perturbing γ; `test_pdk_backward` asserts every new parameter receives a finite gradient.

**Parameters added.** 3 context stems (stage 2: `Conv3d(1,64,k=s=(2,4,4))` = 2,048 + 64; stage 3, stage 4 similarly) + γ per block: **< 20k on SF-S (4,437,568) = < 0.5%**. Printed by the smoke test, as the house rule requires. **A same-cost control is mandatory** (§2.1 gate 4) or a reviewer will read the result as capacity — even at 0.5%.

**Config flags.** `model.pdk: false` (default), `model.pdk_stages: [2,3,4]`, `model.pdk_grid: [8,8,8]`, `data.halo: {enabled: false, factor: 2}`. Everything else byte-identical to the deployment config.

**Pre-registered gate, in this order.**
0. **Power check (Rule 3, mandatory).** Phase S1b measures the stock ladder first. Let `G = raw F0.5(sf_base_t192_s42) − raw F0.5(sf_base_t128_s42)`. **If `G < 0.02`, the 2×2 is declared underpowered before it is run**, no 2×2 cells are spent, and PDK is evaluated at deployment scale only (gate 3) with the ERF readout as its mechanism evidence.
1. **Same-cost control (mandatory, runs with the 2×2).** `sf_pdkctl_t128_s42` = PDK with the halo channel replaced by **a 2× downsampled copy of the tile itself** (same tensor shapes, same parameters, same FLOPs, **no new information**). This isolates "new pixels" from "extra keys + extra parameters" and is the answer to the capacity objection. *This control is what makes C1 a claim about information rather than about width.*
2. **Payoff 2×2**, all cells at batch 4 (identical 1000 steps/epoch): {stock, PDK} × {t128, t192}. **Promote to 3 seeds only if PDK@t128 ≥ stock@t128 + 0.02 *and* ≥ 50% of `G` *and* PDK@t128 − PDKctl@t128 ≥ 0.008.**
3. **Deployment gate.** PDK enters `sf_c1_t256` only if `sf_pdk_t256_s42` ≥ +0.008 over `sf_base_t256_s42` (the stem lesson: a screen-scale win must be confirmed at deployment scale). The stock twin is always kept for attribution.

**Prediction.** If prediction 2 holds (SegFormer3D is a global-attention context maximaliser with r90/tile ≥ 50%), the stock ladder is steep (`G ≥ 0.03`) and PDK@t128 closes ≥ half of it; `γ_h` trains away from 0 (**plot it regardless — the γ_da lesson**); the ERF measured **on the halo channel** becomes non-zero, which is the mechanism figure. At t256, ≈ neutral.

**Informative null (pre-written).** *"Context that is not present at the query resolution cannot be injected through the key path. Combined with nnFormer's CSSA null — coarse features already inside the tile do not help either — the tile ladder's +0.16 is not about information content at all: it is about the resolution of the queries and of the decoder, and about what a larger tile does to the epoch's sampling distribution."* That is a strictly stronger statement than either previous chapter can make alone, and it closes the thesis's largest open question about its own dominant lever. **Either sign is a headline.**

**Redundancy check.** *Not CSSA*: CSSA re-routed coarse features **from inside the same tile**, adding no information, and failed (+0.0037 / +0.0020, 32% of a 0.0063 gap); PDK adds voxels the tile does not contain — CSSA's pre-written null is the **hypothesis PDK tests**, not a verdict against it. *Not tile growth*: query count, decoder cost, final-expand cost and activation memory are unchanged (stage-1 N stays 32,768 at t128 rather than 131,072 at t256). *Not word-box sampling* (measured −0.04…−0.17): the tile grid, the sampler and the blind sliding-window inference are untouched, and the halo is a deterministic function of the tile coordinate at **both** train and inference time — so there is no train/deployment distribution mismatch, which is the exact mechanism that killed word-box. *Not deep supervision, not loss shaping, not photometric*: nothing about pixel values changes (`/255` only).

**Touchpoints.** `$WORK/SegFormer3D/src/segformer3d_ink/{attention.py, backbone.py, contrib.py}`, `tests/test_smoke.py`, `configs/`; `$BENCH_DIR/models/wrapper.py` (the new `elif name == "segformer3d":` branch). **Shared-engine changes — the largest footprint in this plan, and the one real engineering risk:** `data/dataset.py` (≈ 40 lines: halo read + pool + stack, entirely inside `if self.halo:`) and `engine/infer.py` (≈ 25 lines: build the halo for each sliding-window tile). `engine/train.py` **unchanged**. Both changes are default-off and must be **proven** byte-identical when off (§3.4 item 8).

**Cost.** 2 screens (`sf_pdk_t128_s42` ≈ 3 h, `sf_pdk_t192_s42` ≈ 5 h) + 1 same-cost control (≈ 3 h) + 1 gated t256 screen (≈ 8 h) + 8 promotion runs **only if** the gate passes. The stock 2×2 cells come free from the tile ladder (S1b).

**Risks and pre-planned responses.** (a) *Halo tiles at segment borders are partly zero* → record `halo_mask_frac` per tile and re-check the gate on interior-only tiles. (b) *The peripheral stem is a second input distribution and may destabilise training* → restrict `pdk_stages` to `[3,4]` (pre-decided fallback). (c) *Engine surgery must be provably inert when off* → §3.4 item 8. (d) *The halo doubles per-tile I/O* and these runs are expected to be **loader-bound** (§5, budget note), so measure the loader before blaming the model.

---

### 2.2 C2 — Depth-Coordinate Mix-FFN (DCM): **ADOPT — the positional claim, tested where its premise fails**

**Verdict: the cheapest contribution in the chapter, the only one that tests the source paper's own headline claim in a domain that violates its premise, and the only one whose learned parameter can be read directly against an independent probe.**

**Motivation (measured).** SegFormer's positional-encoding-free design rests on translation invariance — *"we argue that positional encoding is actually not necessary for semantic segmentation"* — with position leaked **only** by the zero padding of the Mix-FFN 3×3×3 depthwise convolution (Xie et al. §3.1, after Islam et al.). In the ink volume the two in-plane axes **are** translation-invariant (tiles are cut on a blind grid; nothing distinguishes a tile's position), but **the depth axis is not**: the ink band sits at a physically determined absolute index (E7/D1: z27–33 within the `[16,48)` window), and all 20 models measured across both chapters lock onto it. The ink volume is therefore **precisely the case the PE-free argument does not cover**: one axis where identity is physically privileged, two where it is not. No published work has tested the claim in that setting.

> **DCM = a zero-gated, depth-indexed bias added to the Mix-FFN hidden state immediately after the depthwise convolution — i.e. injected at exactly the point that constitutes the architecture's entire positional apparatus.**
> `h ← h + γ ⊙ P_i[·]`, `γ ∈ R^{4C_i}` zero-initialised, tables shared across the two blocks of a stage.
> `frame: absolute` → `P_i ∈ R^{D_i × 4C_i}` indexed by the token's depth index `d`.
> `frame: surface` → `P_i ∈ R^{(2D_i−1) × 4C_i}` sampled at `d − ŝ` by linear interpolation, with `ŝ` a zero-init soft-argmax over a mean+std depth descriptor, computed in an **fp32 island** with `nan_to_num` (this exact guard was load-bearing in both previous ports).

The token's depth index is available for free because the port threads the true `(D,H,W)` grid through every block — which it must do anyway, since the reference's `cube_root(N)` is unusable (§3.4).

**Exact no-op at init.** `γ = 0` ⇒ `h` unchanged ⇒ bitwise-identical network (`test_dcm_exact_reduction`, `max|Δ| == 0.0`). **Note the deliberate asymmetry:** `P` is initialised `trunc_normal_(std=0.02)` like every other table in the model, **not** zero — zero-initialising both `γ` and `P` would leave the branch dead at both ends with no gradient signal, which is the DA-EPA safe-init failure mode (γ_da ended at 0.1010 from init 0.1 because it had nothing to add). `γ` alone carries the exactness.

**Parameters added.** Absolute frame, shared per stage, depth-preserving stem: `32·128 + 16·256 + 8·640 + 4·1024 = 4,096 + 4,096 + 5,120 + 4,096 = 17,408`, plus γ (one vector of 4C_i per stage: 128+256+640+1024 = **2,048**) ⇒ **19,456 on 4,437,568 = 0.44%**. Surface frame ≈ 2× the table. Measured and printed by the smoke test.

**Config flags.** `model.dcm: false` (default), `model.dcm_frame: absolute | surface`, `model.dcm_stages: [1,2,3,4]`.

**Pre-registered gate — δ=0 F0.5 is explicitly NOT the primary readout.**
1. **The learned table, read against an independent probe.** Plot `‖P_i[d]‖` against `d`, and against the Inkalyzer IBA depth profile measured on the same checkpoint. **Pre-registered, falsifiable:** the absolute table's learned profile peaks inside z27–33. A contribution whose learned parameter can be checked against an independent measurement is rare in this project; it is reported either way (the γ_da lesson).
2. **Frozen-τ stress curve, band vs band** (`stress_zshift`, δ ∈ [−8,+8]). **Pre-registered sign asymmetry:** the *absolute* frame should **degrade** the tails (it hard-codes a band index that the induced offset invalidates) while the *surface* frame should not. If that asymmetry appears, it is a clean mechanistic result **independent of any F0.5 movement**, and it is the first time in this thesis that a depth-frame mechanism makes a *differential* prediction rather than an amount prediction.
3. Entry into `sf_c1_t256` only at ≥ +0.008 on a t256 screen, then 3 seeds, stock twin reported.

**Prediction.** δ=0 within noise for both frames (prediction 6); absolute-frame table peaks in z27–33; absolute frame loses ≥ 0.02 at |δ| ≥ 6 relative to stock while the surface frame does not.

**Informative null (pre-written).** *"A network with no positional encoding of any kind still concentrates its depth attribution on z32–33 with ≈0 mass at the window edges, and supplying an explicit depth coordinate changes nothing. Depth identity in this data is recoverable from content — from the intensity and texture profile itself — and need not be supplied as a coordinate. Xie et al.'s positional-encoding-free claim survives translation into a volume with a physically privileged axis."* This is a publishable cross-domain sentence and it is the **expected** outcome.

**Redundancy check.** *Not SR-PB* (nnFormer C1): SR-PB was an additive **key-side bias on attention logits**; DCM writes into the **FFN feature stream** and never touches attention. *Not `n_abs` / the learned-absolute collapse*: those are **readout** weights applied once over the final feature volume; DCM asks whether depth identity is needed *inside* the representation, where attention can still use it. *Not UNETR++'s `pe_mode: 2d`*: that tiles an in-plane positional encoding across depth — it supplies exactly the coordinate DCM does **not** supply and withholds exactly the one DCM adds, which makes the pair a clean cross-model statement. *Not aux-ŝ supervision* (measured to hurt on both models): nothing here is supervised, and no loss term is added. *Not a fourth geometry-accuracy run* (E7): its pre-registered readout is a **differential** stress asymmetry, not a δ=0 gain.

**Touchpoints.** `src/segformer3d_ink/{blocks.py, contrib.py}`, `tests/test_smoke.py`, `configs/`. **No shared-engine change.**

**Cost.** 2 screens (`sf_dcm_abs_t192_s42`, `sf_dcm_surf_t192_s42`) ≈ 5 h each; both readouts are inference-only and ride the existing analysis job.

**Risks.** (a) Two frames = two runs for a likely double null — accepted, because **the null is the result** here and it costs ~10 GPU-hours. (b) With the isotropic stem, stage-4 depth is 1 and stage-3 is 2, so DCM is nearly vacuous there → DCM is **defined on the depth-preserving stem**, which must therefore be the base config (it is). (c) Do not let DCM drift into "add a positional encoding and see": the write-up must name the Mix-FFN slot and the zero-padding-leak argument it tests.

---
### 2.3 C3 — Overlapping Sequence Reduction (OSR): **GATED — the efficiency claim, audited**

**Verdict: one seed-42 screen. It closes a nameable design inconsistency in the source architecture, its expected outcome is a null, and the null is worth stating.**

**Motivation (measured).** SegFormer3D advertises an **overlapping** patch embedding *"that preserves the local continuity of the input voxels"* (stage 1: k=7, s=4 → overlap 3), and then applies a **non-overlapping** K/V reduction (`Conv3d(C,C,k=r,s=r)`, no padding) — *the architecture contradicts its own stated design principle exactly where most of its compute lives* (stage-1 attention = **276.56 of 497.5 GFLOPs** at t256, preset `144_122`). A hard block-pooling of the key grid aliases: at the stage-1 grid (t/4) a 30–50 px stroke is 7–12 tokens wide, so an `r=4` in-plane block is the same order as stroke width, and blocks straddling a stroke edge average ink with papyrus in a phase-dependent way. The frequency probe says mid-scale structure is exactly what matters: the 8–16 px band is worth **−0.119** on `nnf_c1_t256_s42` (−0.077 on UNETR++ `t256_lean_s42`), while 2–4 px "crackle" is worth **−0.001**.

> **OSR = replace each sequence-reduction convolution `Conv3d(C,C,k=r,s=r)` by `Conv3d(C,C,k=2r,s=r,p=r/2)` on every axis whose `r` is even, leaving the output grid identical (`(n + 2·(r/2) − 2r)/r + 1 = n/r`).**

**Exact reduction at init — with an honest caveat.** Build the wide kernel as zeros and copy the stock `k=r` initialisation into the tap block at offsets `[r/2, 3r/2)` on each overlapped axis; keep the bias. In exact arithmetic the extra taps contribute exactly 0.0, so the output is identical. **Caveat that must be written into the test, not discovered later:** cuDNN may select a different algorithm and therefore a different summation order for a different kernel size, so bitwise equality is guaranteed only on CPU. `test_osr_exact_reduction` asserts `max|Δ| == 0.0` **on CPU** and `max|Δ| ≤ 1e-6` on CUDA. **Init trap, also unit-tested:** the stock init is `normal_(0, sqrt(2/fan_out))` with `fan_out = prod(kernel)·C`, so the core block must be drawn with the **stock** kernel's fan-out or the initialisation distribution silently changes and the "ablation" becomes an init change (`test_osr_init_stats`).

**Parameters added.** Preset `144_122`: stage 1 `32·32·(1·8·8) − 32·32·(1·4·4) = +49,152`; stage 2 `64·64·(1·4·4) − 64·64·(1·2·2) = +49,152` → **+98,304 (+2.2%)**. Preset `444_222`: stage 1 `32·32·(8³−4³) = +458,752`, stage 2 `64·64·(4³−2³) = +229,376` → **+688,128 (+15%)** — so **if A2 selects `444_222`, OSR must be reported with a parameter-matched note**, and the adjacent sr preset is the parameter-matched comparison. FLOPs: the stage-1 sr conv goes 0.54 → ≈ 2.16 GFLOPs at t256 (≈ 0.4% of the model).

**Config flag.** `model.sr_overlap: false` (per-axis application is automatic wherever `r` is even).

**Gate.** Seed-42 screen at t192 vs `sf_base_t192_s42`: `|Δraw| ≤ 0.008` → parked as an informative null, **no seeds spent**; `Δraw ≥ +0.008` → 3 seeds, then a t256 confirmation ≥ +0.008 before entering the final config. **Secondary readout (the actual mechanism claim):** frequency band-stop with OSR on/off — the specific prediction is that the **8–16 px and 16–32 px drops shrink**.

**Prediction.** +0.000…+0.008, i.e. most likely a null, stated in advance.

**Informative null (pre-written).** *"Aliasing in the sequence reduction is not a binding constraint for ink: the coarse K/V summary can be blocky because the evidence the model uses is coarser than the blocks. This is a positive result for the efficient design — SegFormer3D's non-overlapping reduction is not a corner it should not have cut."*

**Redundancy check.** *Not the texture stem* (fixed filters at the **input** in the 2–8 px band, priced at ≈ 0): OSR is a **learned** operator at token resolution in the 8–32 px band, priced at −0.08…−0.17. *Not CSSA* (no new K/V source). *Not capacity*: +2.2% params on the default preset, with the adjacent sr preset as the parameter-matched control.

**Touchpoints.** `src/segformer3d_ink/attention.py` (sr construction + init copy), `tests/test_smoke.py` (+2 checks), `configs/`. **No shared-engine change.** **Cost:** 1 screen (≈ 5 h), 1–5 runs if gated.

---

### 2.4 C4 — Decoder Spatial Mixing (DSM): **GATED and headline-excluded — the decoder claim, audited**

**Verdict: one seed-42 screen, first in the cut order. Its value is a *measured substitution* for the best post-hoc lever in the whole project, at 7k parameters.**

**Motivation (measured).** FP mass lives in **small isolated islands** (UNETR++ 115k px in 342 islands; nnFormer `nnf_c1_t256_s42` **190,433 px in 514 islands** of 230,643 total FP px), and the 8-connected min-size filter is worth **+0.010…+0.020 on every model** — the largest post-hoc lever measured anywhere in the thesis. The learned refiner (a 1.93M-param second-stage 2D U-Net over probability maps) added +0.004…+0.013 and cut seed variance ≈ 2.5×. **Both are spatial operations on the output map.** Meanwhile SegFormer3D's decoder is **purely pointwise**: per-stage `Linear(C_i→256)` → trilinear resize → concat → `Conv3d(1024→256,1×1×1)`+BN+ReLU → `Conv3d(256→N,1×1×1)`. After the encoder there is **no spatial mixing anywhere in the decoder**. Xie et al. license this by the encoder's large ERF and show the same decoder fails on CNN encoders — a premise **this thesis can measure rather than assume**.

> **DSM = one zero-gated depthwise convolution on the fused decoder feature map.**
> After `linear_fuse`: `F ← F + γ ⊙ DWConv3d(F, k=3, padding=g, groups=256, dilation=g)`, `γ ∈ R^{256}` zero-init, `g ∈ {1,2}`.
> `g=2` gives support ≈ 4 grid steps ≈ **16 px at full resolution** — stroke-to-ring scale, the quantity the left-word blob analysis identified as decisive.

**Exact no-op at init.** `γ = 0` ⇒ bitwise-identical logits (`test_dsm_exact_reduction`, `max|Δ| == 0.0`).

**Parameters added.** `256·27 + 256 (bias) + 256 (γ) = 7,424` ⇒ **+0.17%**; ≈ 1.8 GFLOPs at t256 (≈ 0.4%).

**Config flags.** `model.dsm: false` (default), `model.dsm_dilation: 1|2`.

**Gate — F0.5 is the *secondary* readout.** Primary, pre-registered:
1. **Post-filter substitution.** Stock's `post − raw` gain is +0.010…+0.020. **DSM must reduce that gain by ≥ 40% while raw F0.5 does not fall (`|Δraw| ≤ 0.008`).** This is a claim about *where the gain comes from*, and it is measurable at n=1 seed from the existing `postprocess.json` sweep.
2. **Error-anatomy FP statistics** (`interp_error_anatomy.py`, CPU, free): isolated-island count and island px drop ≥ 20% at matched recall.
3. Entry into the final config only at ≥ +0.008 post at t256, 3 seeds.

**Prediction.** Raw F0.5 neutral; post-filter gain shrinks; island px drops; τ* shifts down slightly.

**Informative null (pre-written).** *"Island false positives are not a decoder-locality artefact — they are input-level ambiguity that no amount of in-network neighbourhood mixing removes. The min-size filter and the learned refiner are therefore genuine, non-redundant pipeline stages, not patches over an architectural omission."* That is a real defence of the thesis's pipeline result and it is worth one run.

**Redundancy check.** *Not morphology* (opening/closing/fill-holes/bbox-closing, measured dead three times): DSM is a learned feature-space operator trained end-to-end, not a fixed binary-map operation. *Not the min-size filter*: the filter stays on, identically, in every cell — DSM is measured **by its effect on the filter's gain**. *Not the refiner*: the refiner is 1.93M params in a second stage over saved probability maps; DSM is 7.4k params inside the first stage, and the comparison between them is the point. *Not the full-res conv skip* (`skip_stem`): that mixes **raw input voxels** at full resolution with a 5-px receptive field; DSM mixes **fused semantic features** at 1/4 resolution. `skip_stem` is held **fixed ON** in every DSM cell and ablated separately (**A4**) so the two are never confounded.

**Excluded from the cross-model headline whatever happens** — it is a decoder change, and the benchmark's claim is "only the backbone changes". Reported as a SegFormer3D-only row, exactly as the nnFormer plan handled `n_sads`.

**Touchpoints.** `src/segformer3d_ink/{backbone.py (decoder), contrib.py}`, `tests/test_smoke.py`, `configs/`. **No shared-engine change.** **Cost:** 1 screen (≈ 5 h) + free CPU probes. **Paired free probe:** a **decoder stage-dropout** test — zero one stage's slice of the decoder concat at inference and re-score (inference-only, no training) — which is Xie et al.'s Table 1(d) ablation transported to ink.

---

### 2.5 Considered and rejected (these appear in the thesis as verdicts, not as omissions)

| # | Idea | Decisive reason |
|---|---|---|
| R1 | **Surface-weighted / band-sparse sequence reduction** (spend the K/V budget on the ink band) | This is the geometry family, **third instance**. Measured twice: UNETR++ surfrel (**±0.000 post** at the deployment tile) and nnFormer SR-PB (**+0.0065**, inside the floor, after the seed-42 trap deflated +0.0127→+0.0065). The learned form never learns; supervising it hurts on both architectures; and the domain premise (per-segment depth drift) was a **documentation error**. Spending 4–6 runs for a third null in one family is the worst available use of this chapter's GPU budget and in a viva it reads as running the same experiment three times. **What is done instead, free:** every SegFormer3D run already carries the shared `DepthCollapse(surface_relative)`, so the collapse-level geometry prior is inherited at zero cost and Table S-VI is filled by an **inference-only** `stress_zshift` job — third architecture, ~2 GPU-hours instead of ~30, with honestly scoped claims (the prior is *identical shared code* in all three models). |
| R2 | Any further **depth-frame accuracy** mechanism (per-window ŝ, surface-aligned resampling, warp stabilization) | E7 closed the family; per-pixel ŝ and warp-stabilization were measured neutral on UNETR++ and are inherited anti-goals. C2/DCM is admitted **only** because its mechanism (a coordinate in the FFN stream) and its readout (a differential stress asymmetry) are both different. |
| R3 | **2.5D depth-as-channels MiT as a contribution** | It has no exact-no-op-at-init form — it is a different network — so calling it a contribution would break the house rule. Its correct slot is the one the nnFormer plan invented for vFFC: *one control cell and citable convergences, nothing else* (§2.6). |
| R4 | Anything aimed at **"improving confidence on faint ink"** (calibration head, learned temperature, logit prior, focal/Tversky reweighting) | **Rule 2**: swept-τ selection makes any monotone logit rescaling invisible by construction. The near-miss mass is near-miss *at the swept optimum*. Loss reweighting is additionally an inherited anti-goal (E8, measured dead twice). |
| R5 | **Deep supervision** (including a stroke-aware variant) | DS is OFF for every benchmark model by protocol decision; SegFormer3D has none natively. nnFormer's gated `n_sads` was already dropped as lowest-value. |
| R6 | New **losses, photometric augmentation, intensity normalization, morphology** | Inherited anti-goals, each measured dead (E8; morphology dead three times). |
| R7 | **Ensembling, multi-model probability averaging, or the OOF+refiner pipeline as a benchmark row** | The benchmark row is a single-model, single-stage number by construction; the refiner is a separate, already-reported pipeline result of chapter 1. |
| R8 | **Pretrained-init arm in the headline** | Benchmark fairness requires from-scratch for all three models (all previous rows are from-scratch). SegFormer3D's released BraTS weights are nonetheless shape-compatible with the *isotropic* configuration, so a single **optional, gated, non-headline** cell exists (A14) — a curiosity, reported as such. |
| R9 | **t320 for the high-capacity rungs** (SF-L at t320) | Inherited anti-goal; it would recreate exactly the UNETR++ confound the t320 experiment exists to remove. SF-L runs at t256 only. |
| R10 | **Word-box / annotation-anchored sampling** | Measured −0.04…−0.17 with an identified mechanism (train/deployment distribution mismatch). PDK's halo is explicitly designed **not** to reproduce it (deterministic at both train and inference time). |
| R11 | **Attention-temperature tuning as a contribution** | The launch commit uses `1/√num_heads` and the current main uses `1/√head_dim` — a genuine discrepancy worth **documenting** (the released weights and paper numbers correspond to the former), but a temperature is absorbed by training from scratch. One optional zero-cost documentation cell (A11), never a contribution. |
| R12 | **SegFormer3D follow-ups** (LightFormer-3D 2026; SegMaFormer 2026; MoE variants) | Out of scope: the benchmark compares three *published, established* architectures under one protocol. They are cited as related work, not implemented. |

### 2.6 Prior art — 2D SegFormer in the Vesuvius Challenge: **one control cell and three citable convergences**

There is **no published use of SegFormer3D (Perera et al.) on Herculaneum scroll or fragment data**. All SegFormer usage in this community is the **2D** SegFormer applied to depth-collapsed or depth-as-channel inputs, and it is directly relevant prior art for the third chapter:

- **Kaggle "Ink Detection" (2023) 1st place** (ryches team, private F0.5 0.682693): *3D stem → collapse depth → 2D SegFormer*. Concretely `CNN3D_Segformer` = four `Conv3d` (1→4→8→16→32) → `.max(axis=2)` over depth → `SegformerForSemanticSegmentation` (`nvidia/mit-b3`, `num_channels=32`) → two `ConvTranspose2d`; variants used mit-b4/b5 and a `UNETR_Segformer`. Loss `0.5·SoftBCE + 0.5·Dice`, AdamW, warm-up + cosine — **the same loss family and schedule this thesis uses**, arrived at independently.
- **Kaggle 6th place** (chumajin): the cleanest **pure 2.5D** data point — `nvidia/segformer-b3-finetuned-cityscapes-1024-1024`, **3 middle depth slices as channels**, 1024² tiles, stride t/4 → private F0.5 **0.66**.
- **2nd place**: *"2.5D and 3D backbone and decoder without upsampling — 1/32 resolution is sufficient for this task"*; **z-shift ±2 augmentation with p=0.5** — an independent rediscovery of this thesis's `z_shift ±2`.
- **Grand Prize 2023** (Nader/Farritor/Schilliger): TimeSformer-small with divided space-time attention; Nader's retrospective notes *"smaller context models (64×64) had the upper hand over larger context models (256×256 and 512×512)"* — **a direct contradiction of this thesis's measured tile ladder**, and therefore a finding that must be discussed, not buried (the difference is almost certainly label quality, scroll, and the τ/selection protocol).

**What we take.** (1) One control cell `sf_25d_t256_s42` (§6.1, A10) — the same MiT code path built with `nd=2`, the 32 depth layers as input channels: measured **3,433,040 params / 4.8 GFLOPs at t256** against SF-S's 274–497 GFLOPs, a **60–100× compute ratio inside the same architecture family**, answering the thesis's own stated question (*"does 3D attention over depth buy anything at D=32?"*) with the same decoder, the same head, and the same protocol. (2) The three convergences above, cited at zero GPU cost. (3) An honest positioning paragraph: this thesis's numbers are **not** comparable to Kaggle's (different scroll, different labels, different metric surface), which is stated wherever the comparison appears.

**What we do NOT take.** Depth-collapse-by-max before the encoder (that is a different model class, not our backbone); ImageNet/Cityscapes-pretrained encoders (breaks the from-scratch arm); 1024² tiles (outside the ladder and the memory budget); percentile thresholding (τ is swept); ensembling (R7); label-cleaning rounds (our labels are fixed and shared across all three chapters).

**Wiring caveat (binding).** The 2.5D control returns `[B,16,t,t]` and therefore **cannot pass through the shared `DepthCollapse`**, which requires `[B,C,D,H,W]`. It is wired with a singleton depth axis so the shared collapse/head remain byte-identical code, and it is reported as a **control, never as a benchmark row**.

### 2.7 Scope guard

If the schedule tightens, cut in this order: **C4/DSM → C3/OSR → A14/A11/A13 (optional cells) → C2/DCM's second frame → PDK promotion seeds beyond the two decisive bands.** Phases S0, S1, S1b, S2, S3 and S4 ship regardless: **the stock benchmark row and the tile/capacity study are the chapter's obligations to the benchmark; the contributions are on top of them.** No contribution may delay the `sf_c1_t256` 3-seed band.

---
## 3. Architecture spec (SegFormer3D-Ink)

All geometry, parameter and FLOP numbers below were **measured** on a CPU prototype of this exact port (torch 2.6.0, `/home/woody/iwi5/iwi5401h/vesuvius/.venv`), not estimated. Faithfulness check: building the prototype in the reference configuration (isotropic stem, `sr [4,2,1,1]`, prediction-then-upsample, `out_channels=1`) reproduces the released file's parameter count **exactly: 4,491,809**.

### 3.1 Resolution schedule (t256 deployment; exact at every tile in the ladder)

**Default = depth-preserving stem** `Conv3d(1,C₁,k=(3,7,7),s=(1,4,4),p=(1,3,3))`, then stock `(3,3,3)/s2/p1` for stages 2–4:

| Stage | Grid at t256 (D,H,W) | t192 | t128 | t64 | t320 | tokens @t256 | K/V len @t256 (sr `144_122`) | Attention |
|---|---|---|---|---|---|---|---|---|
| Embed → S1 | **(32,64,64)** | (32,48,48) | (32,32,32) | (32,16,16) | (32,80,80) | 131,072 | 8,192 | sr-attention, `sr=(1,4,4)` |
| ↓(2,2,2) → S2 | **(16,32,32)** | (16,24,24) | (16,16,16) | (16,8,8) | (16,40,40) | 16,384 | 4,096 | sr-attention, `sr=(1,2,2)` |
| ↓(2,2,2) → S3 | **(8,16,16)** | (8,12,12) | (8,8,8) | (8,4,4) | (8,20,20) | 2,048 | 2,048 | full attention (`sr=1`) |
| ↓(2,2,2) → S4 | **(4,8,8)** | (4,6,6) | (4,4,4) | (4,2,2) | (4,10,10) | 256 | 256 | full attention (`sr=1`) |
| Decoder | per-stage `Linear(C_i→256)` → trilinear resize to the S1 grid → concat(1024) → `Conv3d(1024→256,1×1×1)`+BN+ReLU → final path to `[B,16,32,t,t]` | | | | | | | |

- **Every stride and every sr ratio divides its grid exactly** for `t ∈ {64,96,128,160,192,224,256,288,320}` (verified on the prototype; all are multiples of 32, so `t/4` is divisible by 4 and `t/8` by 2, while depth 32 is divisible by 4 at S1 and 16 by 2 at S2). **No padding paths**, one code path for the whole ladder. The build asserts divisibility and raises at build time — the reference silently floor-truncates.
- **Params are completely tile-invariant** (no relative-position tables, no positional encoding, no token-axis projections). The only tile-dependent state in the model is the **BatchNorm3d running statistics** (§1, last row) — which is exactly why the resolution-transfer study reports curves with and without them.
- The S1 grid `(32,t/4,t/4)` and bottleneck `(4,t/32,t/32)` **coincide with the nnFormer port's S1 and S4 grids**, so the cross-model stage-resolution table in the thesis lines up row by row.
- **Final full-resolution path** (parity choice, `final: expand`): `ConvTranspose3d(256→16, kernel=stride=(1,4,4))` + the same **full-resolution conv skip** the nnFormer port used (`Conv3d(in,8,3)+GELU+Conv3d(8,16,3)+GELU` on the raw input, fused by `Conv3d(32,16,3)+GELU+Conv3d(16,16,1)`). Alternatives measured and available as ablation **A5**: `trilinear_lowres_pred` (the reference's own order: 1×1×1 to 16 channels at the S1 grid, then upsample) is the cheapest; plain `trilinear` (upsample the 256-channel map, then 1×1×1) is **forbidden above batch 2** — it allocates a 256-channel full-resolution intermediate of **4.0 GB at t256 batch 4 bf16**.
- **PDK hook (§2.1):** the peripheral branch attaches inside the same attention module used by every stage — one implementation, four stages, all zero-gated.
- **DCM hook (§2.2):** the depth-indexed table attaches inside Mix-FFN immediately after the depthwise convolution — one implementation, four stages, all zero-gated.

### 3.2 Capacity rungs (the "does 4.5M suffice?" question, run in both directions)

Blocks and head geometry follow the MiT family; head_dim 32 for SF-S, 64 for SF-M, 96 for SF-L. **Measured parameter counts** (depth-preserving stem, `sr 144_122`, `final: expand` + skip stem, `out_channels=16`, decoder dim 256):

| Variant | dims (S1→S4) | heads | depths | **measured params** | GFLOPs t192 / t256 | Rationale |
|---|---|---|---|---|---|---|
| **SF-S** | [32, 64, 160, 256] | [1,2,5,8] | [2,2,2,2] | **4,437,568** | 202.4 / 497.5 | Stock MiT-B0 — the published SegFormer3D. The efficiency contender and the default for every screen. |
| **SF-M** | [64, 128, 320, 512] | [1,2,5,8] | [2,2,**5**,2] | **19,966,880** | 346.3 / 898.1 | Capacity-matched to **nnF-S (19.87M)** — the cross-model "same size, different family" rung. |
| **SF-L** | [96, 192, 480, 768] | [1,2,5,8] | [3,6,14,3] | **79,353,344** | 836.2 / 2212.6 | Capacity-matched to **UNETR++ t256 = nnF-M (79.1M)** — the fair-comparison rung. t256 only (R9). |

Alternatives at the same 79M budget, kept in the code for reference: `SF-L_deep` [64,128,320,512] depths [3,6,48,3] = 79,090,784; `SF-L_wide` [128,256,640,1024] heads [2,4,10,16] depths [2,4,5,2] = 79,631,264. **SF-L is the width-scaled variant** because scaling depth alone to 48 blocks in stage 3 changes the architecture's shape, not only its capacity.

Reference points: nnF-S 19.87M / 171.8 / 305.6 GFLOPs; nnF-M 79.11M / 900.9 at t256; nnF-F 177.74M / 1860.3; UNETR++ t256 79.1M. **Note the dissociation the efficiency frontier will report:** SF-S has **4.5× fewer parameters than nnF-S but 1.6× more FLOPs at t256** (497.5 vs 305.6) under the depth-preserving stem, because preserving 32 depth rows into stage 1 quadruples the token count. *Parameter efficiency and compute efficiency are not the same axis on anisotropic input* — that is a reportable finding, and it is why A2 (the sr preset) is run early: `444_222` brings SF-S to **274.1 GFLOPs at t256**, below nnF-S, at ±131k params.

**Drop-path** (`drop_path`, timm `DropPath`; the reference uses **none**) is the first rung of the overfit response ladder (§6.4), default **0.0** to stay faithful, raised only on a cell that shows the signature.

### 3.3 Wrapper integration (mirrors the nnFormer port exactly)

- New workspace `$WORK/SegFormer3D/` (own tree, own outputs root — never write into `$WORK/vesuvius/outputs*` or `$WORK/nnFormer/outputs`), laid out exactly like `$WORK/nnFormer/`: `reference/` (pristine clone, never edited) · `src/segformer3d_ink/` · `tests/` · `configs/` · `scripts/` · `docs/` · `outputs/` · `probes/`.
- `$BENCH_DIR/models/segformer3d` → **symlink** to `$WORK/SegFormer3D/src/segformer3d_ink` (single source of truth), exactly as `models/nnformer` is a symlink today.
- Builder signature (mirrors `build_nnformer_ink`):
  ```python
  build_segformer3d_ink(in_ch, img_size=(D,H,W), capacity="S", stem="dp", sr="144_122",
                        final="expand", skip_stem=True, drop_path=0.0, out_channels=16,
                        dec_dim=256, dw_norm="bn", dec_proj_ln=True, scale_mode="head_dim",
                        pdk=False, pdk_stages=(2,3,4), pdk_grid=(8,8,8),
                        dcm=False, dcm_frame="absolute", dcm_stages=(1,2,3,4),
                        sr_overlap=False, dsm=False, dsm_dilation=1, **kw) -> backbone
  ```
- **⚠ CRITICAL — the silent-ignore trap (verified in the existing code, 2026-09-10).** `models/wrapper.py` forwards model keys **one by one** with `m.get(...)`; **any key present in the YAML but not forwarded by the branch is silently ignored, and the run trains a different model than the config describes, with no error.** This is not hypothetical: `build_nnformer_ink` today accepts `cssa_scales`, `dims`, `num_heads`, `depths`, `window_sizes`, `mlp_ratio` and more via `**kw`, and the wrapper forwards **none** of them — so those keys are already dead config surface in chapter 2. The new branch must therefore forward **exactly** this list, and `tests/test_smoke.py` must assert it:
  `capacity · stem · sr · final · skip_stem · drop_path · out_channels(=feature_size) · dec_dim · dw_norm · dec_proj_ln · scale_mode · pdk · pdk_stages · pdk_grid · dcm · dcm_frame · dcm_stages · sr_overlap · dsm · dsm_dilation`
  plus `in_ch` and `img_size=(D,H,W)` from `cfg["data"]`. **Add a `test_config_keys_reach_the_model` check** that builds from each shipped YAML and asserts every `model.*` key (except `name` and `head_classes`) changed something observable — a parameter count, a stage grid, or a forward output. A config key that cannot change the model is a bug, not a comment.
- `data.halo.*` is read by `data/dataset.py`, **not** by the wrapper (it is a data-carrier flag, §2.1); `data.halo.mode: self` selects the same-cost control. The dataset must raise on an unknown `halo.mode` rather than defaulting silently.
  with `backbone.out_channels = 16`, `forward(x:[B,in_ch,D,H,W]) -> [B,16,D,H,W]`, plus the bookkeeping attributes `stage_res`, `input_size`, `feat_size` that `scripts/smoke_test.py` asserts (`prod(feat_size) == input_size[-1]`, `out_channels == cfg.model.feature_size`).
- `models/wrapper.py`: **one** new `elif name == "segformer3d":` branch, lazily importing from `.segformer3d`. Stem / collapse / head untouched: `DepthCollapse(mode="surface_relative")` computes its own ŝ, so surfrel, the φ_rel probe and the stress job all work unchanged.
- **House design rule (obeyed by every component in both previous chapters, and by all four here):** every new component is an **exact no-op behind a default-off flag** (`pdk`/`dcm`/`sr_overlap`/`dsm` all `false`; every gate zero-initialised). That is what makes each ablation a genuine single toggle and lets the parameter count verify each rung.
- Config schema: same YAML families; `model.name: segformer3d` plus the keys above; everything else (`shared`, `data`, `train`, `infer`, `output`) byte-identical to `nnf_c1_t256_s42.yaml`. `output.dir` must equal the `INK_OUTPUT_ROOT` exported in the sbatch (`engine/train.py` does not read the env var; only the post-scripts do).

### 3.4 Port-fix checklist (each item below already bit a previous port, or was measured on this one — check *before* the first submit)

1. **`cube_root(N)` must not survive anywhere.** The released file reshapes flattened tokens to `(n,n,n)` in three places (`SelfAttention` sr path, `DWConv`, the stage loop). Measured consequences for `(B,1,32,t,t)`: **`RuntimeError` at t64, t128, t192, t320** — and at **t256 the forward silently succeeds** because `8·64·64 = 32,768 = 32³`, reinterpreting the `(8,64,64)` grid as `(32,32,32)`, mixing tokens from different depth rows in the sr conv, the depthwise conv and every later stage, and returning shape `(1,1,128,128,128)`. **A t256 run on unmodified code would be silently invalid with no exception raised.** Fix: thread the true `(D,H,W)` grid through every block, assert it at forward time.
2. **`@torch.jit.script` on `cube_root` fails to import under torch 2.6.0** ("Return value was annotated as having type int but is actually of type float") — the file of record does not even load in this environment. Remove it with the reshapes.
3. **Exact-reduction unit tests are mandatory for all four contributions**, in `tests/test_smoke.py`, mirroring `nnFormer/tests/test_smoke.py`: for each of PDK / DCM / DSM, `max|Δ| == 0.0` against the stock forward at init, `max|Δ| > 1e-6` after perturbing the gate, and a finite gradient on every new parameter. For **OSR**, `max|Δ| == 0.0` on **CPU** and `≤ 1e-6` on CUDA (kernel-size-dependent algorithm selection can reorder summation), **plus** an init-statistics test (the core tap block must be drawn with the *stock* kernel's fan-out). **A zero-gated key concatenated into the stock softmax is NOT a no-op** — it enters the denominator; PDK must use a separate softmax branch, and the test is what proves it.
4. **One declared axis convention** (`[B,C,D,H,W]`, D = depth) with asserts at every reshape — the "axis trap" that cost the UNETR++ port (a block-internal `H` was actually depth).
5. **The final upsample must not be isotropic.** The reference's `nn.Upsample(scale_factor=4.0)` restores `(32,t,t)` only because the isotropic stem also divided depth by 4. With the depth-preserving stem it would return depth 128. Use `ConvTranspose3d(kernel=stride=patch_stride[0])`, or interpolate to `x.shape[2:]` explicitly.
6. **bf16 autocast at t192+**; fp32 islands with `nan_to_num` around every soft-argmax chain (DCM's surface frame) — this exact guard was load-bearing in both previous ports. **Never `.to(torch.bfloat16)` the model**: BatchNorm running statistics would be stored with an 8-bit mantissa and the momentum update would lose precision. Under autocast, batch-norm computes in fp32 and is safe.
7. **SDPA is mandatory, not an optimisation.** With the depth-preserving stem the explicit-softmax path would materialise a `[4,1,131072,8192]` attention matrix per stage-1 block = **8.0 GB in bf16 at t256 batch 4** (16 GB if the softmax runs in fp32), and autograd keeps it for backward — the two stage-1 blocks alone would need ≥ 32 GB before anything else. Use `F.scaled_dot_product_attention` (head_dim 32/64/96 ≤ 256, bf16 → flash/mem-efficient kernels, O(N) memory). Keep the explicit path behind a flag **for unit tests only** (it agreed to `max|Δ| = 5.7e-06` on the prototype).
8. **Prove the shared-engine changes inert.** PDK touches `data/dataset.py` and `engine/infer.py`. Before any reported run: (i) `python -m py_compile` both; (ii) `scripts/smoke_test.py --config configs/c1_t256_s42.yaml --device cpu` (a **UNETR++** config) must pass unchanged; (iii) re-run **one epoch** of an existing nnFormer config and diff `train_log.csv` against the archived run — byte-identical or the change is rejected. This is non-negotiable: chapter 1 and chapter 2 results must remain reproducible from the same tree.
9. **Ops rules** (each cost real time once): unique `output.run_name` per run, never reused · `python -m py_compile` every edited file immediately before `sbatch`, re-checking `mtime` · `scripts/smoke_test.py --device cpu` before every submit · **verify `#SBATCH --array` survived any `sed` edit** (a lost array directive once silently ran only task 0) · read SLURM reason fields only after a full scheduler cycle · never compute F0.5 on the test segment, never select τ on it, never touch the val manifest or the mask.
10. **Measured, not estimated:** `scripts/bench.py` (copied from the nnFormer workspace) prints params / GFLOPs / ms-per-tile / peak memory per capacity per tile, filling the efficiency columns of Table S-VII. Run the **t256 batch-4 memory probe on the target GPU before committing the schedule** (nnF-M and nnF-F both OOMed at 40 GB in chapter 2 and had to move to 80 GB nodes; the engine has no gradient accumulation).

---
## 4. Data strategy

1. **Keep the full tile inventory.** Baseline manifests are the existing shared ones — `_reg_t256` (stride 128; 21,360 train / 875 val tiles), `_reg_t192` (stride 96; 37,700 / 1,556), `_reg_t128` (stride 64; 84,680 / 3,504), `_reg` (t64 stride 32; 338,623 / 14,013) — for strict comparability with both previous chapters. **No new manifest is needed for anything except the two t320 cells.**
2. **Two new manifests, both CPU-only, both additive (they create new files and touch nothing existing):**
   - `python -m data.tiling --tile 320 --stride 160 --min-mask 0.5 --label-file ink_reg.npy --suffix _reg_t320` — **already exists** (13,768 tiles), reused as-is.
   - `python -m data.tiling --tile 320 --stride 96 --min-mask 0.5 --label-file ink_reg.npy --suffix _reg_t320_d96` → ≈ 38k tiles. **This is the cell that closes chapter 1's own open recommendation** ("t320-dense: tests whether saturation is sampling or information").
   - The dense t256 manifest `_reg_t256_d64` (84,523 tiles) **already exists** and is reused.
3. **`samples_per_epoch` stays 4000 in every cell.** Density buys **pool diversity per epoch**, not a longer epoch — the same optimizer budget (1000 steps at batch 4) in every comparison. This is what removes the optimizer-budget confound from the density study.
4. **Augmentation: the measured-safe set only** — hflip / vflip / rot90 (p 0.5), z_shift ±2 (p 0.3), **z_flip OFF**, u1 trimap transformed by the same geometric augmentations. No photometric or intensity augmentation (measured harmful; anti-goal). With PDK enabled, **the halo channel receives the identical geometric transform** as the tile.
5. **u1 trimaps** (`proc/<seg>/unc.npy`, built with `--r 4 --label-file ink_reg.npy`) are per-pixel per segment and **reused as-is** for any tile or stride — no new label artifacts anywhere in this chapter.
6. **Labels: `ink_reg.npy` everywhere, without exception.** The `ink_reg2.npy` re-registration round from chapter 2 is a **separate yardstick** and its numbers must never be mixed with these (chapter 2 records this as a hard rule after seven screens were run on it).

**Confound note for the thesis (state it wherever density appears):** a denser stride raises tile *overlap* (70% at stride 96 for t320; 75% at stride 64 for t256), so the extra tiles are not independent samples. The honest claim is *"more distinct letter-crop views under the same optimizer budget."*

---

## 5. Run plan

**Run-naming convention** (mirrors the nnFormer chapter, `n_` → `sf_`): sanity `s0_sanity_t64` · screens and component cells `sf_<toggle>_t<tile>_s<seed>` · capacity/headline runs `sf_<capacity>[_d<stride>]_t<tile>_s<seed>` · final band `sf_c1_t256_s{42,43,44}`. In thesis tables the seed suffix is dropped. Seeds: **42 for every screen; 42/43/44 for the final band and for any promoted contribution cell.**

### Phase S0 — port + verification (CPU, plus one short GPU sanity job)

- Clone `github.com/OSUPCVLab/SegFormer3D` into `$WORK/SegFormer3D/reference/` (**never edited**; file of record `architectures/segformer3d.py`). Port into `src/segformer3d_ink/` following §3.1–§3.3, applying **every** item of the §3.4 checklist.
- Unit tests (`tests/test_smoke.py`): stage-grid and divisibility assertions for `t ∈ {64,128,192,256,320}`; measured parameter counts for SF-S/M/L; the reference-configuration faithfulness check (**must reproduce 4,491,809**); exact-reduction tests for PDK / DCM / OSR / DSM; SDPA-vs-explicit agreement; forward+backward at t64.
- `scripts/bench.py` on the target GPU → params / GFLOPs / ms-per-tile / peak memory for SF-S/M/L at t192/t256/t320 → `docs/bench_table.md`.
- **t256 batch-4 memory probe (bf16, forward+backward) before committing the schedule** — pre-decided fallbacks in order: (i) `sr 444_222` (1.8× cheaper), (ii) 80 GB nodes (`--constraint=a100_80`), (iii) batch 2 with the confound stated. The engine has **no gradient accumulation**.
- `s0_sanity_t64` (2 epochs, **not reported**): loss decreases; τ-sweep → val → postprocess → figures run end-to-end.
- **Bench-inertness proof** (§3.4 item 8) — this gates every later phase.

### Phase S1 — adaptation screens (t192, seed 42, SF-S, A40 or A100 ≈ 5 h each)

t192 is the screen scale (cheaper, and where both previous ablation suites ran). Base recipe for every `sf_` run: SF-S · depth-preserving stem · `sr 144_122` · `final: expand` + full-res conv skip · shared `DepthCollapse(surface_relative)` + shared `Head` · DS off · from scratch · `dice_bce_masked` on u1 · z_flip off · bf16 · batch 4 · 60-epoch cosine · `select_on: sweep`. Reference = its own `sf_base_t192_s42` band. **All 12 t192 cells are listed in Table S-I (§6.1)** (the reference plus A1, A2×2, A3, A4, A5, A6, A7, A8, A12, A13; A10 is a t256 control and A11/A14 are optional).

**Decision rules (identical to both previous chapters):** `|Δ| ≤ 0.008` vs `sf_base` → parked/neutral (*that is itself a Family-D result*); a toggled-off run that **drops** by > 0.008 → the stock component is confirmed and kept; a screen that **gains** > 0.008 → the base config is amended before Phase S2 and the amendment is noted. **Tile-interaction guard:** any large screen gain on a contribution triggers a reduced-tile (t128) check before it is written up as an architecture-level finding — the check that would have caught the DA-EPA artifact early.

**Order matters:** run **A2 (sr preset) and A1 (stem) first**. A2 is a 1.8× compute decision (497.5 vs 274.1 GFLOPs at t256) and its winner sets the cost of every later phase.

### Phase S1b — the tile ladder **and the ERF measurement, run before any context experiment** (the power check)

`sf_base` at **t64, t128, t256, t320** (t192 exists from S1) → the **5-point ladder**, then `interp_erf.py` on the t128/t192/t256 checkpoints. This phase is the chapter's spine and it is deliberately scheduled **before** C1:

- It tests **prediction 2** (the ERF law: r90/tile ≥ 50%, non-shrinking ⇒ steep ladder), the thesis's only predictive claim, made in advance and now on its third architecture.
- It measures `G = raw(t192) − raw(t128)`, which **is** C1's gate parameter (Rule 3). If `G < 0.02`, the 2×2 is cancelled before it is run.
- It gives the **first clean tile ladder in the benchmark**: tile-invariant parameters (E2), so no capacity confound — unlike UNETR++ (25M→112M) and unlike chapter 1's t320 cell.
- **t320 is run here, deliberately, at 4.44M parameters** (Rule 4, prediction 4). Both `_reg_t320` (stride 160, 13,768 tiles) and `_reg_t320_d96` (stride 96, ≈38k tiles) cells, because the confound chapter 1 could not separate — data limit vs overfit vs sampling — needs both.

### Phase S1c — contribution screens (t192/t128, seed 42)

`sf_pdk_t192_s42`, `sf_pdk_t128_s42`, `sf_pdkctl_t128_s42` (the mandatory same-cost control), `sf_dcm_abs_t192_s42`, `sf_dcm_surf_t192_s42`, `sf_osr_t192_s42`, `sf_dsm_t192_s42`. The stock 2×2 cells come free from S1b. Gates exactly as pre-registered in §2.1–§2.4; **no promotion seeds are spent until a gate passes.**

### Phase S2 — capacity × context grid + density (the headline SegFormer3D study)

The experiment neither predecessor could run cleanly (Rule 4, E12). Fixed batch 4, `samples_per_epoch` 4000, 60-epoch cosine in every cell:

| run | capacity | tile | manifest (tiles) | question |
|---|---|---|---|---|
| `sf_base_t192_s42` | SF-S 4.44M | 192 | `_reg_t192` (37,700) | (from S1) |
| `sf_base_t256_s42` | SF-S | 256 | `_reg_t256` (21,360) | (from S1b) |
| `sf_base_t320_s42` | SF-S | 320 | `_reg_t320` (13,768) | (from S1b) — t320 without the capacity confound |
| `sf_base_d96_t320_s42` | SF-S | 320 | `_reg_t320_d96` (≈38k) | is the ladder's end sampling or information? |
| `sf_m_t192_s42` | SF-M 19.97M | 192 | `_reg_t192` | capacity × context, cell 1 |
| `sf_m_t256_s42` | SF-M | 256 | `_reg_t256` | capacity × context, cell 2 |
| `sf_m_t320_s42` | SF-M | 320 | `_reg_t320` | capacity × context, cell 3 |
| `sf_l_t256_s42` | SF-L 79.35M | 256 | `_reg_t256` | the fair-capacity match to UNETR++/nnFormer; expect the overfit signature |
| `sf_s_d64_t256_s42` | SF-S | 256 | `_reg_t256_d64` (84,523) | does density help at 4.4M? (nnFormer's best config used it) |
| `sf_l_d64_t256_s42` — *gated* | SF-L | 256 | `_reg_t256_d64` | run **only if** `sf_l_t256` shows the overfit signature: does density rescue full capacity? |

**Read every cell with three numbers**, exactly as chapter 2 did: swept F0.5 (raw/post), best epoch, and the train-loss-vs-val divergence. **Overfit is never claimed without the signature.**

### Phase S3 — final combo + seeds (produces the benchmark row)

`sf_c1_t256_s{42,43,44}` = best capacity × best density × any S1 amendment (+ PDK / DCM / OSR / DSM **only where their gates passed**) + u1 + z_flip off, with TTA and the (τ, min_size) component filter at rescoring — exactly as `c1_t256` and `nnf_c1_t256`. **This 3-seed band is the benchmark number** against UNETR++ `c1_t256` **0.8111 ± 0.0039 raw / 0.8190 ± 0.0062 post** and nnFormer `nnf_c1_t256` **0.7801 ± 0.0037 / 0.8005 ± 0.0102**. If any contribution is adopted, the **stock twin** (`sf_stock_t256_s42`) is kept so the benchmark row and the contribution row stay separable.

### Phase S4 — the shared analysis layer (inference-only, 2 jobs)

All probes are model-agnostic and already written; commands mirror `slurm_n4_analysis_a100.sbatch` and `slurm_inkalyzer_a100.sbatch`, with `INK_OUTPUT_ROOT=$WORK/SegFormer3D/outputs` and `--out $WORK/SegFormer3D/probes`:

1. **Inkalyzer** depth attribution (`inkalyzer_run.py --runs …`) — does a third architecture also lock onto z27–33? Third replication of the trust result, **plus C2's table-vs-probe comparison** (§2.2 readout 1). *Note: the tile must be divisible by 16 for `attribute_3d`.*
2. **ERF** (`interp_erf.py --models root:run:manifest,…`) — the decisive third point of the ERF law (E10), measured at t128/t192/t256, **with PDK on/off as C1's mechanism figure** (and, uniquely here, the ERF measured **on the halo channel**, via a copy of the probe in the workspace — never edit the shared one).
3. **Frequency band-stop** (`interp_freqband.py`) — morphology-vs-crackle, third replication; **with OSR on/off as C3's mechanism readout** (predicted: the 8–16 px and 16–32 px drops shrink).
4. **Error anatomy** (`interp_error_anatomy.py`, CPU) — per-stratum recall against UNETR++ and nnFormer (the pre-declared axis on which a 4.4M model may win at equal F0.5), **plus C4's FP-island statistics**.
5. **G1 depth-offset stress** (`stress_zshift.py --deltas -8 -6 -4 -2 0 2 4 6 8`, frozen τ) on `sf_base` (3 seeds — *band vs band from the start; the seed-42 lesson, applied prospectively*) and on both DCM frames → **Table S-VI, the cross-architecture geometry table completed on its third architecture for ~2 GPU-hours** (§2.5 R1).
6. **TTA** (`predict_val_tta.py`) on the three final seeds; IoU backfill (`rescore_iou.py`); val 4-panel figures; test-strip qualitative prediction on `flatboi` (τ 0.65–0.70, min_size ≤ 10k, always showing the raw map).
7. **Resolution-transfer study** (new, SegFormer3D-only, inference-only — `scripts/stress_tile.py`, built exactly like `stress_zshift.py`): load `best.pt`, read the frozen τ from `summary.json`, deep-copy the config, override `data.tile = t'` and `infer.stride = t'/2`, run `infer_segment` on the val segment, score at both the frozen and a re-swept τ, for `t' ∈ {128,192,256,320,384}`. **This is only possible because the model has no tile-dependent parameters (E2)**, and it tests the source paper's own claim that a positional-encoding-free design "prevents accuracy loss when there is a resolution mismatch during training and inference" — in 3D, on ink, across a 16× change in token count. Run **with and without** the added BatchNorms (A3), since BN running statistics are the only tile-dependent state. **Pre-registered:** `|Δ F0.5| ≤ 0.008` for one rung up or down ⇒ the claim holds; if train-t192 → infer-t256 gains ≥ +0.008, report it as a **deployment rung** ("train narrow, infer wide") — free context at zero training cost, which would be a genuinely useful practical result.

### Phase S5 — cross-model synthesis (CPU, no GPU)

Fill Tables S-III (component × scale across three models), S-VII (the benchmark headline row), the **efficiency frontier** figure (post F0.5 against measured params / GFLOPs / ms-per-tile / peak memory for all three models, plus the 2.5D control), and the **cross-model agreement table** — chapter 2's six questions, now with a third column. Five of six agreed between UNETR++ and nnFormer; the sixth (does tile size dominate?) disagreed and was *explained* by the ERF ratio. **SegFormer3D is the adjudicating third point.**

### Budget (from measured wall times, E13)

A100: t64 ≈ 1.5 h · t128 ≈ 3 h · t192 ≈ 5 h · t256 ≈ 8 h · t320 ≈ 12 h (SF-S; scale by the FLOPs ratio for SF-M/SF-L, but note these runs are expected to be **loader-bound** at the small end — the per-epoch full-segment validation sweep and the data loader dominate below ~150 GFLOPs, so the t64/t128 cells will not be as cheap as their FLOPs suggest). A40 ≈ 1.45× A100 and is the **reliable fallback** when the account hits `AssocGrpGRES`.

| phase | runs | ≈ GPU-hours |
|---|---|---|
| S0 sanity + bench + memory probe | 1 short | 1 |
| S1 adaptation screens (12 cells) | 12 | 60 |
| S1b tile ladder + t320 dense (5 cells) | 5 | 32 |
| S1c contribution screens (7 cells) | 7 | 38 |
| S2 capacity × context (7 new cells) | 7 | 62 |
| S3 final band (3 seeds + stock twin) | 4 | 32 |
| S4 analysis (2 inference jobs) | — | 10 |
| **core total** | **36** | **≈ 235 GPU-h ≈ 10 GPU-days** |
| *gated:* PDK promotion (8) + deployment screens (3) + optional cells (3) | *up to 14* | *up to +110* |

**Upper bound ≈ 50 runs ≈ 14 GPU-days**, and the upper end only materialises if PDK earns its promotion — which is the right reason to spend it. The scope guard (§2.7) binds; wall-time fallbacks are pre-decided (§S0).

---
## 6. Ablation tables for the thesis (templates, comparable to chapters 1 and 2)

### 6.1 Table S-I — SegFormer3D adaptation + component study (t192, seed-42 screens vs `sf_base_t192`; promoted rows get 3 seeds)

| # | Component toggled | Run | stock reference value | ink value (default) | raw F0.5 @τ | post | Δraw | Pre-registered expectation | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| — | — (reference) | `sf_base_t192` | — | — | | | — | — | |
| A1 | Patch-embed stem: **depth-preserving → isotropic** | `sf_stemiso_t192` | `k7/s4/p3` all axes | `k(3,7,7)/s(1,4,4)/p(1,3,3)` | | | | dp ≥ iso. **If iso is within ±0.008 it is a major efficiency result** (2.3× cheaper at t256: 120.5 vs 274.1 GFLOPs) and becomes the deployment config | |
| A2 | Sequence-reduction preset | `sf_sr444_t192`, `sf_sr244_t192` | `[4,2,1,1]` (=r³ isotropic) | `(1,4,4)/(1,2,2)/1/1` | | | | all three inside ±0.008 ⇒ adopt the cheapest and report the **1.8× compute saving as a measured result** | |
| A3 | Remove the repo's added norms (NVlabs-exact) | `sf_nonorm_t192` | 8 BN after Mix-FFN DWConv + 4 decoder LN | kept | | | | neutral at the training tile; **decisive for the resolution-transfer curve** (S4.7) | |
| A4 | Full-resolution conv skip | `sf_noskip_t192` | absent in the reference | ON (UNETR++/nnFormer parity) | | | | neutral to +0.010; if neutral, the deployment recommendation is "drop it, −74.5 GFLOPs at t256" | |
| A5 | Final full-resolution path | `sf_finlow_t192` | `pred → Upsample` | `ConvTranspose3d` expand | | | | neutral; adopt `expand` for cross-model parity | |
| A6 | Class-A depth collapse: `surface_relative → mean` | `sf_mean_t192` | — (shared layer) | `surface_relative` | | | | inside ±0.008 (the prior fades with tile size; this model deploys at t256+) | |
| A7 | Decoder width 256 → 128 | `sf_dec128_t192` | 256 | 256 | | | | inside ±0.008 ⇒ another cheap deployment saving | |
| A8 | Depths `[2,2,2,2] → [2,2,5,2]` at SF-S width | `sf_deep_t192` | `[2,2,2,2]` | `[2,2,2,2]` | | | | neutral (isolates *depth* from *width* inside the capacity study) | |
| A12 | LR 1e-4 → 5e-5 | `sf_lr5e5_t192` | 1e-4 (repo) / 3e-5 (paper) | 1e-4 | | | | neutral; doubles as the pre-decided instability fallback | |
| A13 | Weight decay 3e-5 → 1e-2 | `sf_wd1e2_t192` | **1e-2** (repo) | 3e-5 (protocol) | | | | neutral; the one hyper-parameter where the source differs by 300× | |
| A10 | **2.5D control** (depth-as-channels 2D MiT) | `sf_25d_t256` | — (prior art) | — | | | | ≤ 0.02 below its 3D twin at 60–100× less compute | *control, never a benchmark row* |
| A11 | Attention scale `1/√head_dim` vs `1/√num_heads` | `sf_scaleheads_t192` — *optional* | launch commit used `1/√num_heads` | `1/√head_dim` | | | | neutral when trained from scratch; the value is the documented discrepancy | |
| A14 | BraTS-pretrained init (isotropic geometry only) | `sf_medinit_t192` — *optional, non-headline* | — | from scratch | | | | neutral to small; informative either way | |

### 6.2 Table S-II — capacity × context grid (the study neither predecessor could run cleanly)

| | t192 | t256 | t320 (stride 160) | t320 (stride 96, ≈38k) | best epoch | overfit signature? |
|---|---|---|---|---|---|---|
| **SF-S** (4,437,568) | | | | | | |
| **SF-M** (19,966,880) | | | | *(not run)* | | |
| **SF-L** (79,353,344) | *(not run)* | | *(anti-goal R9)* | *(not run)* | | |
| SF-S + dense t256 (`_reg_t256_d64`, 84,523) | — | | — | — | | |

Caption: parameters are **tile-invariant** (measured) — unlike UNETR++ (25M → 79.1M → 112.2M across the same rungs), so **this is the benchmark's only capacity-confound-free ladder**. `samples_per_epoch` = 4000 in every cell (density changes pool diversity, not optimizer budget); the overlap confound is stated in §4. Reference points: UNETR++ t320 turned down −0.009 raw / −0.015 post **with** the overfit signature at 112.2M params; nnFormer S/M/F spanned only 0.006 raw across 19.9M→177.8M.

### 6.3 Table S-III — cross-model component × scale matrix (completes chapter 1's Family D on three architectures)

| Component | UNETR++ t64 | UNETR++ t192 | UNETR++ t256 | nnFormer t192 | nnFormer t256 | **SegFormer3D t192** | **SegFormer3D t256** |
|---|---|---|---|---|---|---|---|
| Class-A collapse (surfrel vs mean) | +0.003 | **−0.009** | +0.008 raw / ±0.000 post | (neutral) | | | |
| Texture stem | **−0.073** | +0.000 | absorbed | (not run) | — | (not run) | — |
| Signature attention component | DA-EPA −0.004 | −0.004 | dropped | skip-attn **−0.0002** | | sr preset (A2) | |
| Depth-preserving embedding | a0b: neutral | — | — | `n_embedz2` −0.0115 | — | **A1** | |
| Depth-frame mechanism | DA-EPA −0.004 | −0.004 | dropped | SR-PB **+0.0065** | | **C2/DCM** (δ=0 row; tails in S-VI) | |
| **Tile ladder slope (t128→t192)** | — | **+0.055** | — | **+0.0063** | — | **(prediction 2: ≥ +0.03)** | — |
| **ERF r90 / tile** | 53% | 53% | 53% | **51%** | **45%** | **(prediction 2: ≥ 50%, non-shrinking)** | |

Message to test: *"one real component per regime, nothing everywhere"* — does it replicate on a **third** architecture? And: **is the ERF ratio predictive, or was it a post-hoc story?**

### 6.4 Pre-registered overfit response ladder (applied in order, one step per re-run, only on cells showing the signature)

1. dense manifest (already in the grid) → 2. `drop_path` 0.0 → 0.1 → 3. `drop_path` 0.2 → 4. weight decay 3e-5 → 1e-2 (the repo's own value, and A13 will already have screened it) → 5. capacity rung down. **Stop at the first step that clears the signature; report the path taken.**

### 6.5 Table S-IV — the PDK 2×2 (C1's centerpiece; 3-seed mean ± sd once promoted)

| | t128 | t192 | Δ(128→192) |
|---|---|---|---|
| stock | `sf_base_t128` | `sf_base_t192` | **`G` — the gap PDK must close, and the gate's power check** |
| **PDK** | `sf_pdk_t128` | `sf_pdk_t192` | |
| **PDK same-cost control** (halo = downsampled copy of the tile) | `sf_pdkctl_t128` | — | |
| Δ(PDK − stock) | | | |
| Δ(PDK − control) | **the information claim** | — | |

Caption: SF-S, batch 4 / 1000 steps per epoch in every cell (optimizer-budget confound removed). Pre-registered success = PDK@t128 ≥ stock@t128 + 0.02 **and** ≥ 50% of `G` **and** ≥ control + 0.008. **If `G < 0.02` the 2×2 is declared underpowered and cancelled before it runs** (the CSSA lesson, E11). `γ_h` trajectories in the appendix figure regardless of outcome. Plus one line: the `sf_pdk_t256` deployment-gate result.

### 6.6 Table S-V — C2/DCM readouts (F0.5 is the *secondary* column)

| Readout | stock | DCM absolute | DCM surface | Pre-registered |
|---|---|---|---|---|
| raw F0.5 @ t192 (δ=0) | | | | all within ±0.008 |
| Learned table peak `argmax_d ‖P₁[d]‖` (absolute frame) | — | | — | **falls inside z27–33** |
| Inkalyzer peak z / norm entropy (same checkpoint) | | | | unchanged vs stock |
| Frozen-τ F0.5 at \|δ\| = 6 | | | | **absolute loses ≥ 0.02; surface does not** |
| Frozen-τ mean over δ ∈ [−8,8] | | | | |

### 6.7 Table S-VI — cross-architecture geometry stress (frozen-τ F0.5 vs induced depth offset δ), now on **three** architectures

| Model / mechanism | δ=0 | \|δ\|=4 | \|δ\|=6 | \|δ\|=8 | mean over [−8,8] |
|---|---|---|---|---|---|
| UNETR++ lean (surfrel collapse prior) | 0.775 band | +0.03…0.05 vs OFF | +0.05…0.08 | +0.08…0.10 | |
| UNETR++ aux-ŝ (supervised geometry) | −0.050 | | +0.13 @ −8 | | 0.596 = lean |
| nnFormer `n_base` / `n_srpb` | | | | | (SR-PB better at every δ, mean +0.035, but n=1 on the baseline side) |
| **SegFormer3D `sf_base` (3 seeds — band vs band from the start)** | | | | | |
| **SegFormer3D DCM absolute / surface** | | | | | |

Caption: the cross-architecture geometry claim closes here. The collapse prior is **identical shared code** in all three models, so this row set isolates architecture from mechanism. Note explicitly that chapter 2's SR-PB robustness curve used a **single** baseline seed; this chapter runs the baseline band from the start — the seed-42 lesson applied prospectively rather than retrospectively.

### 6.8 Table S-VII — benchmark headline (the table completes here)

| Model (final config) | F0.5 raw | F0.5 post | +TTA | P | R | IoU_ink | IoU_bg | params (meas.) | GFLOPs | ms/tile | peak mem | recall: near-miss / faint / blind | mean F0.5 over δ∈[−8,8] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UNETR++ `c1_t256` (3 seeds) | 0.8111 ± .0039 | **0.8190 ± .0062** | ~0.826 | | | 0.585 | 0.911 | 79.1M | | | | .32/.49/.187 | |
| nnFormer `nnf_c1_t256` (3 seeds) | 0.7801 ± .0037 | **0.8005 ± .0102** | 0.8133 | | | | | 19.87M | 305.6 | 36.5 | | | |
| **SegFormer3D `sf_c1_t256` (3 seeds)** | | | | | | | | **4.44M** | | | | | |
| *2.5D control `sf_25d_t256`* | | | | | | | | *3.43M* | *4.8* | | | | *(control, not a benchmark row)* |

Caption: identical split / labels / recipe / inference / post-processing / τ-protocol; τ and min_size selected on val per model; IoU reported, never selected on; ± = 3-seed sd; **DS off for all models** (protocol decision); contributions included only where their gates passed, with the stock twin reported for attribution.

### 6.9 Table S-VIII — the efficiency frontier (the chapter's own figure)

Post F0.5 (3-seed mean) against **measured** params, GFLOPs, ms-per-tile and peak training memory, for all three models plus the 2.5D control. The pre-registered reading: **if SegFormer3D lands within 0.02 of UNETR++ at 4.44M vs 79.1M params, the thesis's central claim ("data, scale and protocol dominate architecture") gets its sharpest possible statement** — an 18× parameter reduction costs less than the gap between a swept and a fixed threshold. State the dissociation honestly: SF-S has **fewer parameters but more FLOPs** than nnF-S under the depth-preserving stem (497.5 vs 305.6 at t256), because preserving 32 depth rows quadruples the token count — *parameter efficiency and compute efficiency are different axes on anisotropic input.*

### 6.10 Cross-model agreement table (chapter 2's six questions, adjudicated)

| Question | UNETR++ | nnFormer | **SegFormer3D** | Agree? |
|---|---|---|---|---|
| Do stock architecture toggles matter? | No | No | | |
| Does capacity help? | No (t320 overfit) | No (S≈M≈F) | **(4.4M→79.4M, 3 tiles)** | |
| Where in depth does the model look? | z29–34, ~0 at edges | z32–33, ~0 at edges | | |
| Is fine crackle texture the signal? | No (2–8 px ≈ 0) | No (2–4 px = −0.001) | | |
| Is the residual error profile the same? | 71.6% unexplained | 72.1% unexplained | | |
| **Does tile size / context dominate?** | **Yes (+0.16)** | **No (+0.0063)** | **(prediction 2: Yes)** | **the adjudicating point** |

---
## 7. How this preserves the benchmark structure

**Unchanged, byte-identical, shared:** split (train 8 / val `20241113090990` / test `flatboi`, qualitative only) · registered labels `ink_reg.npy` + masked evaluation · swept-τ selection and early stop (`select_on: sweep`, `sweep_every: 1`) · 60-epoch cosine, warmup 3, AdamW 1e-4, wd 3e-5, grad-clip 1.0, bf16, batch 4, `samples_per_epoch` 4000, 1:1 positive-tile oversampling (`pos_thresh` 0.05) · augmentation set (hflip/vflip/rot90 + z_shift ±2, z_flip OFF) · u1 trimaps + `dice_bce_masked` · sliding-window inference at stride t/2 with Gaussian blending · 8-fold dihedral TTA · (τ, min_size) component filter selected on val · seeds 42/43/44 and the ±0.008 noise floor · IoU reported at both operating points, never selected on · the probe suite (Inkalyzer / ERF / frequency band-stop / error anatomy / G1 stress) · artifact layout (`summary.json`, `postprocess.json`, `train_log.csv`, `figs/`) · ops rules (unique run names, `py_compile` + CPU smoke test before every `sbatch`, own output root `$WORK/SegFormer3D/outputs`) · **deep supervision OFF for every benchmark model**.

**The only degrees of freedom this model gets** are (a) its own backbone hyper-parameters (capacity, stem, sr preset, decoder width — Tables S-I/S-II), (b) one LR point and one weight-decay point, (c) `drop_path` as its native regulariser, and (d) the **gated, ablation-isolated contributions** (PDK, DCM, OSR, DSM), each reducing exactly to stock at init and each excluded from the headline unless its gate passed, with the stock twin reported.

**Two structural exceptions, both stated in the thesis methods rather than hidden:**
1. **PDK is the first contribution in this benchmark that touches the shared data path** (`data/dataset.py`, `engine/infer.py`). It is default-off and its inertness is *proved* (§3.4 item 8) by re-running an existing config and diffing the training log. Without that proof it does not ship.
2. **t320 is promoted from anti-goal to experiment for this model only**, because the reason it was an anti-goal (a 112M-parameter model overfitting 13,768 tiles) does not apply to a 4.44M-parameter model with tile-invariant parameters. **SF-L is still never run at t320** (R9).

**Anti-goals inherited verbatim** from `plan.md` §12 and the nnFormer plan: no loss engineering · no intensity normalization or photometric augmentation · no morphology beyond the min-size filter · no LOFO · no word-box sampling · no probability ensembling · no hysteresis · no inference-time region priors · never touch the val manifest, the mask, or the test labels · never select any hyper-parameter on the test segment. **Newly added anti-goals from §2.5:** surface-weighted / band-sparse K/V reduction (R1), any further depth-frame accuracy mechanism (R2), 2.5D as a contribution (R3), the entire confidence/calibration family (R4, retired analytically by Rule 2), a pretrained-init headline arm (R8), SF-L at t320 (R9), and SegFormer3D follow-up architectures (R12).

---

## 8. Risk register (pre-made responses)

| Risk | Pre-planned response |
|---|---|
| The reference code's `cube_root` reshape silently "works" at t256 and someone trains with it | §3.4 item 1 is the first checklist item, and `tests/test_smoke.py` asserts the true `(D,H,W)` grid at every stage for all five tiles. **A t256 run on unmodified code is invalid without raising** — this is the single most dangerous property of the source repo. |
| SF-S at t256 with the depth-preserving stem OOMs at batch 4 | Pre-decided order: (i) `sr 444_222` (497.5 → 274.1 GFLOPs, and A2 may already have selected it), (ii) `--constraint=a100_80`, (iii) batch 2 with the optimizer-budget confound stated. **Never** `final: trilinear` above batch 2 (4.0 GB intermediate). |
| Explicit-softmax attention path is used by accident (e.g. an old torch, a masked call) | 8 GB attention matrix per stage-1 block at t256 batch 4. `tests/test_smoke.py` asserts SDPA is the active path; the explicit path exists only for unit tests. |
| Training is loader-bound, so the small cells are not as cheap as their FLOPs suggest | Expected and stated in the budget. Measure the loader (`num_workers`, per-epoch time) in S0 **before** blaming the model or the GPU. |
| PDK's engine changes break a previous chapter's reproducibility | §3.4 item 8: `py_compile`, the UNETR++ smoke test, and a one-epoch `train_log.csv` diff against an archived run. If the diff is not byte-identical, the change is rejected and PDK falls back to a workspace-local dataset subclass. |
| PDK's gate is unreachable because SegFormer3D's own ladder is flat | **That is the CSSA failure mode, and Rule 3 catches it before the runs**: if `G < 0.02` the 2×2 is cancelled, PDK is evaluated at deployment scale only, and the flat ladder itself becomes the headline (it would refute prediction 2 and the ERF law, which is a bigger result than PDK). |
| PDK helps, but only because it adds parameters/keys | The **same-cost control** (`sf_pdkctl`) is mandatory and part of the gate, not an afterthought. |
| PDK's halo at segment borders is mostly zeros | `halo_mask_frac` is recorded per tile; the gate is re-checked on interior-only tiles. |
| DCM's γ never leaves zero | Expected outcome (prediction 6) and the pre-written null is the result. The **table-vs-Inkalyzer** comparison and the **stress sign asymmetry** are the deliverables, not the F0.5. |
| Both DCM frames are neutral and a reviewer asks why two runs were spent | Because the *contrast* between frames is the experiment: the absolute frame is predicted to break under offset while the surface frame is not. One run cannot show an asymmetry. |
| OSR's "exact reduction" is not bitwise on GPU | Known and pre-stated: cuDNN may reorder summation across kernel sizes. The test asserts bitwise on CPU, ≤ 1e-6 on CUDA. |
| A capacity rung shows the overfit signature | Apply the response ladder §6.4 in order, one step per re-run, and report the path taken. |
| SegFormer3D beats UNETR++ by > 0.02 | Treat as suspect first (project history: surprising jumps are protocol bugs) — check the label file, the manifest, the τ column, that val was never sampled, and that the geometry assertions are active — **then** celebrate. |
| SegFormer3D lands far below both (> 0.05 worse) | A legitimate result for a 4.44M model, but first check the sr preset (A2), the stem (A1), and whether the run is loader-starved. Report the efficiency frontier regardless: the frontier is the chapter's contribution even if the accuracy is not competitive. |
| The account hits `AssocGrpGRES` on a100 | **A40 fallback with 24 h wall** (`--gres=gpu:a40:1`), measured at ≈ 1.45× A100 wall time. Do not re-read scheduler reason fields within one cycle. |
| A `sed` edit drops a `#SBATCH --array` directive | Verified after every edit (`grep -c '^#SBATCH --array'`) — this silently ran only task 0 once in chapter 2, and the log filename `..._4294967294.out` was the only symptom. |
| Contribution work crowds out the benchmark | Scope guard §2.7: cut DSM → OSR → optional cells → DCM's second frame → PDK promotion seeds. **Phases S0–S4 ship regardless.** |
| The 2.5D control's output shape breaks the shared collapse | Known (§2.6): it returns `[B,16,t,t]`. Wire it with a singleton depth axis so the shared collapse/head stay byte-identical, and report it as a control, never as a benchmark row. |

---

## 9. Deliverables checklist

- [ ] `$WORK/SegFormer3D/` workspace + pristine `reference/` clone + `src/segformer3d_ink/` port + `models/segformer3d` symlink + the `wrapper.py` branch (Phase S0)
- [ ] `tests/test_smoke.py`: geometry assertions for t ∈ {64,128,192,256,320}, measured params SF-S/M/L, the **4,491,809 reference-faithfulness check**, exact-reduction tests for PDK / DCM / OSR / DSM, SDPA-vs-explicit agreement, **and `test_config_keys_reach_the_model`** (every `model.*` key in every shipped YAML provably changes the built model — the silent-ignore trap, §3.3)
- [ ] `docs/PORT_LOG.md` with the deviation table D1…Dn (the `cube_root` scrambling at t256 is D1 and is a **thesis-reportable finding**, not just a bug)
- [ ] `docs/bench_table.md` — measured params / GFLOPs / ms-per-tile / peak memory for SF-S/M/L at t192/t256/t320
- [ ] Bench-inertness proof for the `data/dataset.py` + `engine/infer.py` changes (one-epoch `train_log.csv` diff)
- [ ] `train_reg_t320_d96.csv` dense manifest (CPU)
- [ ] Table S-I (adaptation + component screens) with verdicts
- [ ] Table S-II (capacity × context grid) with overfit diagnostics — **the capacity-confound-free ladder**
- [ ] Table S-IV (PDK 2×2 + same-cost control), promoted to 3 seeds only if the gate passed
- [ ] Table S-V (DCM: learned table vs Inkalyzer profile; stress sign asymmetry)
- [ ] Table S-VI (cross-architecture geometry stress, **three** architectures, baseline band from the start)
- [ ] `sf_c1_t256` 3-seed band, TTA'd and post-filtered → Table S-VII row (+ stock twin if a contribution was adopted)
- [ ] Table S-VIII efficiency frontier figure (3 models + 2.5D control)
- [ ] Resolution-transfer curves (S4.7), with and without the added BatchNorms
- [ ] Probe artifacts: Inkalyzer depth profile, ERF (incl. **on the halo channel**, PDK on/off), frequency band-stop (OSR on/off), error anatomy (per-stratum recall + FP islands), G1 stress, γ_h / γ_dcm / ‖P‖ trajectories → cross-model figures
- [ ] `scripts/collect_results.py` (copied and re-pointed) with **band-vs-band gates from day 1** — the seed-42 lesson encoded in the tooling, not in the analyst's memory
- [ ] Cross-model agreement table (§6.10) and the ERF-law verdict — **the benchmark's closing statement**
- [ ] Thesis text: adaptation map (§1), the four contribution verdicts and the considered-and-rejected table (§2.5), prior-art positioning (§2.6, Kaggle 2023 SegFormer solutions), deviations from the published SegFormer3D **including the paper-vs-repo recipe discrepancy and the irreproducibility note**, the ladder figure with the params-constant note, Family-D extension (S-III)

---

## 10. Appendix A — reference configs (copy-paste ready)

### A.1 `configs/sf_base_t192_s42.yaml` (the screen reference; every S1 cell is a one-key delta from this)

```yaml
# SegFormer3D-Ink screen reference (Phase S1). Deltas from this file define Table S-I.
model:
  name: segformer3d
  feature_size: 16          # == backbone.out_channels (asserted by smoke_test)
  capacity: S               # SF-S = 4,437,568 params (measured)
  stem: dp                  # depth-preserving (3,7,7)/s(1,4,4)/p(1,3,3); 'iso' = A1
  sr: "144_122"             # (1,4,4)/(1,2,2)/1/1; '444_222' | '244_222' = A2
  final: expand             # ConvTranspose3d(256->16, k=s=(1,4,4)); 'trilinear_lowres_pred' = A5
  skip_stem: true           # full-res conv skip (UNETR++/nnFormer parity); A4 toggles
  drop_path: 0.0            # reference default; overfit-response rung 2/3
  dw_norm: bn               # repo's added BatchNorm3d after the Mix-FFN DWConv; A3 = 'none'
  dec_proj_ln: true         # repo's added decoder LayerNorm;                A3 = false
  dec_dim: 256              # A7 = 128
  scale_mode: head_dim      # A11 = 'heads' (launch-commit convention)
  pdk: false                # C1
  dcm: false                # C2
  sr_overlap: false         # C3
  dsm: false                # C4
shared:
  texture_stem: false
  collapse: surface_relative   # A6 = 'mean'
data:
  tile: 192
  depth: 32
  center_layer: 32
  n_layers: 65
  normalize: 255.0
  label_file: ink_reg.npy
  manifest_suffix: _reg_t192
  pos_thresh: 0.05
  samples_per_epoch: 4000
  use_uncertainty: true
  halo: {enabled: false, factor: 2}   # C1 data carrier
  aug: {hflip: true, vflip: true, rot90: true, z_flip: false, z_flip_p: 0.5,
        z_shift: 2, z_shift_p: 0.3}
train:
  loss: dice_bce_masked
  lr: 0.0001                # A12 = 5.0e-05
  weight_decay: 3.0e-05     # A13 = 1.0e-02
  epochs: 60
  warmup_epochs: 3
  batch: 4
  num_workers: 8
  amp: true
  amp_dtype: bf16
  grad_clip: 1.0
  val_tau: 0.4
  select_on: sweep
  sweep_every: 1
  patience: 20
  seed: 42
infer:
  stride: 96                # t/2
  batch: 4
output:
  dir: /home/woody/iwi5/iwi5401h/SegFormer3D/outputs   # == INK_OUTPUT_ROOT in the sbatch
  run_name: sf_base_t192_s42
```

### A.2 Deltas for the key cells

| run | delta from A.1 |
|---|---|
| `sf_base_t{64,128,256,320}_s42` | `data.tile`, `data.manifest_suffix` (`_reg` / `_reg_t128` / `_reg_t256` / `_reg_t320`), `infer.stride` = t/2, `output.run_name` |
| `sf_base_d96_t320_s42` | tile 320, `manifest_suffix: _reg_t320_d96`, stride 160 |
| `sf_s_d64_t256_s42` | tile 256, `manifest_suffix: _reg_t256_d64`, stride 128 |
| `sf_m_*` / `sf_l_*` | `model.capacity: M` / `L` |
| `sf_pdk_t{128,192,256}_s42` | `model.pdk: true`, `model.pdk_stages: [2,3,4]`, `model.pdk_grid: [8,8,8]`, `data.halo.enabled: true` |
| `sf_pdkctl_t128_s42` | as `sf_pdk_t128_s42` plus `data.halo.mode: self` (halo = 2× downsampled copy of the tile itself) |
| `sf_dcm_abs_t192_s42` / `sf_dcm_surf_t192_s42` | `model.dcm: true`, `model.dcm_frame: absolute` / `surface` |
| `sf_osr_t192_s42` | `model.sr_overlap: true` |
| `sf_dsm_t192_s42` | `model.dsm: true`, `model.dsm_dilation: 2` |
| `sf_c1_t256_s{42,43,44}` | tile 256 + best density + gate-passing contributions + `train.seed` |
| `s0_sanity_t64` | tile 64, `_reg`, `epochs: 2`, `warmup_epochs: 1`, `batch: 16`, `samples_per_epoch: 2000`, `infer: {stride: 32, batch: 32}` |

### A.3 sbatch template (copy `$WORK/nnFormer/scripts/slurm_final_a100.sbatch` and change only the marked lines)

```bash
#!/bin/bash -l
#SBATCH --job-name=sf_s1
#SBATCH --gres=gpu:a100:1          # A40 fallback: --gres=gpu:a40:1 with --time=24:00:00
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0-11               # VERIFY THIS SURVIVES ANY sed EDIT (12 cells)
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module load python; module load cuda
SF_DIR="$WORK/SegFormer3D"; BENCH_DIR="$WORK/vesuvius/vesuvius-ink-bench"
cd "$BENCH_DIR"; source "$WORK/vesuvius/.venv/bin/activate"
export INK_OUTPUT_ROOT="$SF_DIR/outputs"; mkdir -p "$INK_OUTPUT_ROOT"
RUNS=(sf_base_t192_s42 sf_stemiso_t192_s42 sf_sr444_t192_s42 sf_sr244_t192_s42 \
      sf_nonorm_t192_s42 sf_noskip_t192_s42 sf_finlow_t192_s42 sf_mean_t192_s42 \
      sf_dec128_t192_s42 sf_deep_t192_s42 sf_lr5e5_t192_s42 sf_wd1e2_t192_s42)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}; CFG="$SF_DIR/configs/${RUN}.yaml"
set -e
python -m py_compile engine/train.py engine/infer.py data/dataset.py models/wrapper.py \
    "$SF_DIR"/src/segformer3d_ink/*.py
python scripts/smoke_test.py --config "$CFG" --device cpu
srun python -m engine.train --config "$CFG"
python -m scripts.postprocess --run "$RUN"
python scripts/plot_train_curves.py --run "$RUN"
TAU=$(python -c "import json;print(json.load(open('$INK_OUTPUT_ROOT/$RUN/summary.json'))['best_tau'])")
python scripts/visualize.py --prob "$INK_OUTPUT_ROOT/$RUN/val_prob_best.npy" \
    --seg 20241113090990 --label-file ink_reg.npy --tau "$TAU" --downsample 2 || true
```

---

## Appendix B — verification record (what in this plan is measured, and what is not)

Written 2026-09-10. The distinction matters because chapter 2's central methodological lesson was that an unverified number survived four documents before it was caught.

**Independently re-derived from first principles and confirmed exact** (a closed-form parameter/FLOP model of the architecture spec in §3, reproduced to the digit):
- Every parameter count in §3.2: SF-S **4,437,568** (encoder 3,957,408 / decoder 396,800 / final 83,360), SF-S with `sr 444_222` 4,568,640, SF-M **19,966,880**, SF-L **79,353,344**, and — the key control — the reference configuration reproducing the released file's **4,491,809** exactly.
- Every GFLOPs figure in §3.1/§3.2: 202.4 / 497.5 / 1054.0 (`144_122` at t192/t256/t320), 131.7 / 274.1 / 508.6 (`444_222`), 120.5 (isotropic stem, t256), SF-M 898.1, SF-L 2212.6.
- The contribution arithmetic: DCM 17,408 + 2,048 = **19,456**; OSR +49,152 per stage on `144_122` (+458,752 / +229,376 on `444_222`); DSM 7,424; the OSR output-grid identity `(n + 2·(r/2) − 2r)/r + 1 = n/r`.
- **The cube-root defect**: t256 is the *unique* tile in the ladder whose stage-1 token count is a perfect cube (8·64·64 = 32,768 = 32³); t64/t128/t192/t320 give 2,048 / 8,192 / 18,432 / 51,200, none of them cubes. This is why the reference code raises at four tiles and *silently corrupts geometry* at the fifth — the deployment tile.

**Verified against the live artifacts on disk:** UNETR++ ERF r50/r90 (24/35, 41/68, 68/105, 81/134), its error anatomy (unexplained 71.58%, boundary 13.49%, FP 150,890 px of which 114,867 in 342 islands), its frequency band-stop (8–16 px −0.0773, 2–4 px −0.0005, 128–256 px −0.3147), the tile ladder and the t320 turn; nnFormer's benchmark rows, the SR-PB band-vs-band Δ +0.0065, `n_srpb_aux` 0.7607, vFFC 0.7844/0.7992, the capacity span and the best-epoch-37 overfit signature, its ERF and Inkalyzer values, its error anatomy (72.1% unexplained, 190,433 FP px in 514 islands); every manifest tile count; the measured SLURM wall times; the bench harness contract (`smoke_test.py`'s two assertions and its 1-channel forward; `engine/train.py` not reading `INK_OUTPUT_ROOT`; every analysis-script flag used in Phase S4; the `data/tiling.py` CLI; the Inkalyzer's tile-divisible-by-16 requirement).

**NOT independently re-verified in this pass — check before quoting in the thesis:** the SegFormer3D paper's own published figures (4.51M / 17.5 GFLOPs) and its stated training recipe; the released repo's BraTS config values; the Kaggle 2023 placement scores and architectures in §2.6; and the claim that no published SegFormer3D application to Herculaneum data exists. These come from a primary-source reading whose working notes were lost with the session scratch directory. **They are cited, not measured, and none of them gates an experiment** — but §1's paper-vs-repo discrepancy table and §2.6's prior-art numbers should be re-checked against the arXiv page and the repository before they enter the written thesis.

---

*End of plan. Every run name, config key, code touchpoint, decision threshold, gate, pre-registered prediction and fallback above is fixed. An executing agent should proceed top-to-bottom: Phase S0 (port + inertness proof) → S1 (adaptation screens, A2 and A1 first) → **S1b (ladder + ERF — the power check that decides whether C1 runs at all)** → S1c → S2 → S3 → S4 → S5.*
