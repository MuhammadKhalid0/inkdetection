# PLAN — Benchmark Model 2: nnFormer for Ink Detection (with contributions)

**Project:** MSc thesis, ink detection in carbonized papyrus (Scroll 5 / PHerc 0172, Unlost project)
**Written:** 2026-07-19 (rev. 2 — contributions integrated from `nnformer_implementation_plan_v2 (1).md`) · **Status:** ready for execution
**Prerequisite reading:** `UNTER_Plus_Plus_Final.md` (results + protocol), `plan.md` §0/§10 (ops rules), `vesuvius_methodology_plan (1).md` §4 (nnFormer source config), `nnformer_implementation_plan_v2 (1).md` (contribution proposals evaluated in §2 below), `vFFC_Quattrini2023_summary.md` (prior art evaluated in §2.4)
**Role in the thesis:** second entry of the 3-model benchmark (UNETR++ ✅ → **nnFormer** → SegFormer3D), carrying **two nnFormer-native contributions**: CSSA (cross-scale skip attention — the chapter centerpiece) and SR-PB (surface-relative position bias — the cross-architecture geometry claim). The benchmark contract is: *identical data, splits, labels, recipe, inference, post-processing, metrics, probes; only the backbone changes; contributions are ablation-isolated on top.*

---

## 0. What the UNETR++ study already decided for us (binding evidence)

Every design choice below is derived from a **measured** UNETR++ result, not from the medical-segmentation literature. This section is the audit trail.

| # | UNETR++ evidence | Consequence for nnFormer |
|---|---|---|
| E1 | Context scale is the master lever (+0.16 t64→t256), **saturating at t256**; t320 overfit (112M params on 13.3k tiles: val peaked e27 while train loss kept falling) | Deploy at **t256**; treat **capacity-per-tile** as the central nnFormer question (150M-class model = t320 territory). Never run t320. Context is also the target of contribution **C2/CSSA** (§2.2). |
| E2 | Params↔tokens confound: UNETR++ params grew 25M→79M→112M with tile (Linformer projections + PEs) | nnFormer's window attention has **tile-size-invariant parameters** — its tile ladder is *cleaner* than UNETR++'s. State this as a benchmarking advantage and report measured params per rung anyway. |
| E3 | Fixed loop + swept-τ selection is non-negotiable (fixed-τ under-reported by up to 0.163; broken cosine generated fake architecture wins) | Identical training loop, 60-ep cosine, patience 20, per-epoch τ-sweep selection, `select_on: sweep`. No exceptions. |
| E4 | Noise floor: single-seed \|Δ\| ≤ 0.008 is noise; seed 42 was the weakest UNETR++ seed | Same discipline: seed-42 screens only kill or promote; all claims = 3-seed mean ± sd (42/43/44). Applies to the contributions **without exception**. |
| E5 | Adopted pipeline levers: registered labels (`ink_reg.npy`), u1 ignore-band (`dice_bce_masked` + trimaps, +0.012), z_flip OFF (+0.009/+0.011), TTA (+0.0035), component min-size filter (+0.010…0.020) | nnFormer's **final config inherits all of them from day 1** — the aim is to not regress below 0.819 post, so the model competes at full pipeline strength. |
| E6 | Class-A shared layer at deployment scale: texture stem **absorbed** (neutral ≥ t192; freq probe: model reads ≥128 px morphology, not 2–8 px crackle), collapse = **surface_relative tile-level** (the lone architectural survivor, +0.009 at t192; a centred-soft-window *prior*, fading but free at t256) | Stem stays OFF (kept as one t64-only ablation cell for the cross-model Family-D matrix). Collapse = surfrel tile-level (per-pixel + warp-stab measured neutral → off). `DepthCollapse` computes its own ŝ from the feature volume → transfers to any backbone unchanged. |
| E7 | DA-EPA (depth-attention gate) neutral at every scale. Geometry mechanism fully resolved: unsupervised, ŝ never learns (r≈0.2) and φ_rel never leaves init — the family works as a **centred prior** that still buys offset robustness (+0.05…+0.095 at \|δ\|≥4, t192); supervised (aux-ŝ), it learns (r=0.907) but **redistributes centre→tails at zero aggregate gain** (0.596 = 0.596) | Do **not** port a parallel depth-attention branch. The depth-frame question enters nnFormer only through **C1/SR-PB** (§2.1), evaluated primarily on the **stress curve**, with the pre-registered expectation "neutral at δ=0, value in the tails" — the measured signature of the geometry family. |
| E8 | Loss engineering dead (Tversky/recall/precision arms ≤ noise, twice); intensity normalization (photometric/CLAHE/z-score/detrend) all harmful; raw /255 correct | Loss = `dice_bce_masked` (u1). No photometric aug, no normalization variants. These are anti-goals. Also caps expectations for any deep-supervision variant (§2.3). |
| E9 | Error anatomy: residual FN mass is **confidence on faint ink** (32% near-miss p.5–.75, 49% faint, only 18.7% blind); ERF ≈ 53% of tile at every rung; frequency occlusion: ≥128 px structure dominant (−0.31), crackle ≈ 0 | The benchmark readout is **per-stratum recall**, not only F0.5. The ERF/frequency pair is also the direct motivation for **CSSA**: the model demonstrably wants letterform-scale context and consumes all it gets. |
| E10 | 60-epoch budget confirmed (e90 < e60 same-seed); LR 1e-4 fine for UNETR++; EMA neutral; per-segment centering parked; word-box sampling rejected (distribution mismatch); z_shift ±6 rejected | Reuse the recipe wholesale; the only recipe screens justified for a new backbone are **one LR point** (5e-5) and **drop-path** (nnFormer-native knob, the pre-planned overfit response). |

**Predictions to pre-register (write in the lab notebook before any GPU run):**
1. Pipeline ≫ architecture ⇒ the capacity-matched stock nnFormer lands **within ±0.01 of 0.8190 post**.
2. nnF-F (150M-class) at baseline data density shows the t320 overfit signature (best epoch < ~35, train/val divergence).
3. **CSSA:** closes ≥ half of the stock t128→t192 gap at t128 (the make-or-break prediction, §2.2); at t256 it is ≈ neutral (saturation there is data/capacity-limited, not perceptual).
4. **SR-PB:** neutral at δ=0 (\|Δ\| ≤ 0.008); improves the frozen-τ stress tails at \|δ\| ≥ 4 — the surfrel-prior signature, reproduced in a second architecture family.
5. **vFFC bottleneck swap (§2.4):** neutral-to-small vs GV-MSA (\|Δ\| ≤ 0.008 most likely) — global context is already available at the bottleneck (ERF), and Quattrini et al.'s own margin over a no-bottleneck baseline was +0.01. The three-way bottleneck study is run for the *mechanism-class* comparison (attention vs spectral conv), not because a win is expected.

---

## 1. Medical → ink adaptation map (every assumption, explicitly)

nnFormer [Zhou23] is an interleaved conv+attention U-shape built for multi-organ CT/MRI: 3D in → 3D softmax out, nnU-Net trainer, ~150.5M params / 213.4 GFLOPs at 128³-class inputs. Each medical assumption and its ink replacement:

| Medical assumption (source) | Ink reality | Adaptation |
|---|---|---|
| 3D volume in → **3D** multi-class mask out | 3D surface volume in → **2D** binary ink map out | Backbone returns a full-resolution **feature volume** `[B,16,D,t,t]`; the shared Class-A `DepthCollapse` (surface_relative) + shared `Head` (3×3→GELU→1×1→1 logit) produce `[B,1,t,t]`. Identical wrapper contract as UNETR++ (`models/wrapper.py`). |
| Near-isotropic anatomy; embedding halves depth (`H/4×W/4×D/2`) | Signal is a **thin band at z27–33** (Inkalyzer: every model's attribution is a unimodal surface bump; edge layers carry ≈0) | **Depth-preserving embedding**: 4× 3×3×3 conv stem with strides (1,2,2)+(1,2,2) → stage-1 grid `(32, t/4, t/4)`. Depth is only downsampled *inside* the hierarchy (32→16→8→4), mirroring the UNETR++ ink port (patch_size (1,4,4)). Ablated (N-I). This also protects SR-PB: a depth-frame mechanism is pointless if depth is crushed to 2 tokens before attention sees it. |
| Dataset-specific strides/windows, **hardcoded** to medical crop sizes in the official repo | t ∈ {64…256}, D=32 | All token counts, window sizes, and relative-position-bias tables **recomputed from config** — the same class of port fix that made UNETR++ valid (§1.2 port fixes: the published repos silently mis-size on non-medical shapes). |
| Deep supervision at 3D decoder scales, bilinearly downsampled GT | 2D labels; DS was removed from UNETR++ as structurally incompatible (deep dive §4.3.1) and replaced by a full-res conv skip | **Default OFF for all benchmark models — a benchmark-fairness decision, stated explicitly in the thesis methods** (v2 §5 is right that a silent DS-on/DS-off asymmetry would confound the headline). Replaced by the same full-resolution conv skip as UNETR++ for parity. A single gated **stroke-aware** DS screen exists as an nnFormer-only ablation excluded from the cross-model headline (§2.3). |
| nnU-Net recipe: SGD, poly LR 0.01, CE+Dice softmax, aggressive spatial aug | Benchmark protocol; measured-dead loss/aug variants | AdamW 1e-4, wd 3e-5, warmup-3 + cosine-60, `dice_bce_masked` (u1 trimaps), aug = hflip/vflip/rot90 + z_shift ±2, **z_flip OFF**. Deviation from the original recipe is stated in the thesis and justified by protocol identity (methodology §4.4). One LR screen at 5e-5. |
| Synapse/ACDC pretrained checkpoints | Shapes incompatible after adaptation (embedding, windows, head); UNETR++ ran from scratch | **From scratch** (benchmark-equalized initialization; methodology §2.6 arm (a)). Documented as such. (v2 §7 proposed a Synapse-init toggle on B-full; after the window/stride/embedding adaptation almost no tensor shapes survive, so the "pretrained" arm would be mostly-random anyway — cut, documented as a deviation.) |
| Intensity: CT windowing / z-score | Raw /255 is the measured optimum (E8) | Unchanged `/255`; depth window `[16,48)`, center_layer 32. |

### 1.1 What stays *exactly* the nnFormer identity (the architecture under test)

1. **Convolutional embedding** (4× 3×3×3 convs, GELU+LayerNorm) — kept.
2. **LV-MSA** (local volume-based MSA, Swin-style windows, 3D relative position bias) in encoder/decoder high-res stages — kept.
3. **GV-MSA** (global volume MSA) in the bottleneck — kept (ERF says the model is a context maximalist → pre-register "GV helps or is neutral, removing it should not help"; ablated in N-I).
4. **Skip attention** (encoder features as K/V, decoder as Q) instead of concat skips — kept; this is nnFormer's signature component, it gets the "signature ablation" treatment that DA-EPA got in UNETR++, **and it is the mechanism CSSA extends** (§2.2).
5. Interleaved conv down/up-sampling — kept.

---

## 2. Contributions — verdict on the v2 proposals

The three proposals in `nnformer_implementation_plan_v2 (1).md` were written before the final UNETR++ rounds. Before adopting them, the evidence base they cite has to be corrected; the verdicts follow from the corrected base.

### 2.0 Corrections to the v2 evidence base (each one changes a design decision)

| v2 premise | Measured status (July 2026 finals) | Effect on the proposals |
|---|---|---|
| "One segment peaks at z=42; per-segment surface drifts" | **Documentation error.** All 8 train segments peak z28–30, val z30 (verified against processed volumes); z_shift ±2 covers the true drift. | C1's *within-corpus* motivation is weak — there is no depth-offset pressure in this data (which is exactly why UNETR++'s ŝ never learned). C1's honest regime is the **synthetic-offset stress test** and cross-scroll deployment, not δ=0 F0.5. |
| "Labels misregistered 20–54 px → absolute depth frame is wrong" | The 20–54 px misregistration is **in-plane** (dy along y), and it is already fixed **data-side** by registration. It is not a depth-frame problem. | Removes the strongest-sounding line of C1's motivation; what remains (band drift under deployment shift) is real but lives in the stress/cross-scroll regime. |
| "Tile 256 assessed and abandoned (batch collapses to 2); t192 is deployment" | **t256 was run and adopted** (+0.029, 3 seeds, batch 4; saturation at t256; t320 overfit). | C2's framing "the tile mechanism does not scale" is overstated — it scaled to 256. The *surviving* motivation for CSSA: context enters at quadratic memory cost, saturation at t256 was data/capacity-bound, and the measured t128→t192 gap (+0.055) is the clean target for the 2×2. |
| "UNETR++ adapted ≈ 23M params" | 25M at t192, **79.1M at t256** (params scale with tokens, E2). | Capacity-matching target for nnF-M is ~79M-class, not 23M. |
| "Shared layer = texture stem + learned collapse" | Stem **absorbed** at ≥t192 → OFF at deployment; collapse = surface_relative (prior). | Contribution baselines run stem-OFF; b0-style "stem + learned collapse" cells are not built. |
| "DS on (nnFormer) vs DS off (UNETR++) = fairness leak; resolve to off-for-all" | **Correct — adopted.** Already the default in this plan; now stated as an explicit protocol decision in the thesis methods. | C3's fairness half survives as protocol; its architecture half is demoted (below). |
| Geometry family works by learned surface tracking | **Refuted:** unsupervised ŝ r≈0.2, φ_rel ≡ init; it is a **centred prior** that nevertheless buys the stress tails; supervised ŝ learns (r=0.907) but redistributes centre→tails, zero aggregate. | Any new geometry mechanism must be designed and *evaluated* as a prior/robustness device, and its "learned" variant must expect the centre-vs-tails trade. |

### 2.1 C1 — Surface-Relative Position Bias (SR-PB): **ADOPT, reduced scope + corrected mechanism**

**Verdict: worth one screen + the stress readout — not worth the v2 three-tier ladder.**

**A necessary technical correction first.** The v2 mechanism — "re-anchor each token's depth coordinate on ŝ before the relative-offset lookup" — is a **no-op as stated**: the relative position bias is indexed by pairwise differences Δz = z_i − z_j, and subtracting a common ŝ from both tokens cancels exactly (Δ(z−ŝ) = Δz). A per-window ŝ has the same cancellation *within* each window, which is where the bias acts. The working formulation is:

> **SR-PB = an additive, per-token, surface-relative depth bias on the attention logits:** each token contributes b(z_i − ŝ) (learned table over binned surface-relative depth, shared per stage/head), added to its attention logits as a key-side bias alongside the standard pairwise table. ŝ = soft-argmax over a mean+std depth descriptor (std because the cue is textural energy), **zero-init** so b ≡ 0 and ŝ = window centre at step 0 → exact reduction to stock nnFormer. This is the depth analogue of the surfrel collapse prior, moved *inside* attention: keys in the ink band get boosted relative to keys outside it, in a frame that follows ŝ.

**What the evidence keeps and what it cuts (vs the v2 tier ladder):**
- **Keep b5a (tile-level ŝ), one seed-42 screen.** Cheap, exactly-reducible, and it completes the cross-architecture geometry claim: *the same physical fix, expressed in each family's native mechanism (collapse prior in UNETR++, attention bias in nnFormer), shows the same signature*. That claim is about the domain, not one trick — the strongest thing the geometry chapter can say (v2 §10 is right about this).
- **Cut b5b (per-window ŝ).** Its UNETR++ analogue — the per-pixel surface field — was measured neutral at every scale, and v2's own honest-risk note (windows are small; a window rarely spans a large offset) points the same way.
- **Cut b5c (surface-aligned window resampling).** It is the warp machinery in new clothes: the UNETR++ warp was neutral at every scale, needed stabilization to not collapse, and b5c additionally breaks the efficient window partition. Measured-dead analogue + real implementation risk + no supporting evidence = not run.
- **Gated extension `n_srpb_aux`** (SR-PB + the label-free aux-ŝ Huber supervision, weight 0.1 — the one measured way to make ŝ actually track, r=0.907): run **only if** the SR-PB screen shows a stress-tail gain ≥ +0.02 at \|δ\| = 6, and pre-register the expected centre-vs-tails trade (the aux_shat result). This mirrors the G5 gating logic.

**Pre-registered evaluation (this is the part that makes the null informative):** primary readout = **G1 frozen-τ stress curve** (δ ∈ [−8,+8]) of `n_srpb` vs `n_base`, not δ=0 F0.5. Expected: δ=0 within noise; tails at \|δ\| ≥ 4 improved. If the tails do *not* improve either, the finding is "the depth-offset fix pays in proportion to the spatial extent the mechanism covers" (v2's own fallback reading) — reportable, and consistent with windows being depth-thin (4 layers).

### 2.2 C2 — Cross-Scale Skip Attention (CSSA): **ADOPT — the nnFormer chapter centerpiece**

**Verdict: the strongest proposal in v2, and the final UNETR++ evidence makes it stronger.** It targets the one lever measured to dominate everything (+0.16), and the interpretability rounds supplied its mechanism-level justification after v2 was written: the ERF shows the model consumes ~53% of *any* tile it is given (context is used, never saturated internally), and frequency occlusion shows the evidence is ≥128 px letterform-scale morphology (−0.31), not fine texture. A mechanism that hands fine-resolution decoding a view of the coarse (letterform-scale) encoder maps is aimed exactly where the measurements point. It is native to nnFormer (extends skip attention), impossible on UNETR++ (concat skips) — the right shape for a per-model contribution.

**Mechanism (unchanged from v2, house design rules applied):** decoder queries at resolution ℓ attend into K/V concatenated from their own scale plus the coarser encoder stages (S3, S4), each coarse scale's K/V scaled by a **zero-init learned gate γ_ℓ′** → at step 0 CSSA ≡ stock skip attention (exact reduction; graceful degradation; clean single-toggle ablation). Coarse stages contribute few tokens (S4 at t256 = 256 tokens), so the added cost is small — the point: letterform context at a fraction of the cost of tile growth. Report the trained γ_ℓ′ per stage (the γ_da lesson: a gate sitting at init is how "nothing to add" shows itself — plot it either way).

**The 2×2 payoff experiment (updated to measured scales).** Central falsifiable claim: *if CSSA supplies letterform context architecturally, CSSA@t128 approaches stock@t192.* The clean gap is **t128→t192 = +0.055** (UNETR++ lean; larger and cheaper than the +0.029 t192→t256 step — more sensitive test per GPU-hour).

- Cells: {stock, CSSA} × {t128, t192}, on nnF-S. Stock cells come free from the N-series ladder.
- **Confound control (the t256-trial lesson):** all four cells at **batch 4** → identical steps/epoch (1000) across tiles, removing the optimizer-budget confound from the ladder.
- Screen at seed 42 (2 new runs); **promotion rule:** if CSSA@128 closes ≥ half the stock 128→192 gap (i.e., ≥ +0.02 raw over stock@128), promote all four cells to 3 seeds (8 further runs) — the 2×2 is claimed only at 3 seeds.
- **Secondary, pre-registered:** CSSA@t256 expected ≈ neutral (saturation there is data/capacity-bound, E1) — so CSSA's value proposition, if it works, is *efficiency* (t192-class F0.5 at t128 memory) plus whatever it adds at deployment scale, and CSSA enters `nnf_c1` only if its t256 screen reads ≥ +0.008 (the stem lesson: adopt at screen scale ⇒ confirm at deployment scale first).
- **Null-result reading (pre-written):** γ→0 or no gap closure ⇒ "letterform context must enter at input resolution; it cannot be retrieved from coarse features after the fact" — which *strengthens* the tile-ladder finding. Either outcome is a headline sentence.

### 2.3 C3 — Stroke-Aware Deep Supervision (SA-DS): **CUT as a contribution; keep its two useful parts**

**Verdict: the fairness half is adopted as protocol; the architecture half is demoted to one gated screen.**

- **Adopted (protocol, zero GPU):** DS **off for all benchmark models** is the shared default, chosen explicitly and stated in the thesis methods — option (a) of v2 §5, which invalidates no completed UNETR++ work. This was already this plan's default; v2's contribution is making it an *on-the-record fairness decision* rather than a silent choice. Credited accordingly.
- **Demoted (one gated screen, `n_sads_t192_s42`):** if DS is tested at all, test the **stroke-aware form directly** (per-scale `DepthCollapse` + max-pooled registered-label targets at 1/4 and 1/8, masked loss, weights 1/0.5/0.25) — testing naive bilinear DS first would be a strawman, since the stroke-width argument (30–50 px strokes → 4–6 px at 1/8) already predicts its failure mode. Excluded from the cross-model headline by construction.
- **Why demoted:** the loss-engineering track record here is dead-on-arrival twice over (E8); training converges cleanly in 60 epochs with no gradient-path pathology for DS to fix; and the u1 masked loss already occupies the "smarter supervision" slot with a measured +0.012. Expected value is the lowest of the three proposals. Run only if time remains after C2 and C1 (v2 §12's own cut order agrees).

### 2.4 Prior art — vFFC (Quattrini et al. 2023, arXiv:2308.05070): **one ablation cell + three citable convergences; nothing else**

vFFC is the closest published prior art: same task family (Vesuvius fragment ink detection), same Fβ=0.5 metric, a 3D-encoder → bottleneck → depth-collapse → 2D-decoder template that matches our wrapper contract, and a Kaggle silver (Fβ ≈ 0.70 public / 0.60 private, 6-model ensemble — Scroll-1 fragments at 3.2 µm, **not comparable to our Scroll-5 0.819**). Their contribution is the **vFFC block**: a split local (3D conv) / global (3D-FFT → 1×1×1 conv in the frequency domain → iFFT) operator, used as a 3-block bottleneck. Verdict against our measured evidence:

**Why the mechanism is *not* dismissed by the texture-stem lesson (checked deliberately):** the stem died at scale because it engineered **fixed, fine-scale** (2–8 px) features — exactly the band the frequency probe priced at ≈0. vFFC's spectral branch is the opposite object: a **learnable, global** mixing operator whose leverage is at coarse/low frequencies — exactly the ≥128 px band the probe priced at −0.31, on the axis (whole-window context) the ERF says the model maximizes. So the mechanism is aimed at the measured evidence band, and it is a legitimate *mechanism-class alternative* to GV-MSA for the same job: global context at the bottleneck.

**What we take (cheap, evidence-compatible):**
1. **`n_vffc_t192_s42` — bottleneck swap:** GV-MSA (3 blocks) → vFFC residual blocks (3 blocks, vendored from `aimagelab/vffc`, param count measured and reported). Together with `n_lvonly` this upgrades the bottleneck ablation into a **three-way study {LV-only, GV-MSA (stock), vFFC}** = "does global bottleneck context matter at all, and does its mechanism class (attention vs spectral conv) matter?" — a cross-paradigm cell no other row provides, connecting the benchmark to the published prior art at the cost of one A40 run. Pre-registered expectation: neutral (prediction 5, §0).
2. **Channel dropout** (their strongest regularizer after dihedral aug) → added as a rung of the overfit response ladder (§6.3) — *ladder-gated, never default*.
3. **Three citable convergences (zero GPU, thesis text):** (a) their LayerCam depth analysis focuses at z≈30 — our Inkalyzer band is z27–33: two independent XAI methods, different models, different scrolls, same surface band (strengthens the trust result); (b) their "ink-weight > 1 hurts" loss ablation independently confirms our loss-engineering-dead verdict (E8); (c) their dihedral-transform ablation (largest single aug gain) matches our adopted D4 aug + 8-fold dihedral TTA.

**What we do NOT take (each conflicts with a measurement or a frozen decision):**
- **Their trained weights / outputs** — Scroll-1 detached fragments at 3.2 µm vs our Scroll-5 segments; wrong domain, and the pretraining arm is cut for benchmark equalization (§1).
- **vFFC as a fourth benchmark model** — the 3-model scope is frozen (and the conv-control row was explicitly cancelled by Omar, plan.md §8 note); vFFC enters as a *component cell inside nnFormer*, not a row.
- **Depth random-crop aug** (their depth-invariance lever) — its analogue was measured: z_shift ±6 lost −0.014 at δ=0 (g3), and the whole centre-vs-tails trade is already mapped by G1/aux-ŝ; their motivation is our stress regime, already covered. Cite as convergent motivation, do not re-run.
- **Majority-vote / k-fold ensembling** — probability ensembles measured harmful for the precision-weighted metric (0.702–0.715 < single best); their high-τ voting variant is not worth runs against that evidence.
- **Inner-crop inference** (keep 128² of 256²) — our Gaussian-blended overlapping sliding window is the measured, stronger version of the same border-artifact fix.
- **Their 16-slice window / OneCycle / 20-epoch recipe** — our D=32 window is validated by Inkalyzer + leave-one-out; the training protocol is frozen (E3).

**Port note (binding):** `torch.fft` does not run under bf16 autocast — the ST-3D block must be an **fp32 island** (cast in → FFT → 1×1×1 conv → iFFT → cast out) inside the autocast region, with `nan_to_num` on exit; added to the §3.4 checklist.

### 2.5 Scope guard (v2 §12, kept verbatim in spirit)

The benchmark is the thesis; the contributions must not consume it. Binding cut order if the schedule tightens: **first** `n_sads`, `n_srpb_aux`, and `n_vffc` (gated/optional extras), **then** CSSA promotion seeds beyond the two decisive cells (CSSA@128 ×3 vs stock@192 ×3 — the gap-closure claim needs these two bands most), **never** the stock benchmark row (Phases N2/N3), which ships regardless.

---

## 3. Architecture spec (nnFormer-Ink)

### 3.1 Resolution schedule (t256 deployment; divides cleanly at every tile in the ladder)

| Stage | Grid at t256 (D,H,W) | t192 | t128 | t64 | Attention | Window |
|---|---|---|---|---|---|---|
| Embed → S1 | (32,64,64) | (32,48,48) | (32,32,32) | (32,16,16) | 2× LV-MSA | **(4,8,8)** |
| ↓(2,2,2) → S2 | (16,32,32) | (16,24,24) | (16,16,16) | (16,8,8) | 2× LV-MSA | **(4,8,8)** |
| ↓(2,2,2) → S3 | (8,16,16) | (8,12,12) | (8,8,8) | (8,4,4) | 2× LV-MSA | **(4,4,4)** |
| ↓(2,2,2) → S4 | (4,8,8) = 256 tok | (4,6,6) = 144 | (4,4,4) = 64 | (4,2,2) = 16 | **3× GV-MSA** (global) | — |
| Decoder | mirrored, skip attention at S3/S2/S1 (**CSSA extends this**, §2.2), patch-expand to full res, final feature volume `[B,16,32,t,t]` | | | | | |

- Window sizes divide every stage grid at t ∈ {64,128,192,256} and D=32 (verified above) → **no padding paths**, one code path for the whole ladder, and the relative-position-bias tables depend only on the window → **params constant across tiles** (E2).
- Shifted-window alternation between the two LV blocks per stage (nnFormer/Swin convention) — kept.
- **Skip attention is windowed** (cross-attention within the same local volumes as LV-MSA). If the vendored code implements it globally, it would be O(N²) at S1's 131k tokens — the port must confirm and, if needed, window it; this is pre-decided (windowed), not a runtime discovery. CSSA's coarse K/V (S3/S4 tokens) are appended to each window's K/V set — few tokens, so the windowed structure survives.
- **SR-PB hook:** the additive surface-relative depth-bias table (§2.1) lives beside the pairwise bias table in the shared attention module used by LV-MSA / GV-MSA / skip attention — one implementation, three mechanisms (the one genuinely nice structural property v2 identified), all zero-init.

### 3.2 Capacity rungs (the "huge parameter count" question, made measurable)

Blocks fixed at [2,2,2]+3 GV; only widths change (v2 §7 agrees: reduce width, never remove stages — that would change the test from "capacity" to "architecture completeness"). Head dim 32 throughout. **Params below are MEASURED (2026-07-19, `nnFormer/tests/test_smoke.py`, t192 build; t192→t256 delta = GV bottleneck bias table only, asserted).** The earlier width-scaling estimates were recalibrated: the published 150.5M corresponds to embed_dim **192** (not 96), so presets were re-based to hit the three *roles* at measured counts.

| Variant | dims (S1→S4) | heads | **measured params** | Rationale |
|---|---|---|---|---|
| **nnF-S** | [64, 128, 256, 512] | [2,4,8,16] | **19.87M** | Data-scarcity-matched; ≈ UNETR++ t192 class (25M). Screen/ladder/contribution workhorse. |
| **nnF-M** | [128, 256, 512, 1024] | [4,8,16,32] | **79.11M** | Capacity-matched to deployed UNETR++ t256 (**79.1M** — exact match) — the *fair-comparison* rung |
| **nnF-F** | [192, 384, 768, 1536] | [6,12,24,48] | **177.74M** | Published-class capacity (paper's 150.5M is embed 192 with DS heads) — the "does 150M-class help or hurt" rung (methodology §4.7); expected to need the dense-data mitigation. Track the **train/val gap**, not just headline val (v2 §7's point, adopted). |

Drop-path (stochastic depth, nnFormer-native): 0.1 for S/M, 0.2 for F at baseline; it is the **first knob of the overfit response ladder** (§6.3), not a tuning playground.

### 3.3 Wrapper integration (mirrors the UNETR++ port exactly)

- New package `models/nnformer/` (vendored from the official repo, adapted like `models/unetrpp/`), exposing `build_nnformer_ink(in_ch, img_size=(D,H,W), dims, depths, window_sizes, num_heads, embed_depth_stride, skip_attention, cssa, sr_pb, gv_bottleneck, drop_path, deep_supervision) -> backbone` with `backbone.out_channels = 16` and `forward(x:[B,in_ch,D,H,W]) -> [B,16,D,H,W]`.
- `models/wrapper.py`: one new `elif name == "nnformer":` branch. Stem/collapse/head untouched — `DepthCollapse(mode="surface_relative")` computes its own ŝ from the feature volume (verified in `models/shared.py`; `last_s_hat` defaults to None via `getattr`), so **surfrel, the G2/φ_rel probes, and the aux-ŝ hook all work on nnFormer with zero changes**.
- **House design rule (every UNETR++ component obeyed it; the contributions obey it too):** every new component starts as an **exact no-op behind a default-off flag** (CSSA γ zero-init; SR-PB table zero-init + ŝ zero-init; `cssa: false` / `sr_pb: false` defaults) — that is what makes every ablation a genuine single toggle and lets param counts verify each rung.
- Config schema: same YAML families; `model.name: nnformer` plus the keys above; everything else (`shared`, `data`, `train`, `infer`, `output`) byte-identical to `c1_t256_s42.yaml`.

### 3.4 Port-fix checklist (each one bit UNETR++ once; check *before* first submit)

1. No hardcoded medical crop sizes / token counts anywhere — everything derived from `img_size` + config; assert shapes at build time for t ∈ {64,192,256}.
2. One declared axis convention (`[B,C,D,H,W]`, D=depth) with asserts at every reshape/window partition — the "axis trap" (UNETR++ block-internal H was depth).
3. Relative-position-bias tables sized from `window_size`, index tables built once per stage; **SR-PB's additive table sized from D per stage, and unit-tested for the exact-reduction property (all-zero table ⇒ bitwise-stock attention)**.
4. Norm layers: `_safe_num_groups`-style guard anywhere GroupNorm appears (the 11-channel stem incident); LayerNorm eps default.
5. bf16 autocast at t192+ (fp16 activations overflow from t192 up — measured); fp32 + `nan_to_num` in every soft-argmax chain (SR-PB's ŝ included — this exact guard was load-bearing in UNETR++). **If `n_vffc` is built: `torch.fft` does not run under bf16 autocast — the ST-3D block is an fp32 island (cast → FFT → 1×1×1 conv → iFFT → cast back) with `nan_to_num` on exit (§2.4).**
6. `python -m py_compile` + `scripts/smoke_test.py --config <cfg> --device cpu` (forward + one backward at t64 and shape-check at t256) before every `sbatch`; unique `run_name` per run; output root **`$WORK/nnFormer/outputs`** via `INK_OUTPUT_ROOT` (own tree per Omar 2026-07-19; vesuvius outputs* history never written); never touch `outputs*` history, val manifest, mask, or test labels.
7. Smoke test prints **measured params** per variant + per-tile; add a tiny bench mode for GFLOPs (thop/fvcore or manual) and ms/tile + peak memory on one A40/A100 batch — these fill the efficiency columns of the headline table. Memory check at batch 1 for nnF-F@t256 **before** committing the schedule (v2 §6, adopted).

---

## 4. Data strategy: density instead of shrinkage

The parameter problem is attacked from the **data side first**, architecture side second:

1. **Keep the full tile inventory.** Baseline manifests stay `_reg_t256` (stride 128, 21,360 train tiles / 875 val) for strict comparability with UNETR++.
2. **Dense-stride training manifests** (new): `python -m data.tiling --tile 256 --stride 64 --label-file ink_reg.npy` → `train_reg_t256_d64.csv`, ~4× tiles ≈ **85k**; optional middle rung stride 96 ≈ 38k (`_d96`). *Training-only*: the val manifest and sliding-window inference stride (t/2, Gaussian-blended) are protocol-fixed and untouched. `samples_per_epoch` stays 4000 — density buys **pool diversity per epoch** (the exact antidote to the t320 failure mode: same optimizer budget, more distinct views), not longer epochs.
3. **Augmentation: the measured-safe set only** — hflip/vflip/rot90 (p .5), z_shift ±2 (p .3), z_flip OFF, u1 trimap transformed with the same geometric augs. No photometric/intensity aug (measured harmful, anti-goal). Anything new enters only through the overfit response ladder (§6.3), as a gated single screen, never silently.
4. u1 trimaps (`proc/<seg>/unc.npy`, r=4) are per-pixel per segment — **reused as-is** for any tile/stride. The t128 CSSA cells need `train_reg_t128` (exists) — no new label artifacts.

Confound note for the thesis: denser stride raises tile *overlap* (75% at stride 64), so the 85k tiles are not 4× independent samples; the honest claim is "more distinct letter-crop views under the same optimizer budget." Report it next to the density ablation.

---

## 5. Run plan

### Phase N0 — port + verification (CPU + 1 short GPU sanity job)

- Vendor + adapt code, checklist §3.4, measured params/GFLOPs/ms table for S/M/F at t192/t256, exact-reduction unit tests for CSSA (γ=0) and SR-PB (zero table).
- 1-epoch GPU sanity run (`n0_sanity_t64`, not reported): loss decreases, τ-sweep/val/figs/postprocess pipeline runs end-to-end, memory headroom at t256 batch 4 (fallback: batch 2 + grad-accum 2 — same effective batch, pre-decided).

### Phase N1 — architecture + contribution screens (t192, seed 42, A40, ~14 h each)

t192 is the screen scale (cheaper; matches where the UNETR++ ablation suite ran). Base recipe for all N-runs: lean-t192-equivalent + the adopted pipeline (registered labels, u1 masked loss, no z_flip); reference = its own `n_base` band. Capacity: nnF-S for all screens.

| run | config delta | question (UNETR++ analogue) |
|---|---|---|
| `n_base_t192_s42` | nnF-S, surfrel collapse, skip-attn ON (stock), GV ON, embed depth-stride 1, DS OFF, CSSA/SR-PB OFF | the nnFormer reference point (lean) |
| `n_mean_t192_s42` | collapse: mean | Class-A shared layer off (a0 vs a2 axis) |
| `n_abs_t192_s42` | collapse: learned absolute | prior vs frame (a0s vs a2 axis) |
| `n_skipconv_t192_s42` | skip attention → plain concat skip | signature-component test (the abl_da analogue); also the baseline CSSA must beat |
| `n_lvonly_t192_s42` | GV-MSA → LV-MSA in bottleneck | does global bottleneck context matter? (pre-registered: removing it should not help, per ERF) |
| `n_vffc_t192_s42` — **optional** | GV-MSA → 3× vFFC residual blocks (§2.4) | mechanism class: attention vs spectral conv for the same global-context job; completes the three-way bottleneck study {LV-only, GV, vFFC} |
| `n_embedz2_t192_s42` | embedding depth stride (2,2,2)→ stage1 (16,·,·) | depth preservation (a0b analogue) |
| `n_lr5e5_t192_s42` | lr 5e-5 | the one recipe screen a new backbone earns |
| **`n_srpb_t192_s42`** | + SR-PB (tile-level ŝ, additive bias, zero-init) | **C1** — scored at δ=0 *and* fed to the G1 stress job (§Phase N4); primary readout = stress tails |
| **`n_cssa_t192_s42`** | + CSSA (zero-init γ, coarse K/V from S3+S4) | **C2** — the t192 column of the 2×2 |
| `n_sads_t192_s42` — **gated** | + stroke-aware 2D deep supervision (§2.3) | C3 remnant; run only if time remains; excluded from cross-model headline |
| `n_srpb_aux_t192_s42` — **gated** | SR-PB + label-free aux-ŝ (Huber, w=0.1) | only if `n_srpb` stress tails ≥ +0.02 at \|δ\|=6; expect the centre↔tails trade (aux_shat result) |

Decision rules (identical to UNETR++ rounds): \|Δ\| ≤ 0.008 vs `n_base` → parked/neutral (that *is* a Family-D result); Δ < −0.008 → component confirmed helpful (keep); Δ > +0.008 → base config amended before Phase N2 and noted. **Tile-interaction guard (v2 §8.4, adopted — the check that would have caught the DA-EPA artifact early):** any large screen gain on `n_skipconv`, `n_srpb`, or `n_cssa` triggers a reduced-tile (t64 or t128) check before it is written up as an architecture-level finding.

### Phase N1b — tile ladder (Family-A cross-model figure; also the stock 2×2 cells)

`n_base` at **t64, t128** (seed 42, batch 4 at t128 for 2×2 comparability; hours each; t192 exists, t256 comes from N2) → the 4-point nnFormer scale curve next to UNETR++'s 0.639→0.719→0.774→0.804. Because nnFormer params are tile-invariant (E2), this curve is free of UNETR++'s capacity confound — worth a sentence in the thesis. `n_base_t128` doubles as the stock@128 cell of the CSSA 2×2.

### Phase N1c — the CSSA 2×2 (the chapter's key experiment; pre-registered in §0)

| | t128 (batch 4, 1000 steps/ep) | t192 (batch 4, 1000 steps/ep) |
|---|---|---|
| stock skip attention | `n_base_t128` (N1b) | `n_base_t192` (N1) |
| CSSA | **`n_cssa_t128_s42`** (1 new run) | `n_cssa_t192` (N1) |

**Promotion rule:** CSSA@128 closes ≥ half the stock 128→192 gap (≥ +0.02 raw over stock@128) → all four cells to 3 seeds (8 runs; if the budget tightens, the two decisive bands first: CSSA@128 ×3 vs stock@192 ×3). Otherwise: report the screened null with the pre-written reading (§2.2) — no seeds spent.
**Deployment gate:** CSSA enters `nnf_c1` only after a t256 screen (`n_cssa_t256_s42`) reads ≥ +0.008 (the stem lesson: screen-scale wins must be confirmed at deployment scale).

### Phase N2 — capacity × data density (t256, A100; the headline nnFormer study)

| run | capacity | train stride (tiles) | question |
|---|---|---|---|
| `nnf_s_t256_s42` | S (~38M) | 128 (21k) | small-capacity anchor |
| `nnf_m_t256_s42` | M (~65M) | 128 (21k) | fair capacity match to UNETR++ (79M) |
| `nnf_f_t256_s42` | F (~150M) | 128 (21k) | published capacity on scarce data — expected overfit |
| `nnf_f_d64_t256_s42` | F | **64 (85k)** | does density rescue full capacity? |
| `nnf_<best>_d64_t256_s42` | best of S/M | 64 (85k) | does density help even matched capacity? (skip if best = F) |

Read each cell with **three numbers**: swept F0.5 (raw/post), best epoch, and the train-loss-vs-val divergence (the t320 overfit signature). Overfit diagnosis without the signature is not claimed.

### Phase N3 — final combo + seeds

`nnf_c1_t256_s{42,43,44}` = best capacity × best density × any N1 amendment (+ CSSA and/or SR-PB **only if** they passed their gates) + u1 + no z_flip; TTA + (τ, ms) min-size post at rescoring, exactly as c1. **This 3-seed band is the benchmark number** vs c1_t256 **0.8111 ± 0.0039 raw / 0.8190 ± 0.0062 post**. If a contribution is adopted here, the stock `nnf_c1` twin (one seed) is kept as the attribution control so the benchmark row and the contribution row stay separable.

### Phase N4 — the shared analysis layer (inference-only, 1–2 jobs)

On `nnf_c1` (all probes are model-agnostic, already written):
1. **Inkalyzer** depth attribution — does nnFormer also lock onto z27–33? (trust result + cross-model figure).
2. **ERF** — is nnFormer also a ~53%-of-tile context maximalist, or does windowed attention bound it? **With CSSA on vs off, the ERF comparison is the mechanism figure for C2** (did cross-scale K/V visibly extend the effective field at fixed tile?).
3. **Frequency occlusion** — same morphology-vs-crackle reading.
4. **Error anatomy** — **per-stratum recall (near-miss / faint / blind) vs UNETR++'s 32/49/18.7%** — the pre-declared axis on which a competitor can "win" at equal F0.5 (E9).
5. **G1 z-shift stress** (δ ∈ [−8,+8], frozen-τ + resweep) — `n_base` vs `n_srpb` (**C1's primary readout**) vs `n_mean` (collapse-prior contribution isolated), extending the geometry chapter's stress figure cross-model. γ_cssa and SR-PB bias-table norms plotted over training (the γ_da lesson).
6. IoU backfill columns (`rescore_iou.py` covers new runs) + val 4-panel figs + test-strip qualitative (τ 0.65–0.70, ms ≤ 10k, always show the raw map).

### Budget

≈ 11 N1/N1b screens (A40, ≤14 h; t64/t128 cheaper) + 1 CSSA t128 cell + up to 8 CSSA promotion seeds (gated) + 1 CSSA t256 screen (gated) + 5 t256 capacity/density runs (A100, ~14–24 h; F may need batch 2+accum) + 2–3 final seeds + ≤3 gated/optional extras (`n_sads`, `n_srpb_aux`, `n_vffc`) + 2 inference jobs ≈ **20–31 training runs ≈ 13–18 GPU-days**. Upper end only materializes if CSSA earns its promotion — the right reason to spend it. Wall-time fallbacks pre-decided (§3.4.7, §5 N0). Scope guard §2.5 binds.

---

## 6. Ablation tables for the thesis (templates, comparable to UNETR++)

### 6.1 Table N-I — nnFormer component study (t192, seed-42 screens vs `n_base`; promoted rows get 3 seeds)

| Component toggled | Run | raw F0.5 @τ | post | ΔF0.5 raw | Verdict |
|---|---|---|---|---|---|
| — (reference) | n_base_t192 | | | — | |
| Collapse: surfrel → mean | n_mean_t192 | | | | |
| Collapse: surfrel → learned-abs | n_abs_t192 | | | | |
| Skip attention → conv skip | n_skipconv_t192 | | | | |
| GV bottleneck → LV | n_lvonly_t192 | | | | |
| GV bottleneck → vFFC (prior art, §2.4) | n_vffc_t192 | | | | |
| Embed depth stride 1 → 2 | n_embedz2_t192 | | | | |
| LR 1e-4 → 5e-5 | n_lr5e5_t192 | | | | |
| **+ SR-PB (C1)** | n_srpb_t192 | | | | (δ=0; stress verdict in N-VI) |
| **+ CSSA (C2)** | n_cssa_t192 | | | | (full verdict in N-V) |
| + stroke-aware DS (gated; headline-excluded) | n_sads_t192 | | | | |
| + SR-PB + aux-ŝ (gated) | n_srpb_aux_t192 | | | | |

### 6.2 Table N-II — capacity × data density (t256; the "150M params on 9 segments" study)

| | stride 128 (21k tiles) | stride 64 (85k tiles) | best epoch | overfit signature? |
|---|---|---|---|---|
| nnF-S (~38M measured: __) | | | | |
| nnF-M (~65M measured: __) | | | | |
| nnF-F (~150M measured: __) | | | | |

Caption: params measured at t256; `samples_per_epoch` = 4000 in all cells (density changes pool diversity, not optimizer budget); overlap confound stated (§4).

### 6.3 Pre-registered overfit response ladder (applied in order, one step per re-run, only on cells showing the signature)

1. Dense stride (already in the matrix) → 2. drop-path +0.1 → 3. channel dropout 0.2 (the vFFC paper's regularizer, §2.4) → 4. weight decay 3e-5 → 1e-4 → 5. capacity rung down. Stop at the first step that clears the signature; report the path taken.

### 6.4 Table N-III — cross-model component × scale matrix (extends UNETR++ Family D)

| Component | UNETR++ t64 | UNETR++ t192 | UNETR++ t256 | nnFormer t192 | nnFormer t256 |
|---|---|---|---|---|---|
| Class-A collapse (surfrel vs mean) | +0.003 | **−0.009** | +0.008 raw / ±0 post | | |
| Texture stem | **−0.073** | +0.000 | absorbed | (one optional t64 cell) | — |
| Signature attention component (DA-EPA / skip-attn) | −0.004 | −0.004 | dropped | | |
| Depth-preserving embed | (a0b: neutral F0.5, depth-behaviour shift) | — | — | | |
| Depth-frame fix (DA-EPA / **SR-PB**) | −0.004 | −0.004 | dropped | | (δ=0 row; tails in N-VI) |

Message to test: *"one real component per regime, nothing everywhere"* — does it replicate on a second architecture?

### 6.5 Table N-V — the CSSA 2×2 (C2's centerpiece; 3-seed mean ± sd once promoted)

| | t128 | t192 | Δ(128→192) |
|---|---|---|---|
| stock skip attention | | | (the gap CSSA must close) |
| CSSA | | | |
| Δ(CSSA − stock) | | | |

Caption: nnF-S; batch 4 / 1000 steps-per-epoch in all cells (optimizer-budget confound removed); γ_ℓ′ trajectories in the appendix figure; pre-registered success criterion = CSSA@128 closes ≥ half the stock gap. Plus one line: CSSA@t256 screen result (deployment gate).

### 6.6 Table N-VI — geometry cross-architecture stress table (C1's evidence; frozen-τ F0.5 vs induced δ)

| Model / mechanism | δ=0 | \|δ\|=4 | \|δ\|=6 | \|δ\|=8 | mean over [−8,8] |
|---|---|---|---|---|---|
| UNETR++ lean (surfrel collapse prior) | 0.775 band | +0.03…0.05 vs OFF | +0.05…0.08 | +0.08…0.10 | (from outputs2/5) |
| UNETR++ aux-ŝ (learned geometry) | −0.050 | | +0.13 @−8 | | 0.596 = lean |
| nnFormer `n_base` (surfrel collapse only) | | | | | |
| nnFormer `n_srpb` (**+ SR-PB**) | | | | | |
| nnFormer `n_srpb_aux` (gated) | | | | | |

Caption: the cross-architecture geometry claim lives here — same physical fix, native mechanism per family, same signature (neutral centre, robust tails) or not. Either outcome is the chapter's closing statement.

### 6.7 Table N-IV — benchmark headline (grows one row per model)

| Model (final config) | F0.5 raw | F0.5 post | +TTA | P | R | IoU_ink | IoU_bg | mIoU | params (meas.) | GFLOPs | ms/tile | peak mem | recall: near-miss / faint / blind | mean F0.5 over δ∈[−8,8] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UNETR++ c1_t256 (3 seeds) | 0.8111 ± .0039 | 0.8190 ± .0062 | ~0.826* | | | 0.585 | 0.911 | | 79.1M | | | | .32/.49/.187 (FN shares) | |
| nnFormer nnf_c1_t256 (3 seeds) | | | | | | | | | | | | | | |
| SegFormer3D (future) | | | | | | | | | | | | | | |

*pending c1 TTA rescore. Caption: identical split/labels/recipe/inference/post/τ-protocol; τ and ms selected on val per model; IoU reported never selected on; ± = 3-seed sd; DS off for all models (protocol decision, §2.3); contributions (CSSA/SR-PB) included only where their gates passed, with the stock twin reported for attribution.

---

## 7. How this preserves the benchmark structure

Unchanged, byte-identical, shared: split (train 8 / val 090990 / test flatboi qualitative) · registered labels + masked eval · swept-τ selection + early stop · 60-ep cosine, warmup 3, AdamW 1e-4, wd 3e-5, grad-clip 1, bf16, batch 4 (accum fallback), `samples_per_epoch` 4000, 1:1 pos-tile oversampling (pos_thresh .05) · aug set · u1 trimaps + `dice_bce_masked` · sliding-window stride t/2 + Gaussian blending · 8-fold dihedral TTA · (τ, ms) component filter selected on val · seeds 42/43/44 + ±0.008 noise floor · IoU reporting at both operating points · probe suite (Inkalyzer/ERF/freqband/anatomy/G1) · figs/artifacts layout (`summary.json`, `postprocess.json`, `figs/`, `train_curves`) · ops rules (unique run names, smoke test + py_compile before sbatch, own output root `$WORK/nnFormer/outputs`) · **deep supervision OFF for every benchmark model (explicit protocol decision)**.

The *only* degrees of freedom the new model gets are (a) its own backbone hyper-parameters (capacity, windows — Tables N-I/N-II), (b) one LR point, (c) drop-path as its native regularizer, (d) the **gated, ablation-isolated contributions** (CSSA, SR-PB), each reducing exactly to stock at init and each excluded from the headline unless its gate passed with the stock twin reported — mirroring how DA-EPA/geometry sat on top of the UNETR++ benchmark entry without contaminating it. Anti-goals inherited verbatim from plan.md §12 (no loss engineering, no intensity normalization, no morphology beyond min-size, no t320, no LOFO, never touch val/test); newly added anti-goals from §2: **per-window ŝ (b5b), surface-aligned window resampling (b5c), naive-bilinear DS as a contribution, Synapse-init arm, vFFC as a fourth benchmark model, vFFC weights/outputs transfer, depth random-crop aug, majority-vote/k-fold ensembling, inner-crop inference** (§2.4).

---

## 8. Risk register (pre-made responses)

| Risk | Pre-planned response |
|---|---|
| nnF-F OOMs at t256 (GV-MSA + full-res decoder) | batch 2 + grad-accum 2 (same effective batch); if still OOM: final expand channels 16→8 for F only, reported |
| Skip attention global in vendored code → O(N²) at S1 | pre-decided: window it (same volumes as LV-MSA); if the port is nontrivial, S1 falls back to concat skip and skip-attn lives at S2/S3 only — documented, and N1's `n_skipconv` still isolates the mechanism |
| SR-PB implemented as pairwise re-anchoring | **it cancels (Δz is shift-invariant) — the no-op trap identified in §2.1**; implementation must be the additive per-token bias, and the smoke test asserts non-equivalence to stock for a nonzero table plus exact equivalence at zero |
| SR-PB's ŝ never moves (no gradient pressure at δ=0 — the measured UNETR++ outcome) | expected; that is why the primary readout is the stress curve, and why `n_srpb_aux` exists behind its gate |
| CSSA gate γ→0 (nothing retrieved from coarse scales) | the pre-written informative null (§2.2): "letterform context must enter at input resolution" — strengthens the tile-ladder finding; report the γ trajectory as evidence |
| CSSA helps at t128 but not t192/t256 | that *is* the tile-conditionality pattern (stem, DA) — report in the component × scale matrix; CSSA's efficiency claim (t192-class F0.5 at t128 cost) survives if the 2×2 supports it |
| All capacities overfit even at stride 64 | that *is* the finding (methodology §4.7: "150M under scarcity — underperforming is a result, not a failure"); report with the overfit-signature evidence and let nnF-S carry the benchmark row |
| nnFormer (or a contribution) beats c1 by > +0.02 | treat as suspect first (project history: surprising jumps are protocol bugs) — check label file, manifest, τ column, val never sampled, then celebrate |
| Windows don't divide a future tile size | §3.1 table is the contract; any new tile must satisfy it or gets padding + a code path audit |
| t192 screens contradict t256 behaviour (tile-conditionality, as with the stem) | components adopted at screen scale get one t256 confirmation cell before entering `nnf_c1` (the stem lesson; CSSA's deployment gate is exactly this) |
| Training instability (loss spikes, NaN) | bf16 already default ≥ t192; fp32 softmax + `nan_to_num` guards per §3.4; if persistent: LR 5e-5 arm becomes the base (the N1 screen doubles as the fallback) |
| vFFC ST-block NaNs / dtype crash under autocast | pre-decided fp32 island around FFT→conv→iFFT (§2.4, §3.4.5); if the vendored block resists clean porting within a day, `n_vffc` is dropped (optional cell, zero impact on the benchmark) and the three-way bottleneck study shrinks back to two |
| Contribution work crowds out the benchmark | scope guard §2.5: cut `n_sads`/`n_srpb_aux`/`n_vffc` first, then CSSA promotion beyond the two decisive bands; Phases N2/N3 ship regardless |

---

## 9. Deliverables checklist

- [ ] `models/nnformer/` port + smoke tests (incl. CSSA/SR-PB exact-reduction unit tests) + measured params/GFLOPs/ms table (Phase N0)
- [ ] `train_reg_t256_d64.csv` (+ optional `_d96`) dense manifests (CPU)
- [ ] Table N-I (component + contribution screens, t192) with verdicts
- [ ] Table N-V (CSSA 2×2) — screened, and promoted to 3 seeds if the gate passes
- [ ] Table N-VI (cross-architecture geometry stress table) with the SR-PB curves
- [ ] Table N-II (capacity × density, t256) with overfit diagnostics
- [ ] `nnf_c1_t256` 3-seed band, TTA'd + post-filtered → Table N-IV row (+ stock twin if a contribution was adopted)
- [ ] Probe artifacts: Inkalyzer profile, ERF (CSSA on/off), freqband, error anatomy (per-stratum recall), G1 stress curves, γ/bias-table trajectories → cross-model figures
- [ ] Thesis text: adaptation map (§1), v2-corrections + contribution verdicts (§2, incl. the SR-PB cancellation fix and the DS fairness decision on the record), deviations from published nnFormer + justification, ladder figure with the params-constant note, Family-D extension (Table N-III), cross-architecture contribution section (SR-PB vs DA-EPA/geometry family; CSSA vs the tile ladder)
- [ ] Prior-art section: vFFC (Quattrini et al. 2023) positioning + the three zero-GPU convergences (LayerCam z≈30 ↔ Inkalyzer z27–33; ink-weight>1 hurts ↔ E8; dihedral aug ↔ D4+TTA) + the three-way bottleneck cell result if `n_vffc` ran (§2.4)
