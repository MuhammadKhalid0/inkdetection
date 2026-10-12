# UNETR++ Ink Detection — Overall Verdict, Ablation Answers & Recommendations

**Synthesis of ALL model results (F0.5 ladder + §7 ablations + seeds) and ALL Inkalyzer XAI
results.** Prepared 2026-07-14. Two evidence streams are combined per component:
**(A) performance** = swept masked-F0.5 on VAL (reg labels), from the final report; **(B) XAI** =
Inkalyzer input-space IBA depth attribution (job 3856890, `Inklayzer_UNTER++.md`).

---

## 0. TL;DR verdict

The project's real result is a **ranking of levers, not an architecture win**. Ordered by measured
effect on F0.5:

> **tile size (+0.16) ≫ checkpoint-selection protocol (+0.03) > label registration (+0.013) ≈
> component filter (+0.015) ≈ learned refiner (+0.010) > surface-relative depth (+0.009) >
> everything else ≈ 0 (within seed noise).**

Every "clever" architecture piece except **surface-relative depth** is neutral at the deployed tile
size. The two things that are genuinely yours and defensible — **label registration** (a data-side
contribution) and the **XAI story** — both survive scrutiny and *reinforce each other*. The honest,
strong thesis is: **for cross-segment ink detection, data + scale + protocol dominate architecture.**

---

## 1. Every ablation question, answered

Sign convention: Δ = change vs the `a2bs_t192` reference (raw 0.7735 / post 0.7857, seed 42) **when
the component is removed/toggled**. Negative Δ ⇒ the component *helps*.

### Q1. Does the **texture stem** help? (`abl_stem`)
- **Performance:** Δraw +0.000 / Δpost **+0.007** at t192 → removing it is *neutral-to-slightly-good*.
  At t64 it was worth **+0.10** (a0→a0s). **Verdict: real at t64, neutral at t192 (tile-conditional).**
- **XAI:** a0→a0s *broadened* depth usage (surf% 58.6→50.9, deep% 22→33). The stem adds texture
  channels but does not sharpen depth focus.
- **Answer: NO at the deployed scale.** Keep only if you want the t64 story; drop for parsimony.

### Q2. Does **depth attention (DA-EPA)** help? (`abl_da`)
- **Performance:** single-seed Δraw **+0.014** looked like it *hurt*, but the **3-seed lean
  replication (§7.1) killed that** — mean Δ −0.004, i.e. within noise. **Verdict: NEUTRAL at t192.**
  At t64 it was positive (a0b→a1 = +0.05) and `a1` is the most surface-focused t64 model (below).
- **XAI:** `abl_da` vs `a2bs_t192` depth profiles are near-identical (entropy 0.907 vs 0.899, peak z31
  vs z32). Turning the depth branch off does **not** change where the model looks in depth.
- **Answer: NO at the deployed scale** (real but tile-conditional at t64). XAI corroborates neutrality.

### Q3. Does **surface-relative depth** help? (`abl_surfrel`)
- **Performance:** Δraw **−0.009** / Δpost **−0.006** → the **only architectural component that is
  clearly positive at t192.**
- **XAI:** *invisible* to the depth profile — `abl_surfrel` ≈ `a2bs_t192` (entropy 0.899=0.899, peak
  z32=z32). Its benefit is **per-pixel/spatial** (aligning the depth reference to the local surface),
  which a tile-averaged depth metric cannot see. Not a contradiction — a scope limit of the metric.
- **Answer: YES — keep it. It is the one architecture piece the data supports.**

### Q4. Does **per-pixel** surface warp beat tile-level? (`abl_perpix`)
- **Performance:** Δraw −0.000 / Δpost −0.003 → **neutral.**
- **XAI (t64 proxy a2→a2b):** per-pixel *diffused* depth attention (a2b is the most diffuse t64 model,
  entropy 0.868) with no F0.5 payoff.
- **Answer: NO measurable benefit over tile-level.** Keep tile-level for simplicity + memory.

### Q5. Does **warp stabilization** help? (`abl_warpstab`)
- **Performance:** Δ ±0.000 at t192 → **neutral on F0.5.** But it was the **engineering fix** that made
  t192 trainable at all in fp16 (before bf16); with bf16 its numerical role is moot.
- **XAI (t64 a2b→a2bs):** re-concentrated depth after per-pixel diffused it (entropy 0.868→0.834).
- **Answer: NEUTRAL for accuracy; historically load-bearing for stability. Keep, it's free.**

### Q6. Does **label registration** help? (`abl_labels`)
- **Performance:** Δraw **−0.012** / Δpost **−0.014** → **clearly positive**, one of the largest levers,
  and it is a *data-side* contribution independent of the model (+0.03…0.07 rescore across all models).
- **XAI (headline):** `abl_labels` (unregistered) is the outlier on **every** axis — deepest peak
  (z34 vs z32), lowest surface mass (45% vs 50%), most shallow leakage (22% vs 16%), highest entropy.
  Misaligned labels prevent the model from locking onto the surface, so it **hedges across depth.**
- **Answer: YES, strongly — and XAI gives independent, model-internal proof of *why*.** Best single
  finding in the whole study.

### Q7–Q8. Does **loss asymmetry** help (recall- or precision-weighted Tversky)? (`abl_loss_recall`, `abl_loss_prec`)
- **Performance:** Δ −0.001 (recall) and −0.005/+0.001 (precision) → **irrelevant in both directions.**
  The per-epoch τ-sweep absorbs the calibration shift; recall-leaning loss did **not** rescue the left
  word (consistent with a6 at t64).
- **XAI:** not separately imaged (loss changes the decision threshold, not the depth evidence).
- **Answer: NO. Loss engineering is a dead lever here; keep plain Dice+BCE + τ-sweep.**

### Q9. (Methodology) Do the DA/surfrel models become **more surface-concentrated & offset-robust**? (A0s vs A2/A2b)
- **XAI answer, nuanced:** *within t64*, yes — **a1 (DA) is the most surface-concentrated** (surf 59.8%,
  peak z29 = true VAL surface). But surfrel per se does **not** show as sharper depth concentration
  (Q3). The cross-tile "flatter at t192" trend is a **method artefact** (larger tiles average more
  pixels → flatter gates), *not* evidence of broader depth integration — do not over-read it.
- **Also:** **every** model peaks in the z27–33 surface band with ≈0 at window edges → a **trust
  result**: the models rely on the physical ink layers, validating the `[16,48)` depth window.

### Q10. (Open, not yet run) Does **architecture** matter vs the **pipeline**?
- Given Q1–Q8 (only surfrel + data/scale/protocol move the needle), the untested **plain 3D U-Net
  control** is now the single highest-value experiment. See Recommendation R1.

---

## 2. What ALL the model results say (44 models, one paragraph)

The ladder is honest and the story compounds: baseline 0.416 → texture stem/collapse 0.512 →
DA-EPA 0.558 → registration retrain 0.593 → **tile 64→128→192 is the giant lever** (0.639→0.719→0.774)
→ postprocess component filter and the letterform-pretrained OOF refiner **r4** top it out at
**0.7986 (0.7984 ± 0.0033 over 3 seeds)**, best single checkpoint **0.8016 (s44)**. The seed study is
the maturity signal: seed 42 (used for every ablation) was the *weakest* first-stage seed, so any
single-seed Δ < ±0.008 is noise — which is exactly why the DA "win" evaporated on replication. Left
word: invisible (0.10) → majority-recovered (0.54), remainder provably out of reach under current
labels. Nothing in the preprocessing family (photometric, z-score, CLAHE, detrend) helped; raw /255
is correct because ink contrast *is* the signal.

## 3. What ALL the Inkalyzer results say (15 models, one paragraph)

Depth attribution is a single unimodal bump on the papyrus surface (peak z29–34) for every model,
≈0 at the edges. It delivers three things the F0.5 table cannot: **(a) trust** — the models look where
ink physically is; **(b) an independent proof of the registration finding** — unregistered labels
measurably smear and deepen the depth attention (`abl_labels` outlier on all axes); **(c) a mechanistic
reading of the ablations** — DA-EPA barely changes the depth profile (⇒ neutral, matching §7.1), and
surfrel's benefit is spatial not depth-histogram (⇒ the metric's blind spot, stated honestly). The 3D
per-pixel saliency is too blocky to carry quantitative claims and should stay qualitative.

## 4. Where the two streams agree / disagree
- **Agree:** registration matters (both); DA-EPA neutral at t192 (both); stem tile-conditional (both).
- **Complement (not disagree):** surfrel is F0.5-positive but XAI-invisible — because it acts spatially.
  Report this as *"performance and depth-XAI are consistent; the surfrel mechanism is spatial, shown
  qualitatively in 3D saliency, not in the depth histogram."*

---

## 5. Recommendations (ranked, actionable)

**R1 — Run the plain 3D U-Net control (MONAI DynUNet) at the identical final recipe. HIGHEST VALUE.**
Everything above says architecture is mostly inert here. If a vanilla U-Net lands within ~0.01 of
UNETR++, you get a *clean, quotable thesis claim*: **"pipeline ≫ architecture for this task."** If it
loses by >0.02, you've earned the transformer. Either outcome is a strong result and it closes the one
open question (Q10). ~1 training run at t192.

**R2 — Freeze the reported models.** First stage = `a2bs_t192` (0.774/0.786); best pipeline = **r4**
(0.7984 ± 0.0033); best single = `a2bs_t192_s44` (0.8016, always with the seed band). Do not tune
further — R5's stop-list is exhausted.

**R3 — Consider the lean config as the *reported* architecture (parsimony).** a2bs_t192 minus DA minus
stem is statistically indistinguishable (−0.004 mean over 3 seeds), has ~2× lower seed variance and
fewer parameters. Given Q1–Q2, defend the model on parsimony: keep **surfrel + registration + scale**,
drop DA + stem. This makes the architecture chapter *match* the ablation evidence instead of fighting it.

**R4 — TTA (hflip/vflip/rot90 averaging) in predict.** Untested, typically +0.005–0.01, near-free.
Only cheap remaining accuracy win.

**R5 — XAI chapter figures (from this study):** (i) `depth_profiles.png` grouped by tile — the surface
bump (trust); (ii) 2-line overlay **a2bs_t192 vs abl_labels** — registration sharpens/re-centres depth
attention (the cleanest picture); (iii) one `attr3d` triptych captioned qualitative. **Optional:** the
methodology also asked for the **per-stage γ trajectory** over training (is depth-attention γ actually
moving / concentrating in early stages?) — not yet produced; a cheap, complementary XAI figure if you
want to close that sub-question.

**R6 — Thesis framing.** Lead with the **ranked levers** + the **two model-independent contributions**
(label registration, reported upstream to Fei Wu; and the XAI corroboration). Present the DA seed-42
artefact and its replication as a *methods-rigor* highlight — it's a genuinely instructive negative
result. State the surfrel performance/XAI complementarity explicitly so a reviewer can't read the
"identical depth profile" as a contradiction.

**Stop list (all measured ≤ noise or negative):** t256, wider/deeper backbones, more refiner variants,
image-domain enhancement (photometric/CLAHE/detrend/z-score), morphology, loss engineering, γ-init
tuning, prob ensembles, hysteresis, region priors.

---

### One-line bottom line
Your best number is **0.7986 (r4)** / **0.8016 (single)**; your best *contribution* is showing —
with performance ablations **and** independent XAI — that **data (registration), scale (tile), and
protocol beat architecture**, with **surface-relative depth** the lone architectural survivor. Run the
3D U-Net control to nail that claim, report the lean config for parsimony, and you have a tight thesis.
