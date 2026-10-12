# VAL label findings (2026-07-20) — left-word FP diagnosis

**Trigger (Omar):** all models show FPs near the left word in the same place → suspected
residual label misregistration. **Method:** the project's validated prob-correlation
registration, made *local* (per x-window dy scan), across 4 independent checkpoints
(UNETR++ c1_t256 / t256_lean / a2bs_t192 + nnFormer n_base) — architecture-independent
by construction. Artifacts: `docs/leftword_diagnosis.json`, `docs/figs/leftword_fp_*.png`,
script `scripts/leftword_fp_diagnosis.py`.

## Finding 1 — CONFIRMED: smooth residual misregistration tilt on VAL (±15 px at the ends)

Local dy* (label shift that maximizes prob-mass on labels), consistent across all 4 models:

| x-window | dy* (4-model range) |
|---|---|
| 750–3750 (left, incl. left word) | **+10 … +17** (labels sit ~13–15 px ABOVE the true ink) |
| 6000–9000 (centre) | **0 … +1** (registration correct) |
| 9750–11479 (right) | **−12 … −18** (labels sit below; gain up to ×1.065 far right) |

The 2026-07 registration fitted a linear dy(x) warp (−54…−22); the residual is a smooth
end-tilt it under-corrected. Consequences, equal for every model: FP slivers on the
**bottom edges** of left-word strokes + FN band on their top edges (visible in the c1
overlay), understated left-word recall, and a deflated absolute F0.5 for all models.
NOTE: the u1 trimap (r=4, train-only) does NOT absorb this — it is (a) ±4 px vs a
13–17 px offset and (b) never applied to VAL scoring.

## Finding 2 — the blobs ABOVE the word are most plausibly UNANNOTATED INK (adjacent text line)

At y ≈ 100–300 (≈400–500 px above the text line), both overlays show a horizontal row of
letter-scale, letter-spaced probability blobs at the SAME x-positions in every model
(this matches the measured FP population with median |offset| ≈ 400–460 px in
a2bs/nnFormer, and cross-model FP IoU 0.2–0.4). Interpretation: the top of the VAL strip
clips an adjacent (previous) line of text whose ink was never annotated — same class as
the two dead components (x≈6177/9724) and the contrast-inverted letter (x≈9852). Every
model detects it; every model is penalized for being right.

## Recommendations (benchmark-preserving; label changes need supervisor sign-off)

1. **Upstream report to Fei Wu** (extends the existing §6.2 obligation): the ±15 px
   end-tilt table above + the suspected unlabeled line at the top-left of 090990.
2. **Do not silently re-register mid-benchmark** (breaks comparability with 44 UNETR++
   runs). The clean move mirrors the original registration study: fit dy(x) with one more
   degree of freedom (quadratic or per-window piecewise) from the 4-model prob-correlation,
   produce `ink_reg2.npy`, and **rescore all reported checkpoints on both label versions**
   (no retraining; expected uplift concentrated in precision/left-word recall, order
   preserved — that is itself a result). Decision belongs to Omar + supervisor.
3. Thesis text: both findings strengthen the label-noise chapter — a third independent
   line of evidence (after registration-by-contrast and Inkalyzer depth-smearing) that
   remaining "errors" are partly annotation-side.

## Bearing on "nnFormer shows no improvement" (status 2026-07-20, t192 seed-42 screens)

- n_base 0.7581 raw / 0.7747 post vs UNETR++ t192 same-seed anchors (lean 0.7784/0.7957,
  u1 0.7869/0.7992): nnFormer sits ~0.02 BELOW at screen scale with 20M vs 25M params.
  This matches pre-registered prediction 1 (pipeline ≫ architecture, ±0.01–0.02 band).
- Screen verdicts (±0.008 single-seed floor): **embedz2 +0.0119 raw / +0.0218 post — the
  one above-noise winner → per the pre-registered rule the N2 base is amended to
  patch (2,4,4), with a (1,4,4) confirmation cell at t256** (the stem lesson).
  n_mean +0.009 raw (surfrel not helping nnFormer at t192 — borderline, park).
  GV removal −0.008/−0.011 (GV helps, matches pre-registration). Skip-attn removal
  −0.006 raw/+0.007 post (neutral — the DA-EPA pattern). lr5e5 +0.006 (noise, park).
- The decisive comparison is still pending: t256 + capacity-M (79M match) + density +
  TTA + min-size + contributions (CSSA/SR-PB).
