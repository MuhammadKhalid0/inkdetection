# nnFormer — Final Summary and Verdicts

**Status:** all training and analysis complete (last job 4006121, 2026-08-14). 27 runs on disk.
**Labels:** `ink_reg.npy` for everything reported here. (The seven `*_s42` t192 screens on `ink_reg2.npy` are a *separate yardstick* and must never be mixed with these numbers.)
**Discipline:** noise floor ±0.008; nothing claimed below 3 seeds; τ swept every epoch; gates fixed before the runs.

---

## 1. The benchmark row

| Configuration | Seeds | raw F0.5 | post F0.5 | best seed | +TTA (best) |
|---|---|---|---|---|---|
| stock `nnf_s_d64_t256` | 3 | 0.7825 ± 0.0117 | 0.7981 ± 0.0123 | 0.8120 | 0.8126 |
| **`nnf_c1_t256` (+SR-PB)** | 3 | 0.7801 ± 0.0037 | **0.8005 ± 0.0102** | **0.8122** | **0.8133** |
| *UNETR++ `c1_t256`* | *3* | *0.8111 ± 0.0039* | *0.8190 ± 0.0062* | *0.8261* | *~0.826* |

**nnFormer does not beat UNETR++.** 3-seed post means: 0.8005 vs 0.8190 = **−0.0185**. Best seed to best seed: 0.8122 vs 0.8261 = −0.0139. Report both; the gap is real and outside the noise floor either way.

---

## 2. Verdicts (final, after full seed replication)

### 2.1 Capacity — dead lever ✔ *(pre-registered prediction confirmed)*

S 19.9M / M 79.1M / F 177.8M at t256 span only 0.006 raw — inside noise, smallest nominally best. nnF-F shows the textbook overfit signature (best epoch 37, train loss falling a further 15% while validation declined). Dense stride-64 sampling mitigated it (best epoch 37→53). **150M-class capacity buys nothing on 9 segments.**

### 2.2 Stock architecture toggles — all neutral

Skip attention removed: **−0.0002**. GV bottleneck removed: +0.0053. LR halved: +0.0005. Collapse variants ≤ |0.010|. **Not even nnFormer's signature component (skip attention) changes the result.**

### 2.3 SR-PB (contribution C1) — ⚠️ **verdict revised at the last step**

| | raw | post |
|---|---|---|
| stock t192 (3 seeds) | 0.7657 ± 0.0066 | 0.7836 ± 0.0086 |
| SR-PB t192 (3 seeds) | 0.7722 ± **0.0013** | 0.7831 ± **0.0030** |
| **Δ band vs band** | **+0.0065** (inside ±0.008) | **−0.0006** |

Individual stock seeds: s42 **0.7581**, s43 0.7687, s44 0.7703.

**Seed 42 was the weakest stock seed.** Every earlier SR-PB delta (+0.0127…+0.0152) had been measured against it. Against the proper band, **SR-PB is neutral on F0.5**.

This is the DA-EPA seed-42 episode reproducing on a second architecture, and it is the single strongest methodological result of this chapter: *the same failure mode recurred, and the same discipline caught it again.*

**What does survive, and is claimable:**
- **Variance reduction, large and consistent**: raw seed spread 5× tighter at t192 (±0.0013 vs ±0.0066) and ~3× tighter at t256 (±0.0037 vs ±0.0117). Independently reproduced at two scales.
- **Mechanism (Inkalyzer)**: SR-PB is the outlier on every depth-attribution axis — highest entropy (0.828 vs 0.792), lowest concentration (0.444 vs 0.501), peak one layer deeper. **It makes the model read a wider depth band.** This is a within-model observation, unaffected by the seed issue.
- **Notably**, in the UNETR++ study the equivalent surface-relative mechanism was *invisible* to this metric and had to be recorded as a limitation. Here it is visible — that limitation is resolved.

**⚠ Not claimable without more runs:** the depth-offset robustness curve (SR-PB better at every δ, mean +0.035) compared against **`n_base_t192_r1_s42` only** — the weak seed. Tail gaps (+0.096, +0.077) are large relative to the 0.0066 seed spread, so the effect likely survives, but it is **n=1 on the baseline side**. Two inference-only stress runs on base seeds 43/44 would settle it.

### 2.4 CSSA (contribution C2) — mechanism confirmed, metric neutral

- F0.5: +0.0037 (t192), +0.0020 (t128) — inside noise. Pre-registered 2×2 gate: closed 32% of the stock t128→t192 gap, needed ≥50%. **FAIL.**
- **But the ERF grew measurably**: r50 63→75 px (+19%), r90 97→107. Recall rose 0.529→0.601 at precision cost.
- **The gate was also underpowered by construction**: nnFormer's stock t128→t192 gap is only **+0.0063** (UNETR++'s was +0.055) — there was almost no gap to close.

**Honest framing:** the intervention did what it was designed to do (more context genuinely reached fine decoding, measured) and did not convert into the precision-weighted metric.

### 2.5 SR-PB + supervised surface (`n_srpb_aux`) — gate met, hypothesis refuted ✔

raw **0.7607** / post 0.7747 — *below* both stock (0.7657) and plain SR-PB (0.7722). **Supervising the surface estimate hurts**, replicating UNETR++'s aux-ŝ finding that learned geometry costs centre accuracy. Pre-registered expectation confirmed; item closed.

### 2.6 vFFC prior art (Quattrini et al. 2023) — best number, still provisional

`n_vffc_t192_r1_s42`: raw **0.7844** / post **0.7992** — the best t192 result in the study. Against the stock *band* (0.7657 ± 0.0066) that is **+0.0187**, outside noise.

**Two reasons it is not yet a claim:**
1. **Capacity confound**: 65.0M params vs 20M stock. §2.1's capacity-flatness only partly covers this — that experiment varied *attention width*, not operator class.
2. **n=1** — and §2.3 just demonstrated exactly how a single-seed delta can mislead here.

**To settle it:** vFFC seeds 43/44 + one parameter-matched control (≈3 runs).

### 2.7 Epoch budget ✔

e90 vs e60 same seed: +0.0062 raw (inside noise), best epoch 54/90. **60 epochs adequate**, matching the UNETR++ precedent.

---

## 3. Why nnFormer trails UNETR++ — the mechanistic answer

| Model | tile | r90 | **r90 / tile** |
|---|---|---|---|
| `n_base_t192_r1_s42` | 192 | 97 | **51%** |
| `n_cssa_t192_r1_s42` | 192 | 107 | 56% |
| `nnf_s_d64_t256_s42` | 256 | 115 | **45%** |
| `nnf_c1_t256_s42` | 256 | 124 | 48% |
| *UNETR++ (all rungs)* | *64→256* | *35 / 68 / 105 / 134* | *~53% constant* |

UNETR++ consumed a **constant ~53%** of whatever tile it was given — more context, more performance (+0.055 for t128→t192). nnFormer's fraction is lower **and shrinks** with tile size (51%→45%): its windowed local attention *bounds* how far information propagates. Consequence: its own tile ladder is nearly flat (+0.0063 for t128→t192).

**Thesis sentence:** *local windowed attention caps context utilisation; EPA's paired channel/spatial attention does not.*

This also explains CSSA's outcome — it raised the ERF, but the binding constraint on final accuracy lies elsewhere.

---

## 4. Cross-model agreement (the benchmark's real payload)

| Question | UNETR++ | nnFormer | Agree? |
|---|---|---|---|
| Do stock architecture toggles matter? | No | No | ✔ |
| Does capacity help? | No (t320 overfit) | No (S≈M≈F, F overfits) | ✔ |
| Where in depth does the model look? | z29–34 bump, ~0 at edges | z32–33 bump, ~0 at edges | ✔ |
| Is fine crackle texture the signal? | No (2–8px ≈ 0) | No (2–4px = −0.003) | ✔ |
| Is the residual error profile the same? | 71.6% unexplained, R 0.578 | 72.1% unexplained, R 0.588 | ✔ |
| **Does tile size / context dominate?** | **Yes (+0.16)** | **No (+0.0063)** | **✘** |

Five of six agree → those five are properties of **the task**, not of one architecture. The single disagreement is *explained*, not left open (§3).

---

## 5. What this chapter contributes

1. **A cross-architecture replication** of the study's central claim: data, scale and protocol dominate architecture. Two independent transformer families, same conclusion.
2. **A mechanistic explanation of an architecture's limits** — the ERF ratio, which predicts *in advance* which architectures can exploit larger context. This is more transferable than any single F0.5.
3. **A methodological result**: the seed-42 trap recurred on a new model, and band-vs-band discipline caught it. Reported deltas of +0.013–0.015 collapsed to +0.0065 (noise) once the baseline had its own band.
4. **Two implemented and honestly-evaluated contributions**: SR-PB (variance reduction + a visible depth mechanism; accuracy neutral) and CSSA (mechanism confirmed, metric neutral) — both exact no-ops at init, so every delta is attributable.
5. **A port-level finding**: the published relative-position-bias indexing is unsafe for anisotropic windows (silently collides distinct offsets). The same class of defect was found in the UNETR++ port — published 3D medical repos do not transfer safely.
6. **A negative result worth recording**: supervising the surface estimate hurts, on both architectures.

---

## 6. Remaining optional work (none blocks the chapter)

| Item | Cost | Value |
|---|---|---|
| Stress curves on base seeds 43/44 | 2 inference jobs | Upgrades SR-PB's robustness claim from n=1-baseline to band-vs-band |
| vFFC seeds 43/44 + parameter-matched control | ~3 training runs | Settles the study's best unexplained number |
| `n_sads` (stroke-aware DS) | ~80 lines shared-engine surgery | Lowest value; plan already excludes it from the headline |

**Recommendation:** run the two stress jobs (cheap, closes a live claim). Treat vFFC as a documented open question unless the supervisor wants it settled. Drop `n_sads`.

---

## 7. Reproduction map

| What | Where (`$WORK=/home/woody/iwi5/iwi5401h`) |
|---|---|
| Model + contributions | `$WORK/nnFormer/src/nnformer_ink/` |
| Unit tests (11, incl. exact-reduction proofs) | `$WORK/nnFormer/tests/test_smoke.py` |
| Port deviations D1–D9 | `$WORK/nnFormer/docs/PORT_LOG.md` |
| Auto-generated gate report | `$WORK/nnFormer/docs/RESULTS.md` |
| Efficiency table | `$WORK/nnFormer/docs/bench_table.md` |
| Runs | `$WORK/nnFormer/outputs/<run>/` |
| Probes | `$WORK/nnFormer/probes/`, `outputs/zshift_stress/`, `outputs/tta/` |
| Tutorial | `$WORK/nnFormer/docs/TUTORIAL_nnformer.md` |
| Presentation script | `$WORK/nnFormer/docs/presentation.md` |
| Label re-registration round | `$WORK/nnFormer/rereg/summary.md` |
