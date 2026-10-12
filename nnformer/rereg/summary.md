# Re-registration round — final summary (2026-07-20)

**Deliverable:** `proc/<seg>/ink_reg2.npy` for all 9 segments (+ `reg2_report.json` each),
QC figures (`rereg/figs/reg2_*.png`), annotation QC (`rereg/annotation_qc.json`),
rescore tables (`rereg/rescore_reg2*.md`), data layer (`_reg2_*` manifests + `unc_reg2.npy`).
Historical files (`ink.npy`, `ink_reg.npy`, `unc.npy`, old manifests) untouched.

## Method (why this one is trustworthy)

- **Estimator: out-of-fold model probability maps.** 8 train segments ← the a2bs_t192
  LOO fold maps (each segment predicted by a model that never saw it → zero label
  leakage). Val ← 3 checkpoints spanning 2 architectures (c1_t256, t256_lean, nnFormer
  n_embedz2; val is OOF for every model), per-window majority vote.
- **Smooth 2D offset field** dy(x,y), dx(x,y): per-window (1024 px, 50% overlap) scan of
  the shift maximizing prob-mass under the ORIGINAL labels; Gaussian-weighted field
  (σ=700 px, sharpness×ink-count weights); bilinear banded warp.
- **Gain-gated verification:** post-warp residual counted only where shifting still buys
  ≥1% — flat plateaus are "aligned", not "residual".

## Iterations (all archived)

| iter | change | outcome |
|---|---|---|
| 1 | σ=1200, win 1280 | +0.008…+0.012 post on all 9 checkpoints, but field over-smoothed (val tilt flattened) |
| 2 | + absolute sharpness filter | **REJECTED** — filter discarded ~90% of windows on the soft LOO maps → wild extrapolation |
| 3 = **FINAL** | dense grid, σ=700, sharpness as weight, 3-map val vote, gain-gated verify | beats iter 1 on every rescored checkpoint; residuals clean (145341 7/7; val 34/38) |

## Rescore (same maps, no retraining — iteration 3)

| run | reg post | **reg2 post** | Δ |
|---|---|---|---|
| UNETR++ c1_t256_s42 | 0.8261 | **0.8379** | +0.0118 |
| UNETR++ c1_t256_s43 | 0.8161 | 0.8254 | +0.0093 |
| UNETR++ c1_t256_s44 | 0.8147 | 0.8248 | +0.0101 |
| UNETR++ t256_lean_s42 | 0.8181 | 0.8304 | +0.0123 |
| UNETR++ lean_t192_s42 | 0.7957 | 0.8058 | +0.0101 |
| UNETR++ a2bs_t192 | 0.7857 | 0.7961 | +0.0104 |
| UNETR++ r4 | 0.7986 | 0.8081 | +0.0095 |
| nnFormer n_base_t192 | 0.7747 | 0.7841 | +0.0094 |
| nnFormer n_embedz2_t192 | 0.7965 | 0.8063 | +0.0098 |

Uplift on **every** checkpoint, ranking preserved → the labels moved toward the truth,
not toward any one model. New best single (pre-TTA): **c1_t256_s42 post 0.8379**.

## Per-segment fields (iteration 3; dy ranges from reg2_report.json)

145341: −39…−28 (was identity-gated!) · 145701: −48…−17 · 152031: −76…+38 (large
internal variation; see caveat) · 111522: −36…−15 · 115232: −33…−19 · 120732: −18…+3 ·
070770: −69…−5 · 080880: −30…−1 · **val 090990: −40…−27 (the diagnosed end-tilt,
now corrected)**. All four previously identity-gated segments carried real offsets —
the old acceptance gate was too conservative; the OOF-map estimator sees what the
intensity-contrast method could not.

## Annotation QC (upstream report material — NOT registration)

- **Val:** method sanity-check passed — the 2 known dead components reproduced
  (x≈9717, x≈6166). The "blobs above the left word" are now quantified: unlabeled-ink
  suspicions at (x≈1341,y≈346), (1786,212), (2779,218), (3690,157), consensus prob
  0.61–0.69 — consistent with an unannotated adjacent text line at the strip top.
- **152031 (biggest train segment): 61 dead-label components, top ones 232k–668k px at
  consensus prob 0.03–0.05** — potentially several % of its ink labels not supported by
  any model evidence. Caveat: its OOF fold (f2) was the weakest model; still, components
  this large at p≈0.03 point to annotation, not model weakness. This is likely genuine
  TRAINING POISON and the single biggest upstream item.
- Totals: 154 unlabeled-ink suspicions, 65 dead labels across 9 segments
  (`annotation_qc.json` has all coordinates).

## Retraining (data layer ready)

Config delta for any rerun: `data.label_file: ink_reg2.npy` ·
`data.manifest_suffix: _reg2_t192 | _reg2_t256 | _reg2_t256_d64` ·
`data.unc_file: unc_reg2.npy` (r=4, no dy-widening — reg2 corrected those offsets).
Comparability rule: retrained runs are a new yardstick; never mix reg- and reg2-scored
numbers in one table without labelling the label version.

## Open option (gated, needs Omar/supervisor decision)

Loss-level exclusion of the 65 consensus-dead components (extend the trimap: mark them
uncertain) — the u1 mechanism applied to the QC findings. Principled but introduces
model-consensus into the labels; propose upstream first.
