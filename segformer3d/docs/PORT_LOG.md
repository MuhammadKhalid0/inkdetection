# SegFormer3D-Ink Port Log

**Started:** 2026-09-10 · **Plan:** `~/plan(md_files)/segformer3d_plan.md`
**Reference:** `reference/SegFormer3D/` = pristine clone of github.com/OSUPCVLab/SegFormer3D
(`main` @ `e314242f14`; file of record `architectures/segformer3d.py`, 664 lines). **Never edited.**
**Port:** `src/segformer3d_ink/` · **Tests:** `tests/test_smoke.py` (50 checks, all passing on
CPU, torch 2.6.0, venv `$WORK/vesuvius/.venv`).

## Status vs plan Phase S0

| Item | Status |
|---|---|
| Directory skeleton (reference / src / tests / configs / scripts / docs / outputs / probes) | ✅ |
| Clone official repo | ✅ |
| Stock backbone port (patch embed, sr-attention, Mix-FFN, all-MLP decoder, final path) | ✅ |
| Ablation toggles: `stem`, `sr`, `final`, `skip_stem`, `dw_norm`, `dec_proj_ln`, `dec_dim`, `scale_mode`, `drop_path` | ✅ |
| Contributions C1 PDK / C2 DCM / C3 OSR / C4 DSM, all exact no-ops at init | ✅ (unit-tested, max\|Δ\| = 0.0) |
| Measured params SF-S / SF-M / SF-L + reference-faithfulness control | ✅ (below) |
| GFLOPs / ms / peak-mem bench (`scripts/bench.py`) | ✅ CPU; GPU numbers from job 4209294 |
| Bench integration: `models/segformer3d` **symlink**, wrapper branch, bookkeeping attrs | ✅ |
| Shared-engine halo carrier (`data/dataset.py`, `engine/infer.py`), default-off | ✅ + inertness proof |
| 30 configs generated + CPU smoke-tested | ✅ |
| `train_reg_t320_d96.csv` dense manifest (37,558 tiles; plan predicted ≈38k) | ✅ |
| A40 batch jobs S0 / S1 / S1b / S1c / S2 submitted | ✅ 4209294, 4209296–4209299 |

## Measured parameter counts (`tests/test_smoke.py`)

| Preset | dims (S1→S4) | heads | depths | **measured** | plan target | role |
|---|---|---|---|---|---|---|
| SF-S | [32,64,160,256] | [1,2,5,8] | [2,2,2,2] | **4,437,568** | 4,437,568 ✅ | stock MiT-B0; the efficiency contender |
| SF-M | [64,128,320,512] | [1,2,5,8] | [2,2,5,2] | **19,966,880** | 19,966,880 ✅ | capacity-matched to nnF-S (19.87M) |
| SF-L | [96,192,480,768] | [1,2,5,8] | [3,6,14,3] | **79,353,344** | 79,353,344 ✅ | matched to UNETR++ t256 = nnF-M (79.1M) |
| *reference config* | *stock* | *stock* | *stock* | **4,491,809** | 4,491,809 ✅ | **faithfulness control**: reproduces the released file's own count exactly |

Params are **tile-invariant** (asserted t64 == t256): no relative-position tables, no
positional encoding, no token-axis projections. The only tile-dependent state in the model
is the BatchNorm running statistics — which is why the resolution-transfer study reports
curves with and without them.

Measured GFLOPs (batch 1, FlopCounterMode) at t192 / t256 / t320, matching the plan:
SF-S dp `144_122` **202.4 / 497.5 / 1054.0** · dp `444_222` **131.7 / 274.1 / 508.6** ·
iso `444_222` **66.4 / 120.5 / 193.3** · SF-M **346.3 / 898.1** · SF-L **836.2 / 2212.6**.

## Deviations from the original (each deliberate, each cited to the plan)

| # | Original | Port | Why |
|---|---|---|---|
| **D1** | `cube_root(N)` + `reshape(B,C,n,n,n)` in `SelfAttention`, `DWConv` and the stage loop | Explicit per-stage `(D,H,W)` grid computed at build time, threaded through every block, asserted at forward time | **The load-bearing fix.** For `(B,1,32,t,t)` the reference raises at t64/t128/t192/t320 and **silently succeeds at t256** (8·64·64 = 32,768 = 32³), reinterpreting a (8,64,64) grid as (32,32,32) and mixing tokens across depth rows. A t256 run on unmodified code is invalid **with no exception raised**. Verified arithmetically in `test_geometry`: t256 is the *only* perfect-cube tile in the ladder. |
| **D2** | `@torch.jit.script def cube_root` | Removed with the reshapes | Fails to import on torch 2.6.0 ("Return value was annotated as having type int but is actually of type float") — the file of record does not load in this environment at all. |
| **D3** | Scalar per-stage `sr_ratio` → `Conv3d(k=s=r)`, i.e. an r³ reduction | Per-axis tuples (`SR_PRESETS`), divisibility asserted at build | The informative part of the 32-layer axis is ~6 layers; the reference's sr conv has no padding and **floor-truncates silently** on a non-dividing grid. |
| **D4** | Isotropic stem `[4,2,2,2]` (depth 32→8→4→2→1) | Depth-preserving stem default `(3,7,7)/s(1,4,4)/p(1,3,3)`; isotropic kept as ablation A1 | Signal is a thin band at z27–33; under the isotropic stem stage-4 depth is 1, so the Mix-FFN 3×3×3 sees only padding along depth and C2/DCM would be vacuous. |
| **D5** | Final `nn.Upsample(scale_factor=4.0)` | Explicit final path (`expand` = `ConvTranspose3d(k=s=stem_stride)`, or `trilinear_lowres_pred`), + full-resolution conv skip | The isotropic ×4 restores (32,t,t) only because the stem also divided depth by 4; with the depth-preserving stem it returns depth 128. The conv skip is UNETR++/nnFormer parity. |
| **D6** | 3-class softmax head, `num_classes` logits | Returns a **feature volume** `[B,16,D,t,t]`; the shared `DepthCollapse(surface_relative)` + `Head` produce `[B,1,t,t]` | Wrapper contract, identical to both previous ports. |
| **D7** | nnU-Net-ish repo recipe (AdamW lr 1e-4 **wd 0.01**, warmup 20, `cosine_annealing_wr` T₀=400, **800 epochs**, batch 2, Dice) — and a *paper* recipe that differs on every axis | Benchmark protocol: AdamW 1e-4 / wd 3e-5, warmup 3 + cosine 60, `dice_bce_masked`, batch 4, bf16 | Protocol identity across the three models. The paper-vs-repo discrepancy is itself reported (§1 of the plan); the maintainers state the paper's ACDC/Synapse runs were done inside the *nnFormer* codebase with code they no longer have. |
| **D8** | SDPA called without an explicit `scale` (defaults to 1/√head_dim); the launch commit used 1/√num_heads | `scale_mode` flag, default `head_dim`, explicit | The released weights and paper numbers correspond to the launch convention; a temperature is absorbed by from-scratch training, but the discrepancy is documented (ablation A11). |
| **D9** | Manual softmax fallback branch (dead on torch ≥ 2.0) | `F.scaled_dot_product_attention` always; explicit path kept behind `use_sdpa=False` **for unit tests only** | With the depth-preserving stem the explicit path would materialise a `[4,1,131072,8192]` attention matrix per stage-1 block = **8.0 GB in bf16 at t256 batch 4**, kept by autograd for backward. SDPA agreement verified: max\|Δ\| = 9.4e-06. |

## Fix log

- **F1 (2026-09-10):** `assert_exact` initially required `(n + 2p − k) % s == 0`, which is
  wrong for the stem (k=7, s=4, p=3 gives exactly n/4 while 63 % 4 ≠ 0). Replaced by the
  real invariant: `conv_out(n) == n // s` **and** `n % s == 0`. Caught by the port's own
  geometry test on first run.
- **F2 (2026-09-10) — the C3/OSR init-order bug, caught by `test_exact_reduction`:**
  the widened sr conv was built inside `SelfAttention3D.__init__`, but
  `self.apply(self._init_weights)` runs *afterwards* in the backbone and re-initialised the
  wide kernel with the **wide** fan-out, overwriting the zero taps and destroying the
  exact-reduction property (measured max\|Δ\| = 3.9e-02 instead of 0.0). Fix: OSR widening
  is applied **after** `_init_weights`, building the wide kernel from the already-initialised
  stock conv. This is exactly the "init trap" the plan pre-registered.
- **F3 (2026-09-10) — FLOP counter under-count on CPU:** `F.scaled_dot_product_attention`
  dispatches to `aten._scaled_dot_product_flash_attention_for_cpu`, which is **not** in
  `FlopCounterMode`'s registry, so a naive CPU count silently omits every attention matmul
  (t192 read 102.8 instead of 202.4 GFLOPs). `scripts/bench.py` now passes an explicit
  `custom_mapping`. The CUDA kernels are registered, so GPU numbers were never affected.

## Shared-engine changes (default-off) and the inertness proof

`data/dataset.py` (+ ~45 lines, all inside `if self.halo_on:`) and `engine/infer.py`
(+ ~30 lines) carry the C1/PDK halo as **input channel 1**. Backups: `*.bak_pre_segformer`.

Proof that chapters 1 and 2 remain reproducible from the same tree:
1. `scripts/smoke_test.py --config configs/c1_t256_s42.yaml` (UNETR++) → ALL CHECKS PASSED.
2. `scripts/smoke_test.py --config nnFormer/configs/nnf_c1_t256_s42.yaml` → ALL CHECKS PASSED.
3. **Byte-identity**: 10/10 sampled tiles from `train_reg_t256.csv` under the UNETR++ config
   are bitwise identical (`x`, `y`, `u`) to the pre-change `dataset.py` under the same RNG
   seed, with an identical key set. With the halo off the batch has no `halo_mask_frac`
   key and `x` has exactly 1 channel.

## The silent-ignore trap (why `test_config_keys_reach_the_model` exists)

`models/wrapper.py` forwards model keys **one at a time** with `m.get(...)`. A key present
in the YAML but not forwarded is **silently ignored**, and the run trains a different model
than the config describes, with no error. This is not hypothetical: `build_nnformer_ink`
accepts `cssa_scales`, `dims`, `num_heads`, `depths`, `window_sizes`, `mlp_ratio` via
`**kw` and the wrapper forwards **none** of them — dead config surface in chapter 2. The
SegFormer3D branch forwards all 20 keys, and the unit test asserts every one of them
provably changes the built model (parameter count or forward output).

## Measured on the target GPU (job 4209294, A40 48 GB, bf16)

**Memory probe (fwd+bwd, before committing the schedule) — nothing OOMs:**
SF-S t256 batch 4 **9.4 GB** · SF-M t256 batch 4 **12.4 GB** · SF-L t256 batch 4 **22.1 GB** ·
SF-S t320 batch 2 **7.4 GB** · SF-M t320 batch 2 **9.8 GB**.

The plan's pre-decided OOM fallbacks (drop to `sr 444_222`, move to 80 GB nodes, halve the
batch) are therefore **not needed at any rung** — including the 79M SF-L. This is itself a
reportable efficiency result: the capacity rung that forced chapter 2 onto 80 GB nodes fits
in 22 GB here.

**Consequence — t320 raised from batch 2 to batch 4 (correctness, not just speed).** The
plan requires `samples_per_epoch` 4000 with **identical 1000 optimizer steps per epoch in
every cell** so the density and ladder comparisons carry no optimizer-budget confound.
Batch 2 at t320 (inherited from chapter 1, where a 112M model forced it) would have given
2000 steps/epoch and silently broken that parity, besides doubling the wall time. The
measured 7.4–9.8 GB at batch 2 leaves ample headroom, so every cell now runs batch 4 /
1000 steps. The sbatch auto-halving retry remains as the safety net.

**Measured efficiency (batch 1, bf16, A40) — every figure matches the plan's prediction:**
SF-S dp `144_122` 202.4 / 497.5 / 1054.0 GFLOPs and 20.9 / 36.8 / 60.0 ms at t192/t256/t320 ·
dp `444_222` 131.7 / 274.1 / 508.6 · iso `444_222` 66.4 / 120.5 / 193.3 ·
SF-M 346.3 / 898.1 / 1970.8 · SF-L 836.2 / 2212.6. Full table in `docs/bench_table.md`.

**Wrapper wiring verified end-to-end:** the training log reports 4,439,985 trainable
parameters = 4,437,568 (backbone) + 2,417, and 2,417 is exactly the shared
`DepthCollapse(surface_relative)` (17 + 63) plus the shared `Head` (2,320 + 17) — i.e. the
Class-A layer is byte-identical shared code, as the benchmark contract requires.

## A10 — the 2.5D depth-as-channels control (prior art, NOT a benchmark row)

Implemented as `mode: "25d"` rather than as a second network: the 32 depth layers become
input channels and the depth axis becomes a **singleton**, so every 3×3×3 conv degenerates
to 3×3 and every sr conv to a 2D reduction — **the identical code path runs as a 2D MiT**.
Measured **3,531,272 params**; stage grids (1,64,64) → (1,32,32) → (1,16,16) → (1,8,8) at
t256, i.e. 4,096 stage-1 tokens against the 3D model's 131,072.

The output is broadcast back over depth to `[B,16,32,t,t]`, so the shared
`DepthCollapse` + `Head` remain **byte-identical shared code** (every depth slice is equal,
so any depth weighting returns the same 2D map — asserted in the port). This resolves the
plan's wiring caveat without a special-case wrapper branch. Reported as a control cell
alongside the Kaggle-2023 prior art, never as a benchmark row.

## Submitted (A40, 24 h wall)

| job | phase | tasks |
|---|---|---|
| 4209294 | S0 sanity: 50 unit tests · GPU bench · memory probe · 2-epoch pipeline | ✅ COMPLETED 7:34 |
| 4209296 | S1 adaptation screens | 12 |
| 4209297 | S1b tile ladder (t64/t128/t256/t320) | 4 |
| 4209298 | S1c contribution screens (PDK ×2 + control, DCM ×2, OSR, DSM) | 7 |
| 4209299 | S2 capacity × context | 5 |
| 4209324 | S2b t320-dense + 2.5D control | 2 |

**30 training runs**, all chained `afterok` on the sanity job. Not yet submitted (both
depend on results): Phase S3 (the 3-seed final band) and Phase S4 (`slurm_s4_analysis_a40.sbatch`,
written and ready). The ladder decides whether C1's 2×2 is powered at all — that gate fires
mechanically in `scripts/collect_results.py`, not by later judgement.

## Wall-time incident and resume support (2026-09-12)

**What happened.** 25 of 30 runs completed; 5 hit the 24 h A40 wall (`TIMEOUT`):
`sf_base_t320_s42` (45/60 ep), `sf_base_d96_t320_s42` (45/60), `sf_m_t256_s42` (38/60),
`sf_m_t320_s42` (22/60), `sf_l_t256_s42` (10/60). GPU utilisation was 98–99% in every one, so
they were compute-bound. Measured A40 time per epoch: SF-S t320 32 min · SF-M t256 38 min ·
SF-M t320 65 min · SF-L t256 144 min.

**Correction to an earlier statement in this log.** After the memory probe, the t320 cells were
raised to batch 4 and the wall-time risk was declared removed. That was wrong. The first
(FLOP-anchored) projection had predicted three over-runs and was close to right; it was replaced by
a more optimistic per-tile estimate that ignored the per-epoch full-segment validation sweep. The
batch-4 change itself was correct (it restores 1000 steps/epoch parity) — only the timing claim was.

Several *completed* runs show fewer than 60 epochs (e.g. `sf_deep_t192_s42` 39, `sf_sr444_t192_s42`
41). That is early stopping after 20 epochs without improvement — the protocol working — not a fault.

**Archive, nothing deleted:** the five folders were moved to
`outputs/_timeout_a40_20260912/` (byte counts verified). They have `best.pt` and `summary.json` but
no `postprocess.json`, and no `last.pt`, so they cannot be continued.

**Decision.** Four cells re-run on **A100** as resumable chains. `sf_l_t256_s42` is **deferred under
the plan's scope guard**: ~144 h on A40, ~99 h on A100, i.e. about four GPU-days even chained, for a
capacity rung whose nnFormer equivalent was flat. The capacity question is still answered by SF-S vs SF-M.

### The engine change: `train.resume` (DEFAULT OFF)

New `engine/resume.py`; `engine/train.py` gains only guarded insertions (the sole original line
changed is the loop header, `range(epochs)` → `range(start_epoch, epochs)`, which is identical when
`start_epoch = 0`). Backup: `engine/train.py.bak_pre_resume`.

| danger in a naive resume | guard |
|---|---|
| first resumed epoch overwrites the real `best.pt` (because `best_f` restarts at −1) | best_f / best_epoch / best_tau restored, and reconciled *upward* against `summary.json` |
| a crash in a resumed job triggers the old retry's `rm -rf` and batch halving | the chain script contains neither; a crash marks the chain crashed and touches no file |
| warm-up + cosine restart from step 0 | `step` saved, restored, and checked against epoch × steps_per_epoch |
| a kill between the log row and the checkpoint duplicates an epoch in `train_log.csv` | rows after the checkpoint removed; remaining rows must be exactly epochs 0..k |
| EMA runs resume from averaged instead of raw weights | raw and EMA weights stored separately; a mismatch is refused |
| config changed between chunks breaks the step→LR mapping | full config must match the checkpoint's except `resume`, `time_limit_min`, `stop_after_epoch` |
| corrupted `last.pt` from a kill mid-write | written to a temp file, fsync'd, then atomically renamed |
| starting "fresh" into a folder that already has results | refused with a clear message |

Plus `train.time_limit_min` (stop cleanly *between* epochs before the wall; 1320 min used, leaving
margin for the smoke test and post-processing) and `train.stop_after_epoch` (chunk control).

### Proof

`tests/test_resume.py` — **45/45 pass** — drives the real `engine.train.main()` on CPU (real SF-S
model, real t64 tiles, real loss/optimizer/schedule; only full-segment validation is replaced by a
deterministic stand-in):

1. **Resume OFF is byte-identical to the pre-change trainer:** `train_log.csv`, `summary.json`,
   `config_used.json`, `val_prob_best.npy`, `best.pt` weights and optimizer state all identical; no
   `last.pt` or `train_done.json` written.
2. **An interrupted-and-resumed run is byte-identical to an uninterrupted one** (4 epochs straight vs
   2 + stop + resume + 2): identical `train_log.csv`, learning-rate column, `summary.json`, `best.pt`
   and final weights. On CPU with `num_workers: 0` this is exact because the full RNG state is
   restored. On GPU with data-loader workers it is equivalent in distribution, not bit-identical —
   the same as any same-seed rerun, since the engine does not request deterministic GPU kernels.
3. duplicate log row removed · 4. a better `best.pt` is never overwritten · 5. a newer `summary.json`
   is adopted · 6. lr / batch / epochs / warm-up / capacity drift refused, chunk keys allowed ·
   7. a used folder without `last.pt` is refused and left untouched · 8. EMA resumes exactly ·
   9. the time guard stops a chunk cleanly between epochs and the continuation matches the straight
   run · 10. resubmitting a finished run changes nothing · 11. an early-stop condition at the
   checkpoint is honoured.

`tests/test_chain_guard.py` — **10/10 pass**: train / post / stop decisions, the runaway cap, the
no-progress stop, crashed chains never retry, and the guard never deletes an artifact.

Chapters 1 and 2 re-verified: `c1_t256_s42` (UNETR++) and `nnf_c1_t256_s42` (nnFormer) smoke tests pass.

### Chains submitted (A100, `scripts/slurm_chain_a100.sbatch`, max 6 chunks each)

| run | first job |
|---|---|
| `sf_base_t320_s42` | 4222275 |
| `sf_base_d96_t320_s42` | 4222276 |
| `sf_m_t256_s42` | 4222277 |
| `sf_m_t320_s42` | 4222278 |

**Reporting note.** These four ran on A100 while every other cell ran on A40. Accuracy is
unaffected, but wall-time and ms-per-tile figures must be reported per GPU type.

### Verified on real hardware (first A100 chunk, job 4222275, 2026-09-12)

- Each running job passed its config assertion, guard decision, CPU smoke test and chunk record,
  then **queued its own continuation from the compute node** (`afterany`), confirming self-chaining
  works on Alex: 4222275→4222280, 4222276→4222281, 4222277→4222291.
- First `last.pt` after epoch 0: format `ink-resume-v1`, epoch 0, step 1000 = 1 × 1000 steps/epoch,
  optimizer state present, python/numpy/torch/cuda RNG saved, no temp file left, log holds exactly
  epoch 0.
- Epoch-0 sweep F0.5 of 0.0 is expected warm-up behaviour, not a fault: the archived A40 run of the
  same config also scored 0.0 at epoch 0 (0.0036 at epoch 1, 0.467 at epoch 2), as did the completed
  `sf_base_t256_s42`. Epoch-0 train loss 0.66793 (A100) vs 0.66805 (A40).
- **Measured A100 speedup for this model: 2.06×** (15.5 min vs 32 min per epoch at SF-S t320) — higher
  than chapter 2's 1.46× for nnFormer, consistent with SegFormer3D's compute being dominated by
  long-sequence attention, which benefits from the A100's faster memory.

Revised estimates from that single measured epoch (early stopping may shorten any of them):
SF-S t320 and t320-dense ≈ 15.5 h (one chunk each) · SF-M t256 ≈ 18.4 h (one chunk) ·
SF-M t320 ≈ 31.6 h (two chunks) · deferred SF-L t256 ≈ 70 h (about four chunks).

## A15 — the full-65-layer depth-window ablation (2026-09-24)

`sf_depth65_t192_s42`: the reference cell with `data.depth: 32 → 65`, i.e. the **entire depth
stack** instead of the `[16,48)` window. Submitted to A100 as a resumable chain (job 4311373).

- Config diff vs `sf_base_t192_s42` is exactly: `depth 32 → 65`, plus `resume`/`time_limit_min`
  (chunk control) and the run name. Nothing else.
- **Backbone parameters unchanged (4,437,568).** Only the shared collapse's relative-offset table
  grows (63 → 129), so the full model goes 4,439,985 → **4,440,051**, i.e. +66. As close to a pure
  "more input" comparison as the architecture allows.
- Grids `(65,48,48) → (33,24,24) → (17,12,12) → (9,6,6)`; stage-1 tokens **73,728 → 149,760** (2.03×).
- Verified end to end before submission: loader returns `x [1,65,192,192]` with window `[0,65)`;
  smoke test passes; forward/backward finite.
- **Protocol caveat to report with the result:** with `depth == n_layers` the ±2 depth-shift
  augmentation has no room and is inert (verified: every draw returns `[0,65)`), while the 32-layer
  reference does use it. The comparison is therefore "32-layer window + depth jitter" vs "full stack,
  no jitter". Unavoidable — a full stack cannot be shifted.

**Port fix required (F4).** `assert_exact` demanded every axis divide exactly, which 65 does not
(65 → 33 → 17 → 9). That rule was stricter than the real invariant: for a **padded** patch-embed
convolution what matters is (i) the standard ceil output size and (ii) that every input position is
covered by at least one kernel window. The check now tests exactly that and still refuses genuinely
lossy geometry (verified with a deliberately broken stem). The strict divisibility rule is retained
where truncation can actually occur — the sequence-reduction convolution, which has no padding
(`assert_sr_divides`). All 50 port tests still pass.

**Queue note:** the job is pending with reason `AssocGrpGRES` after a full scheduler cycle — the
account's A100 allocation is exhausted by other users, not a fault of the job. It will start when
quota frees. A40 remains the reliable fallback and is now safe for long runs thanks to the chain.
