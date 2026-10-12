# nnFormer-Ink Port Log

**Started:** 2026-07-19 · **Plan:** `~/plan(md_files)/nnformer_plan.md` (rev. 2)
**Reference:** `reference/nnFormer/` = pristine shallow clone of github.com/282857341/nnFormer
(file of record: `nnformer/network_architecture/nnFormer_synapse.py`). Never edited.
**Port:** `src/nnformer_ink/` · **Tests:** `tests/test_smoke.py` (7 checks, all passing on CPU,
torch 2.6.0, venv `$WORK/vesuvius/.venv`).

## Status vs plan Phase N0

| Item | Status |
|---|---|
| Directory skeleton (reference / src / tests / configs / scripts / docs) | ✅ |
| Clone official repo | ✅ |
| Stock backbone port (embed, LV/GV, skip attention, decoder, full-res head) | ✅ |
| Ablation toggles: `gv_bottleneck`, `skip_attention`, `patch_size` | ✅ (smoke-tested) |
| Reserved flags `cssa` / `sr_pb` / `deep_supervision` hard-fail | ✅ (tested) |
| Measured params S/M/F | ✅ (below) |
| GFLOPs / ms-per-tile / peak-mem bench script (`scripts/bench.py`) | ✅ (GPU numbers from job 3871227 → `docs/bench_table.md`) |
| Bench integration: `models/nnformer` **symlink** → `src/nnformer_ink`, wrapper branch, bookkeeping attrs (`stage_res`/`input_size`/`feat_size`) | ✅ bench `smoke_test.py` passes at t64 + t192 incl. real-tile check |
| Configs: `n0_sanity_t64.yaml`, `n_base_t192_s42.yaml` (N-series recipe: reg labels + u1 + no z_flip) | ✅ output root = `$WORK/nnFormer/outputs` (per Omar: own tree, not outputs7) |
| GPU sanity job (`n0_sanity_t64` + t256 batch-4 memory probe + bench) on **A100** | ✅ **job 3871227 PASSED** (3 min, a100_40): pipeline end-to-end clean (train→sweep→postprocess→figs); bench table written |
| Phase N1: 6 screen configs generated + smoke-tested; **array job 3871235 submitted** (n_base + n_mean/n_abs/n_skipconv/n_lvonly/n_embedz2/n_lr5e5, t192 s42, A100, 24 h wall) | 🟡 running |
| CSSA / SR-PB implementation (port step 2) | ✅ **DONE 2026-07-20** — `src/nnformer_ink/contrib.py`; both bit-exact no-ops at init (max\|Δ\|=0.0, unit-tested) and provably live once trained; 11/11 smoke checks pass |
| Phase N2 (capacity × density, t256) | ✅ 5 runs — capacity flat (S 0.7868 / M 0.7753 / F 0.7806 post), F shows the overfit signature (best ep 37, loss −15% after peak), dense stride best overall (S+d64 0.7884 post) |
| Phase N3 + contribution screens | ✅ **array 3978947 — all 8 COMPLETED** (8 tasks: t192 stock/CSSA/SR-PB, seeds 43/44, CSSA 2×2 t128 cells, e90 budget test) |
| Phase N4 analysis | 🟡 job 3978948 **NODE_FAIL** (cluster fault, 6 min in) → resubmitted as **3990324**; second pass **3990361** chained after the final batch |
| **FINAL batch (array 3990360)** | 🟡 SR-PB seeds 43/44 + `nnf_c1_t256` seeds 42/43/44 (S+dense+SR-PB) — the last runs the plan calls for |

## Verdicts (2026-08-10, `docs/RESULTS.md`, gates applied mechanically)

| gate | result |
|---|---|
| **CSSA (C2)** | **FAIL / neutral**: +0.002 (t128), +0.004 (t192), both inside ±0.008. Caveat that matters: the stock t128→t192 gap on nnFormer is only **+0.0063** (UNETR++: +0.055) — the 2×2 had almost no dynamic range, so this is an *underpowered* test, not a clean refutation. CSSA consistently traded precision for recall (R .540→.605, .529→.601). |
| **SR-PB (C1)** | **+0.0127 raw at δ=0, ABOVE the noise floor** (single seed), best precision of any t192 run (.894) — and *contrary to its own pre-registration* (predicted neutral at centre). Confirmation seeds now queued. |
| Benchmark row | nnf_s_d64_t256 3 seeds **0.7825 ± 0.0117 raw / 0.7981 ± 0.0123 post** (best seed 0.8120) vs UNETR++ 0.8111/0.8190. Seed variance 2–3× UNETR++'s. |
| Budget (e90) | +0.0062 raw vs e60 same seed (inside noise), best ep 54/90 → **60 epochs adequate**, matching the UNETR++ precedent. Flag closed. |

## N0 sanity results (job 3871227, 2026-07-19)

- **Pipeline:** 2-epoch t64 run trains, sweeps, postprocesses, plots — no errors. (Val F0.5 ≈ 0.20 at sweep-τ after 2 epochs = the untrained-prior level; expected — this run validates plumbing, not accuracy. Not a reported number.)
- **t256 batch-4 memory probe (bf16 fwd+bwd, A100-40GB):** nnF-S OK (peak 20.1 GB) · **nnF-M OOM · nnF-F OOM**.
  **Decision:** the engine has no grad-accum; instead Phase N2 t256 jobs for M/F request the 80 GB nodes (`#SBATCH --constraint=a100_80` — Alex has 18 of them). If nnF-F still OOMs at 80 GB: batch 2 (no accum; state the batch confound) or implement default-off accumulation in `engine/train.py` — decide then.
- **Efficiency bench (bf16, batch 1, A100; `docs/bench_table.md`):** S 171.8/305.6 GFLOPs, 25.8/36.5 ms, 0.77/1.26 GB (t192/t256) · M 506.6/900.9 GFLOPs, 33.6/59.5 ms · F 1046/1860 GFLOPs, 47.8/86.4 ms. → headline-table efficiency columns.

## Fix log

- **F1 (2026-07-19):** `n_lvonly` smoke caught a geometry error — LV window (4,4,4) does not divide the t192 bottleneck grid (4,6,6). Fix: `_lv_bottleneck_window()` derives the largest grid-dividing, in-plane-strictly-local window capped by the S3 window → (4,3,3) at t192, (4,4,4) at t256. The no-padding design worked as intended (fail at build, not silently at train).

## Measured parameter counts (2026-07-19, `test_param_counts`)

| Preset | dims (S1→S4) | heads | t192 | t256 | role (plan §3.2) |
|---|---|---|---|---|---|
| nnF-S | [64,128,256,512] | [2,4,8,16] | **19.87 M** | 19.90 M | UNETR++ t192-class workhorse |
| nnF-M | [128,256,512,1024] | [4,8,16,32] | **79.11 M** | 79.18 M | capacity match: UNETR++ t256 = 79.1 M |
| nnF-F | [192,384,768,1536] | [6,12,24,48] | **177.74 M** | 177.84 M | published-class (paper: 150.5 M at embed 192 with DS heads) |

t192→t256 delta = GV bottleneck bias table only (asserted exactly in the test): the
bottleneck grid grows with tile, its rel-pos table with it. All other params tile-invariant
(E2 holds with this one stated exception).

Note: the plan's §3.2 width-scaling *estimates* (~38/65/150 M for base 48/64/96) were off —
the published 150.5 M corresponds to embed_dim **192**, not 96. Presets recalibrated to hit
the plan's three *roles* (small / capacity-matched / published-class) at measured counts.
Plan §3.2 updated with measured values.

## Deviations from the original (each deliberate, each cited to the plan)

| # | Original | Port | Why |
|---|---|---|---|
| D1 | Cubic scalar `window_size`, per-block clamp to `min(resolution)` | Anisotropic tuple windows per stage, per-axis clamp | D=32 vs t/4 in-plane is anisotropic; plan §3.1 schedule (4,8,8)/(4,8,8)/(4,4,4)/GV |
| D2 | Rel-pos index strides `*= 3*ws-1` / `*= 2*ws-1` | Standard Swin-3D strides `(2wh-1)(2ww-1)` / `(2ww-1)`, bounds-asserted | Original math collides distinct offsets onto shared table entries (verifiable in `WindowAttention.__init__`); the exact "silently mis-sized" class the plan's checklist targets |
| D3 | F.pad to window multiples every forward | **No padding paths**; divisibility asserted at build | Plan §3.1: schedule divides cleanly at t∈{64,128,192,256}; padding hides geometry bugs |
| D4 | Shift mask rebuilt every forward (numpy ceil etc.) | Precomputed once at init, registered buffer | Grids are static per config |
| D5 | DS on: `final_patch_expanding` per scale → class logits | DS **off**; single `FinalExpand3D` to a 16-ch full-res feature volume + full-res conv-skip stem + fuse | Benchmark protocol (plan §2.3); wrapper contract `[B,16,D,t,t]` → shared DepthCollapse/Head; conv skip = UNETR++ parity substitute |
| D6 | `PatchEmbed` strides hardcoded from `patch_size` assuming ps0∈{2,4} | Per-axis two-factor stride factorization (1→(1,1), 2→(2,1), 4→(2,2)) | Depth-preserving (1,4,4) default + (2,4,4) ablation share one code path |
| D7 | GV-MSA implicit (window ≥ resolution → no partitioning) | Explicit: bottleneck window = full grid when `gv_bottleneck=True`; `False` → LV window (n_lvonly) | Same behaviour, made a config-level toggle for the three-way bottleneck ablation |
| D8 | Decoder skip fusion fixed (x_up + skip, then KV block) | `skip_mode="attn"` (faithful) or `"conv"` (concat→linear fuse, standard blocks) | n_skipconv ablation (plan Phase N1) |
| D9 | trainer = nnU-Net (SGD poly 0.01, CE+Dice, DS) | Not ported | Bench engine owns training (plan §1 recipe row) |

Faithful (verified against source): Q in skip attention gets **no** linear projection
(`q = x_up` after shared norm1); residual anchored on `x = x_up + skip`; first decoder
block unshifted, subsequent shifted; PatchMerging = GELU→LN→conv3 s2; PatchExpanding =
LN→deconv k2 s2; embedding = two `project` blocks (conv→GELU→LN→conv[→GELU→LN]);
drop-path linspace over blocks; qkv_bias on; softmax-scale 1/√head_dim.

## Checklist §3.4 status

1. No hardcoded sizes; build-time asserts t64–t256 ✅ (`_stage_grids`, `assert_divisible`, input-shape assert)
2. One axis convention `[B,C,D,H,W]` with shape asserts at every reshape ✅
3. Rel-pos tables from window; SR-PB reserved ✅ / exact-reduction unit test for SR-PB ⬜ (with step 2)
4. Norm-layer guards: LayerNorm only in the transformer; conv stem uses plain convs+GELU (no GroupNorm → the 11-channel divisibility trap avoided by construction) ✅
5. bf16/fp32 islands: engine-side (bench autocast); no soft-argmax in the stock backbone ✅ n/a yet
6. py_compile + smoke before sbatch ✅ (practiced); outputs root `outputs7/` ⬜ (at first run)
7. Measured params ✅; GFLOPs/ms/mem bench ⬜ next step
