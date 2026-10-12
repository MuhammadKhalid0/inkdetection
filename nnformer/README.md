# nnFormer — Benchmark Model 2 workspace (Vesuvius ink detection)

MSc thesis benchmark, model 2 of 3 (UNETR++ ✅ → **nnFormer** → SegFormer3D).
Authoritative plan: `~/plan(md_files)/nnformer_plan.md` (rev. 2). Progress: `docs/PORT_LOG.md`.

## Layout

```
reference/nnFormer/    pristine clone of the official repo (never edited)
src/nnformer_ink/      the adapted backbone package
  windows.py           anisotropic window utils, corrected rel-pos index, shift masks
  attention.py         windowed attention (self + skip-attention KV mode)
  blocks.py            SwinBlock3D / SwinBlockKV3D
  sampling.py          PatchEmbed3D / PatchMerging3D / PatchExpanding3D / FinalExpand3D
  backbone.py          Encoder/Decoder/NnFormerInk + build_nnformer_ink + capacity presets
tests/test_smoke.py    Phase-N0 checks (7, all passing on CPU)
configs/               run YAMLs (created at bench-integration step)
scripts/               bench (GFLOPs/ms/mem), slurm templates (next step)
docs/PORT_LOG.md       deviations from the original + checklist status
```

## Contract

`build_nnformer_ink(in_ch, img_size=(32,t,t), capacity="S|M|F", patch_size=(1,4,4),
gv_bottleneck, skip_attention, drop_path, ...)` → module mapping
`[B,in_ch,32,t,t] → [B,16,32,t,t]`, consumed by vesuvius-ink-bench's shared
DepthCollapse (surface_relative) + Head. DS off; `cssa`/`sr_pb` reserved (port step 2).

Measured params (t192): **S 19.87M · M 79.11M (= UNETR++ t256 79.1M match) · F 177.74M**.

## Run tests

```bash
source $WORK/vesuvius/.venv/bin/activate   # $WORK = /home/woody/iwi5/iwi5401h
cd $WORK/nnFormer && python tests/test_smoke.py
```

## Next steps (in order, from the plan)

1. `scripts/bench.py` — GFLOPs / ms-per-tile / peak-mem for S/M/F at t192/t256 (fills the
   headline-table efficiency columns).
2. Bench integration: `models/nnformer/` in vesuvius-ink-bench (symlink or copy of
   `src/nnformer_ink`) + `elif name == "nnformer":` branch in `models/wrapper.py` +
   `configs/n_base_t192_s42.yaml`; `smoke_test.py` on the bench side.
3. `n0_sanity_t64` — 1-epoch GPU sanity run, output root `outputs7/`.
4. Phase N1 screens per the plan (n_base, n_mean, n_abs, n_skipconv, n_lvonly, n_embedz2,
   n_lr5e5, n_srpb*, n_cssa*).  *requires port step 2 (CSSA / SR-PB implementation).
