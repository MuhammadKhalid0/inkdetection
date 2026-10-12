# SegFormer3D-Ink — benchmark model 3

Third and final entry of the 3-model ink-detection benchmark
(UNETR++ ✅ → nnFormer ✅ → **SegFormer3D**).

**Plan:** `~/plan(md_files)/segformer3d_plan.md` · **Port notes:** `docs/PORT_LOG.md`
**Results (auto-generated):** `docs/RESULTS.md` · **Efficiency:** `docs/bench_table.md`

```
segFormer/
  reference/SegFormer3D/     pristine clone of OSUPCVLab/SegFormer3D — NEVER edited
  src/segformer3d_ink/       the port (symlinked into the bench as models/segformer3d)
    geometry.py              stage grids, stems, sr presets, capacity rungs, assertions
    blocks.py                patch embed, sequence-reduced attention, Mix-FFN
    contrib.py               C1 PDK · C2 DCM · C3 OSR · C4 DSM (all exact no-ops at init)
    decoder.py               all-MLP decoder + full-resolution path + conv skip
    backbone.py              SegFormer3DInk + build_segformer3d_ink
  tests/test_smoke.py        50 checks incl. the four exact-reduction proofs
  configs/                   30 configs, each a one-key delta from sf_base_t192_s42
  scripts/                   make_configs · bench · collect_results · stress_tile · sbatch
  outputs/                   run artifacts (INK_OUTPUT_ROOT)
  probes/                    ERF / Inkalyzer / freqband / error-anatomy / transfer
```

## Reproduce

```bash
source $WORK/vesuvius/.venv/bin/activate
python tests/test_smoke.py                       # 50 checks, CPU, ~2 min
python scripts/make_configs.py                   # regenerate every config
cd $WORK/vesuvius/vesuvius-ink-bench
python scripts/smoke_test.py --config $WORK/segFormer/configs/<run>.yaml --device cpu
sbatch $WORK/segFormer/scripts/slurm_s1_screens_a40.sbatch
```

## Two things that will bite anyone who skips the port

1. **The reference code is unusable as published.** `cube_root(N)` reshapes crash at
   t64/t128/t192/t320 and **silently scramble the geometry at t256** — the deployment tile
   — because 8·64·64 = 32,768 = 32³ is the ladder's only perfect cube. No exception is
   raised. `@torch.jit.script` on it also fails to import on torch 2.6.
2. **`models/wrapper.py` silently ignores model keys it does not forward.** A YAML key that
   the branch forgets trains a different model with no error. `test_config_keys_reach_the_model`
   asserts all 20 keys reach the model.

## Contributions (all default-off, all bitwise no-ops at init)

| id | name | mechanism it modifies | primary readout |
|---|---|---|---|
| C1 | Peripheral Downsampled Keys (`pdk`) | the sequence-reduced K/V path | the 2×2 vs a mandatory same-cost control |
| C2 | Depth-Coordinate Mix-FFN (`dcm`) | Mix-FFN = the model's whole positional apparatus | learned table vs Inkalyzer; stress sign asymmetry |
| C3 | Overlapping Sequence Reduction (`sr_overlap`) | the sr conv (the efficiency claim) | frequency band-stop shift at 8–32 px |
| C4 | Decoder Spatial Mixing (`dsm`) | the all-MLP decoder (the decoder claim) | post-filter gain substitution + FP islands |
