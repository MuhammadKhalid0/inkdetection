"""SegFormer3D-Ink unit tests.  Run: python tests/test_smoke.py  (CPU, ~1 min)

The exact-reduction tests are the ones that make every ablation a genuine single toggle:
each contribution must be BITWISE identical to stock at init, provably live once trained,
and must receive gradients.
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from segformer3d_ink import build_segformer3d_ink as B
from segformer3d_ink.geometry import (SR_PRESETS, STEMS, assert_exact,
                                      assert_sr_divides, stage_grids, token_counts)

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")


def _same(a, b):
    return torch.max(torch.abs(a - b)).item()


# ---------------------------------------------------------------- 1. geometry
def test_geometry():
    print("[1] geometry / divisibility contract")
    expect = {64:  [(32,16,16),(16,8,8),(8,4,4),(4,2,2)],
              128: [(32,32,32),(16,16,16),(8,8,8),(4,4,4)],
              192: [(32,48,48),(16,24,24),(8,12,12),(4,6,6)],
              256: [(32,64,64),(16,32,32),(8,16,16),(4,8,8)],
              320: [(32,80,80),(16,40,40),(8,20,20),(4,10,10)]}
    for t, g in expect.items():
        for stem in STEMS:
            assert_exact((32, t, t), stem)
            for sr in SR_PRESETS.values():
                assert_sr_divides(stage_grids((32, t, t), stem), sr)
        check(f"t{t} dp grids", stage_grids((32, t, t), "dp") == g, str(g))
    # the trap that makes the reference silently wrong at exactly one tile
    iso = {t: token_counts(stage_grids((32, t, t), "iso"))[0] for t in expect}
    cubes = {t: round(n ** (1/3)) ** 3 == n for t, n in iso.items()}
    check("cube_root trap: t256 is the ONLY perfect-cube tile",
          cubes == {64: False, 128: False, 192: False, 256: True, 320: False},
          f"stage-1 tokens {iso}")


# ---------------------------------------------------------------- 2. params
def test_params():
    print("[2] measured parameter counts")
    tgt = [("S", dict(capacity="S"), 4437568), ("M", dict(capacity="M"), 19966880),
           ("L", dict(capacity="L"), 79353344)]
    for name, kw, t in tgt:
        n = sum(p.numel() for p in B(img_size=(32, 64, 64), **kw).parameters())
        check(f"SF-{name} == {t:,}", n == t, f"got {n:,}")
    ref = sum(p.numel() for p in B(img_size=(32, 64, 64), capacity="S", stem="iso",
                                   sr="444_222", final="trilinear_lowres_pred",
                                   skip_stem=False, out_channels=1).parameters())
    check("reference faithfulness == 4,491,809 (released file's own count)",
          ref == 4491809, f"got {ref:,}")
    # params are tile-invariant: no rel-pos tables, no PE, no token-axis projections
    a = sum(p.numel() for p in B(img_size=(32, 64, 64)).parameters())
    b = sum(p.numel() for p in B(img_size=(32, 256, 256)).parameters())
    check("params tile-invariant (t64 == t256)", a == b, f"{a:,} vs {b:,}")


# ------------------------------------------------- 3. exact reduction at init
def _pair(seed, **kw):
    torch.manual_seed(seed); stock = B(img_size=(32, 64, 64)).eval()
    torch.manual_seed(seed); mod = B(img_size=(32, 64, 64), **kw).eval()
    # copy every shared parameter so only the contribution differs
    sd = stock.state_dict()
    msd = mod.state_dict()
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            msd[k].copy_(v)
    mod.load_state_dict(msd)
    return stock, mod


def test_exact_reduction():
    print("[3] exact reduction at init (bitwise no-op behind a default-off flag)")
    x = torch.randn(2, 1, 32, 64, 64)
    xh = torch.cat([x, torch.randn_like(x)], dim=1)      # tile + halo channel
    for name, kw, inp in [
            ("C1 PDK",  dict(pdk=True, pdk_grid=(4, 4, 4)), xh),
            ("C2 DCM absolute", dict(dcm=True, dcm_frame="absolute"), x),
            ("C2 DCM surface",  dict(dcm=True, dcm_frame="surface"), x),
            ("C4 DSM",  dict(dsm=True, dsm_dilation=2), x)]:
        stock, mod = _pair(0, **kw)
        with torch.no_grad():
            d = _same(stock(x), mod(inp))
        check(f"{name}: max|delta| == 0.0", d == 0.0, f"max|delta|={d:.3e}")

    # C3 OSR: exact in real arithmetic; bitwise on CPU, <=1e-6 on CUDA (algo choice
    # for a different kernel size can reorder the summation)
    stock, mod = _pair(0, sr_overlap=True)
    with torch.no_grad():
        d = _same(stock(x), mod(x))
    check("C3 OSR: max|delta| == 0.0 on CPU", d == 0.0, f"max|delta|={d:.3e}")

    # OSR init trap: the core tap block must keep the STOCK fan-out
    from segformer3d_ink.contrib import widen_sr_conv
    torch.manual_seed(0)
    c = nn.Conv3d(32, 32, kernel_size=(1, 4, 4), stride=(1, 4, 4))
    fan = 1 * 4 * 4 * 32
    c.weight.data.normal_(0, math.sqrt(2.0 / fan))
    w = widen_sr_conv(c, (1, 4, 4))
    nz = w.weight[w.weight != 0]
    check("C3 OSR init: core block keeps stock fan-out",
          abs(nz.std().item() - math.sqrt(2.0 / fan)) < 0.15 * math.sqrt(2.0 / fan),
          f"std {nz.std().item():.5f} vs {math.sqrt(2.0/fan):.5f}")
    check("C3 OSR: extra taps are exactly zero",
          (w.weight.numel() - nz.numel()) > 0 and
          w.weight[:, :, :, :2, :2].abs().sum().item() == 0.0)


def test_live_and_grads():
    print("[4] contributions are live once trained, and receive gradients")
    x = torch.randn(2, 1, 32, 64, 64)
    xh = torch.cat([x, torch.randn_like(x)], dim=1)
    for name, kw, inp, gate in [
            ("C1 PDK", dict(pdk=True, pdk_grid=(4, 4, 4)), xh, "ctx_gate.gamma"),
            ("C2 DCM", dict(dcm=True), x, "dcm.gamma"),
            ("C4 DSM", dict(dsm=True), x, "dsm.gamma")]:
        stock, mod = _pair(0, **kw)
        with torch.no_grad():
            for n_, p in mod.named_parameters():
                if n_.endswith(gate):
                    p.add_(0.1)
        with torch.no_grad():
            d = _same(stock(x), mod(inp))
        check(f"{name}: live when gate perturbed", d > 1e-6, f"max|delta|={d:.3e}")
        mod.train(); mod.zero_grad()
        mod(inp).mean().backward()
        gs = [p.grad for n_, p in mod.named_parameters()
              if n_.endswith(gate) and p.grad is not None]
        check(f"{name}: gate receives a finite gradient",
              len(gs) > 0 and all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gs))


def test_sdpa_matches_explicit():
    print("[5] SDPA path == explicit-softmax path (unit-test path only)")
    x = torch.randn(2, 1, 32, 64, 64)
    a, b = _pair(0, use_sdpa=False)
    with torch.no_grad():
        d = _same(a(x), b(x))
    check("SDPA vs explicit softmax", d < 1e-5, f"max|delta|={d:.3e}")


def test_train_step():
    print("[6] forward + backward + one optimiser step")
    m = B(img_size=(32, 64, 64)).train()
    x = torch.randn(2, 1, 32, 64, 64)
    loss = m(x).mean(); loss.backward()
    tot = sum(1 for p in m.parameters() if p.requires_grad)
    fin = sum(1 for p in m.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    check(f"all {tot} parameter tensors get finite grads", fin == tot, f"{fin}/{tot}")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=3e-5)
    opt.step(); opt.zero_grad()
    check("second forward stays finite", torch.isfinite(m(x)).all().item())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m(x)
    check("bf16 autocast forward is finite", torch.isfinite(y.float()).all().item())


def test_shapes_all_tiles():
    print("[7] wrapper shape contract at every tile in the ladder")
    for t in (64, 128, 192, 256):
        m = B(img_size=(32, t, t)).eval()
        with torch.no_grad():
            y = m(torch.randn(1, 1, 32, t, t))
        check(f"t{t} -> (1,16,32,{t},{t})", tuple(y.shape) == (1, 16, 32, t, t), str(tuple(y.shape)))
        check(f"t{t} smoke contract prod(feat_size)==input_size[-1]",
              int(torch.tensor(m.feat_size).prod()) == m.input_size[-1])


def test_config_keys_reach_the_model():
    """THE SILENT-IGNORE TRAP (plan 3.3): models/wrapper.py forwards model keys one by
    one, so a key it does not forward is silently ignored and the run trains a different
    model than the config describes, with NO error.  Every key must change something."""
    print("[8] every model.* key provably changes the built model")
    base = B(img_size=(32, 64, 64))
    bn = sum(p.numel() for p in base.parameters())
    x = torch.randn(1, 1, 32, 64, 64)
    with torch.no_grad():
        by = base.eval()(x)
    cases = {
        "capacity": dict(capacity="M"), "stem": dict(stem="iso"),
        "sr": dict(sr="444_222"), "final": dict(final="trilinear_lowres_pred"),
        "skip_stem": dict(skip_stem=False), "dec_dim": dict(dec_dim=128),
        "dw_norm": dict(dw_norm="none"), "dec_proj_ln": dict(dec_proj_ln=False),
        "sr_overlap": dict(sr_overlap=True), "pdk": dict(pdk=True),
        "dcm": dict(dcm=True), "dsm": dict(dsm=True),
    }
    for k, kw in cases.items():
        m = B(img_size=(32, 64, 64), **kw)
        changed = sum(p.numel() for p in m.parameters()) != bn
        check(f"model.{k} changes the model", changed, "param count")
    # these two change behaviour, not parameter count -> check the forward instead
    torch.manual_seed(0); a = B(img_size=(32, 64, 64), scale_mode="head_dim").eval()
    torch.manual_seed(0); b = B(img_size=(32, 64, 64), scale_mode="heads").eval()
    with torch.no_grad():
        check("model.scale_mode changes the forward", _same(a(x), b(x)) > 1e-6)
    torch.manual_seed(0); c = B(img_size=(32, 64, 64), drop_path=0.2).train()
    check("model.drop_path installs DropPath",
          any("drop_path" in n and p.__class__.__name__ != "Identity"
              for n, p in c.named_modules()))


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in (test_geometry, test_params, test_exact_reduction, test_live_and_grads,
               test_sdpa_matches_explicit, test_train_step, test_shapes_all_tiles,
               test_config_keys_reach_the_model):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  ")
        sys.exit(1)
    print("ALL CHECKS PASSED")
