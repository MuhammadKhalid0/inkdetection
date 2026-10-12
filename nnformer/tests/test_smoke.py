"""Phase-N0 smoke tests for the nnFormer-Ink port (CPU).

Covers the plan's §3.4 checklist items that are testable without a GPU:
  1. shape contract at t64 (forward + backward) for stock + every ablation toggle;
  2. build-time geometry asserts at t64/t128/t192/t256 (no padding paths);
  3. measured parameter counts for capacity presets S/M/F (report these, plan §3.2);
  4. relative-position index correctness (bounds + uniqueness of extremes);
  5. reserved flags (cssa/sr_pb/deep_supervision) hard-fail until implemented.

Run:  python tests/test_smoke.py            (from the nnFormer/ root, venv active)
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnformer_ink import build_nnformer_ink, CAPACITY_PRESETS  # noqa: E402
from nnformer_ink.windows import relative_position_index  # noqa: E402

D = 32
OK = []


def check(name, fn):
    t0 = time.time()
    fn()
    OK.append(name)
    print(f"  ok  {name}  ({time.time() - t0:.1f}s)")


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_relpos_index():
    for win in [(4, 8, 8), (4, 4, 4), (4, 6, 6), (2, 8, 8)]:
        idx = relative_position_index(win)
        n = win[0] * win[1] * win[2]
        assert idx.shape == (n, n)
        table = (2 * win[0] - 1) * (2 * win[1] - 1) * (2 * win[2] - 1)
        assert int(idx.max()) == table - 1  # extreme offsets hit table ends
        assert int(idx.min()) == 0
        # zero offset (diagonal) maps to one single entry
        assert len(set(idx.diagonal().tolist())) == 1


def test_forward_backward_t64():
    m = build_nnformer_ink(img_size=(D, 64, 64), capacity="S")
    x = torch.randn(2, 1, D, 64, 64)
    y = m(x)
    assert y.shape == (2, 16, D, 64, 64), y.shape
    y.mean().backward()
    grads = [p for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0.9 * len(list(m.parameters())), "most params must receive grad"


def test_ablation_variants_t64():
    x = torch.randn(1, 1, D, 64, 64)
    for kw in [dict(gv_bottleneck=False),               # n_lvonly
               dict(skip_attention=False),               # n_skipconv
               dict(patch_size=(2, 4, 4))]:              # n_embedz2
        m = build_nnformer_ink(img_size=(D, 64, 64), capacity="S", **kw)
        assert m(x).shape == (1, 16, D, 64, 64), kw


def test_build_all_tiles():
    for t in (64, 128, 192, 256):
        m = build_nnformer_ink(img_size=(D, t, t), capacity="S")
        # geometry: stage grids halve cleanly; bottleneck = (4, t/32, t/32)
        assert m.grids[0] == (D, t // 4, t // 4)
        assert m.grids[-1] == (4, t // 32, t // 32)


def test_param_counts():
    print("    measured params (t192 build; window tables are tile-invariant):")
    counts = {}
    for cap in CAPACITY_PRESETS:
        m = build_nnformer_ink(img_size=(D, 192, 192), capacity=cap)
        m256 = build_nnformer_ink(img_size=(D, 256, 256), capacity=cap)
        counts[cap] = n_params(m)
        # E2: params are tile-invariant EXCEPT the GV bottleneck bias table,
        # which spans the (tile-dependent) bottleneck grid. Assert exactly that:
        # everything but rel-pos tables must match across tiles.
        def core(mod):
            return sum(p.numel() for n, p in mod.named_parameters()
                       if "relative_position_bias_table" not in n)
        assert core(m) == core(m256), f"{cap}: non-bias params vary with tile!"
        print(f"      nnF-{cap}: {counts[cap] / 1e6:8.2f} M (t192) | "
              f"{n_params(m256) / 1e6:8.2f} M (t256; delta = GV bias table only)")
    assert counts["S"] < counts["M"] < counts["F"]


def test_reserved_flags_fail():
    # deep supervision stays OFF for every benchmark model (plan §2.3 protocol
    # decision). cssa / sr_pb were reserved until port step 2 and are now live.
    for kw in [dict(deep_supervision=True)]:
        try:
            build_nnformer_ink(img_size=(D, 64, 64), **kw)
        except (NotImplementedError, AssertionError):
            continue
        raise AssertionError(f"{kw} should have failed (reserved flag)")


def test_forward_t192_batch1():
    m = build_nnformer_ink(img_size=(D, 192, 192), capacity="S").eval()
    with torch.no_grad():
        y = m(torch.randn(1, 1, D, 192, 192))
    assert y.shape == (1, 16, D, 192, 192)


# ---------------------------------------------------------------------------
# Port step 2 (2026-07-20): CSSA / SR-PB exact-reduction contract.
# House rule: every new component is an exact no-op at init, so any ablation
# delta is attributable to the component and not to a changed init/graph.
# ---------------------------------------------------------------------------

def _same_output(kw_a, kw_b, tile=64, seed=0, atol=0.0):
    torch.manual_seed(seed)
    ma = build_nnformer_ink(img_size=(D, tile, tile), capacity="S", **kw_a).eval()
    torch.manual_seed(seed)
    mb = build_nnformer_ink(img_size=(D, tile, tile), capacity="S", **kw_b).eval()
    # copy shared weights so only the new (zero-init) parts differ
    sa, sb = ma.state_dict(), mb.state_dict()
    for k in sa:
        if k in sb and sa[k].shape == sb[k].shape:
            sb[k] = sa[k].clone()
    mb.load_state_dict(sb)
    torch.manual_seed(123)
    x = torch.randn(1, 1, D, tile, tile)
    with torch.no_grad():
        ya, yb = ma(x), mb(x)
    return float((ya - yb).abs().max())


def test_srpb_exact_reduction():
    d = _same_output({}, {"sr_pb": True})
    assert d == 0.0, f"SR-PB not an exact no-op at init (max|delta|={d})"


def test_cssa_exact_reduction():
    d = _same_output({}, {"cssa": True})
    assert d == 0.0, f"CSSA not an exact no-op at init (max|delta|={d})"


def test_contributions_active_when_trained():
    """Non-zero parameters MUST change the output — guards against a silently
    dead branch (the 'no-op forever' failure mode)."""
    torch.manual_seed(0)
    for kw, names in ((dict(sr_pb=True), ("sr_pb.table",)),
                      (dict(cssa=True), ("cssa.gamma",))):
        m = build_nnformer_ink(img_size=(D, 64, 64), capacity="S", **kw).eval()
        x = torch.randn(1, 1, D, 64, 64)
        with torch.no_grad():
            y0 = m(x)
            n_hit = 0
            for pname, p in m.named_parameters():
                if any(pname.endswith(s) for s in names):
                    p.add_(torch.randn_like(p) * 0.5)
                    n_hit += 1
            assert n_hit > 0, f"no parameters matched {names}"
            y1 = m(x)
        assert float((y1 - y0).abs().max()) > 1e-6, f"{kw} branch is dead (output unchanged)"


def test_contributions_backward():
    for kw in (dict(sr_pb=True), dict(cssa=True), dict(sr_pb=True, cssa=True)):
        m = build_nnformer_ink(img_size=(D, 64, 64), capacity="S", **kw)
        m(torch.randn(1, 1, D, 64, 64)).mean().backward()
        # the new params must receive gradient (reachable in the graph)
        for pname, p in m.named_parameters():
            if pname.endswith("cssa.gamma") or pname.endswith("sr_pb.table"):
                assert p.grad is not None, f"{pname} got no gradient ({kw})"


if __name__ == "__main__":
    torch.manual_seed(0)
    check("relpos_index", test_relpos_index)
    check("forward_backward_t64", test_forward_backward_t64)
    check("ablation_variants_t64", test_ablation_variants_t64)
    check("build_all_tiles", test_build_all_tiles)
    check("param_counts_S_M_F", test_param_counts)
    check("reserved_flags_fail", test_reserved_flags_fail)
    check("forward_t192_batch1", test_forward_t192_batch1)
    check("srpb_exact_reduction", test_srpb_exact_reduction)
    check("cssa_exact_reduction", test_cssa_exact_reduction)
    check("contributions_active_when_trained", test_contributions_active_when_trained)
    check("contributions_backward", test_contributions_backward)
    print(f"ALL {len(OK)} CHECKS PASSED")
