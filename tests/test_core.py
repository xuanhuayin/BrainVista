# tests/test_core.py
"""Unit tests for the input-access rules of BrainVista. Run: python -m pytest tests -q"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainvista.data.datasets import WindowDataset, anchor_bounds, run_anchors  # noqa: E402
from brainvista.data.runs import RunData  # noqa: E402
from brainvista.metrics import ForecastMetrics, rowcorr  # noqa: E402
from brainvista.models.predictor import BrainVista, build_attention_mask  # noqa: E402
from brainvista.rollout import rollout  # noqa: E402

L = 6


def _allowed(mask):
    return torch.isfinite(mask)


def _meta(L):
    pos = torch.arange(2 * L)
    return pos // 2 + 1, pos % 2


def test_s2b_mask_rules():
    M = _allowed(build_attention_mask(L, "s2b"))
    t, a = _meta(L)
    for i in range(2 * L):
        for j in range(2 * L):
            if t[j] > t[i]:
                assert not M[i, j], "no attention to later time steps"
            if a[i] == 0 and a[j] == 1:
                assert not M[i, j], "stimulus queries never read brain tokens"
    q = 2 * L - 1
    assert M[q, 0::2].all(), "target query reads every stimulus token up to u"
    assert M[q, 1:q:2].all(), "target query reads every observed brain token"


def test_vanilla_causal_lets_stimulus_read_brain():
    M = _allowed(build_attention_mask(L, "causal"))
    assert M[2, 1], "vanilla causal: stimulus at t=2 reads brain at t=1"
    assert not _allowed(build_attention_mask(L, "s2b"))[2, 1]


def test_input_modes():
    M = _allowed(build_attention_mask(L, "s2b", input_mode="brain_only"))
    q = 2 * L - 1
    assert not M[q, 0::2].any() and M[q, 1:q:2].all()
    M = _allowed(build_attention_mask(L, "s2b", input_mode="stim_only"))
    assert M[q, 0::2].all() and not M[q, 1:q:2].any()
    for mode in ("both", "brain_only", "stim_only"):
        for mask in ("s2b", "causal", "bidirectional"):
            assert _allowed(build_attention_mask(L, mask, mode)).any(1).all(), "no empty rows"


def _model(**kw):
    torch.manual_seed(0)
    return BrainVista(Dv=5, Da=3, Dw=0, Dz_in=9, Dz_out=9, d_model=32, nhead=4, nlayers=2,
                      L=L, z_splits=[4, 5], spatial_dz=16, spatial_nhead=2, dropout=0.0, **kw).eval()


def test_stimulus_stream_independent_of_brain_under_s2b():
    m = _model()
    V, A, W = torch.randn(2, L, 5), torch.randn(2, L, 3), torch.zeros(2, L, 0)
    Z1, Z2 = torch.randn(2, L - 1, 9), torch.randn(2, L - 1, 9)
    hidden = {}

    def run(Z):
        stim = m._prep_stim_tokens(V, A, W)
        brain = torch.cat([m.z_proj_in(Z), m.target_query.expand(2, 1, -1)], 1)
        seq = torch.stack([stim, brain], 2).view(2, 2 * L, -1)
        seq = seq + m.type_embed(m.type_ids)[None] + m.pos_embed(m.pos_ids)[None]
        return m.backbone(seq.transpose(0, 1), m.attn_mask).transpose(0, 1)

    h1, h2 = run(Z1), run(Z2)
    assert torch.allclose(h1[:, 0::2], h2[:, 0::2], atol=1e-5), "stimulus states depend on brain tokens"
    assert not torch.allclose(h1[:, -1], h2[:, -1]), "target query must depend on brain history"


def test_target_aligned_stimulus_is_used_and_future_is_not_input():
    m = _model()
    V, A, W = torch.randn(1, L, 5), torch.randn(1, L, 3), torch.zeros(1, L, 0)
    Z = torch.randn(1, L - 1, 9)
    V2 = V.clone(); V2[:, -1] += 1.0
    assert not torch.allclose(m(V, A, W, Z), m(V2, A, W, Z)), "tau_s^u must influence the prediction"


def _run(T=40, Dz=9):
    rng = np.random.default_rng(0)
    return RunData("sub-01", "r1", rng.standard_normal((T, 5)).astype(np.float32),
                   rng.standard_normal((T, 3)).astype(np.float32), np.zeros((T, 0), np.float32),
                   rng.standard_normal((T, Dz)).astype(np.float32))


def test_anchor_counts_match_paper():
    T, H = 40, 7
    assert len(run_anchors(T, L, 1)) == T - L                   # N^1step = T - m
    assert len(run_anchors(T, L, H)) == T - L - H + 1           # N^H = T - m - H + 1
    ds = WindowDataset([_run(T)], L)
    assert len(ds) == T - L
    for off in (1, 2, -2):
        u_min, u_max = anchor_bounds(T, L, H, off)
        assert u_min - L + 1 - off >= 0 and u_max + H - 1 - off <= T - 1


def test_window_contents_and_stimulus_cutoff():
    run = _run()
    ds0, ds1 = WindowDataset([run], L, 0), WindowDataset([run], L, 1)
    _, u = ds0.items[0]
    V, _, _, Zh, zt = ds0[0]
    assert np.array_equal(V[-1], run.V[u]), "default access: s^{<=u}"
    assert np.array_equal(Zh[-1], run.Z[u - 1]) and np.array_equal(zt, run.Z[u])
    i1 = [k for k, (_, uu) in enumerate(ds1.items) if uu == u][0]
    assert np.array_equal(ds1[i1][0][-1], run.V[u - 1]), "stim_offset=1: s^{<u}"


def test_rollout_never_reads_future_fmri():
    m = _model()
    run = _run()
    V, A, W, Z = (torch.from_numpy(x) for x in (run.V, run.A, run.W, run.Z))
    anchors = run_anchors(run.T, L, 5)
    Zp = rollout(m, V, A, W, Z, anchors, 5, use_amp=False)
    Z_mod = Z.clone()
    for u0 in anchors[:3]:
        Z_mod2 = Z.clone(); Z_mod2[u0:] = 99.0
        Zp2 = rollout(m, V, A, W, Z_mod2, [u0], 5, use_amp=False)
        assert torch.allclose(Zp[anchors.index(u0)], Zp2[0], atol=1e-5)
    # H = 1 rollout equals a direct forward pass.
    u0 = anchors[0]
    direct = m(V[u0 - L + 1:u0 + 1][None], A[u0 - L + 1:u0 + 1][None], W[u0 - L + 1:u0 + 1][None],
               Z[u0 - L + 1:u0][None])
    assert torch.allclose(Zp[0, 0], direct[0], atol=1e-5)


def test_pattern_correlation_definition():
    rng = np.random.default_rng(1)
    pred = torch.from_numpy(rng.standard_normal((3, 4, 10)).astype(np.float32))
    true = pred + 0.5 * torch.from_numpy(rng.standard_normal((3, 4, 10)).astype(np.float32))
    acc = ForecastMetrics(["a", "b"], [4, 6], H=4, keep_segments=False)
    acc.update(pred, true)
    res = acc.finalize(identification=False)
    expect = np.mean([np.corrcoef(pred[n, t], true[n, t])[0, 1] for n in range(3) for t in range(4)])
    assert abs(res["p_corr"] - expect) < 1e-5
    flat = np.corrcoef(pred.numpy().ravel(), true.numpy().ravel())[0, 1]
    assert abs(res["r"] - flat) < 1e-5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
