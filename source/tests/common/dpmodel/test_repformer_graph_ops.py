# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-op parity: repformer graph twins vs the dense reference ops on the
identical (shape-static, center-major) edge layout. Same math, fp64 =>
rtol/atol 1e-12.
"""

import itertools

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.repformers import (
    Atten2EquiVarApply,
    Atten2Map,
    Atten2MultiHeadApply,
    LocalAtten,
    RepformerLayer,
    _cal_hg,
    _cal_hg_graph,
    _make_nei_g1,
    symmetrization_op,
    symmetrization_op_graph,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    center_edge_pairs,
)

NF, NLOC, NNEI, NG = 2, 5, 7, 6


def _mk(seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(NF, NLOC, NNEI, NG))
    h = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    return g, h, mask, sw, n_total, dst


@pytest.mark.parametrize(
    "smooth,use_sqrt_nnei", list(itertools.product([True, False], [True, False]))
)
def test_cal_hg_graph_parity(smooth, use_sqrt_nnei):
    g, h, mask, sw, n_total, dst = _mk()
    ref = _cal_hg(g, h, mask, sw, smooth=smooth, use_sqrt_nnei=use_sqrt_nnei)
    got = _cal_hg_graph(
        g.reshape(-1, NG),
        h.reshape(-1, 3),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
        smooth=smooth,
        use_sqrt_nnei=use_sqrt_nnei,
    )
    np.testing.assert_allclose(got, ref.reshape(n_total, 3, NG), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("axis_neuron", [2, 4])
def test_symmetrization_op_graph_parity(axis_neuron):
    g, h, mask, sw, n_total, dst = _mk(1)
    ref = symmetrization_op(g, h, mask, sw, axis_neuron)
    got = symmetrization_op_graph(
        g.reshape(-1, NG),
        h.reshape(-1, 3),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
        axis_neuron,
    )
    np.testing.assert_allclose(
        got, ref.reshape(n_total, axis_neuron * NG), rtol=1e-12, atol=1e-12
    )


def test_cal_hg_graph_torch():
    import torch

    g, h, mask, sw, n_total, dst = _mk(2)
    ref = _cal_hg_graph(
        g.reshape(-1, NG),
        h.reshape(-1, 3),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
    )
    got = _cal_hg_graph(
        torch.from_numpy(g.reshape(-1, NG)),
        torch.from_numpy(h.reshape(-1, 3)),
        torch.from_numpy(mask.reshape(-1)),
        torch.from_numpy(sw.reshape(-1)),
        torch.from_numpy(dst),
        n_total,
        NNEI,
    )
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12)


def _mk_layer(g1_out_conv: bool, seed: int = 0, **kwargs) -> RepformerLayer:
    cfg = {
        "rcut": 4.0,
        "rcut_smth": 0.5,
        "sel": NNEI,
        "ntypes": 2,
        "g1_dim": 8,
        "g2_dim": NG,
        "axis_neuron": 2,
        "update_chnnl_2": True,
        "g1_out_conv": g1_out_conv,
        "g1_out_mlp": True,
        "precision": "float64",
        "seed": seed,
    }
    cfg.update(kwargs)
    return RepformerLayer(**cfg)


def _mk_g1_nlist(seed=3):
    rng = np.random.default_rng(seed)
    n_total = NF * NLOC
    g1 = rng.normal(size=(n_total, 8))
    # ghost-free: neighbors index local atoms of the SAME frame
    nlist = rng.integers(0, NLOC, size=(NF, NLOC, NNEI)).astype(np.int64)
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    nlist = np.where(mask, nlist, -1)
    sw = rng.random((NF, NLOC, NNEI)) * mask
    # flat-edge view of the same topology
    src = (nlist + np.arange(NF)[:, None, None] * NLOC).reshape(-1)
    src = np.where(mask.reshape(-1), src, 0).astype(np.int64)
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    return g1, nlist, mask, sw, src, dst, n_total


@pytest.mark.parametrize("g1_out_conv", [True, False])
def test_update_g1_conv_graph_parity(g1_out_conv):
    layer = _mk_layer(g1_out_conv)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist()
    rng = np.random.default_rng(4)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = layer._update_g1_conv(gg1, g2, mask, sw)
    got = layer._update_g1_conv_graph(
        np.take(g1, src, axis=0),
        g2.reshape(-1, NG),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
    )
    np.testing.assert_allclose(got, ref.reshape(n_total, -1), rtol=1e-12, atol=1e-12)


def test_update_g2_g1g1_graph_parity():
    layer = _mk_layer(True)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(5)
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = layer._update_g2_g1g1(g1_ext, gg1, mask, sw)
    got = layer._update_g2_g1g1_graph(g1, src, dst, mask.reshape(-1), sw.reshape(-1))
    np.testing.assert_allclose(
        got, ref.reshape(n_total * NNEI, -1), rtol=1e-12, atol=1e-12
    )


def test_update_g1_conv_graph_torch():
    import torch

    layer = _mk_layer(True, seed=6)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(7)
    rng = np.random.default_rng(8)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    ref = layer._update_g1_conv_graph(
        np.take(g1, src, axis=0),
        g2.reshape(-1, NG),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
    )
    got = layer._update_g1_conv_graph(
        torch.from_numpy(np.take(g1, src, axis=0)),
        torch.from_numpy(g2.reshape(-1, NG)),
        torch.from_numpy(mask.reshape(-1)),
        torch.from_numpy(sw.reshape(-1)),
        torch.from_numpy(dst),
        n_total,
        NNEI,
    )
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)


def _pairs(mask, dst, n_total):
    q_e, k_e, pm = center_edge_pairs(
        dst,
        mask.reshape(-1),
        n_total,
        include_self=True,
        ordered=True,
        static_nnei=NNEI,
    )
    return q_e, k_e, pm


@pytest.mark.parametrize(
    "has_gate,smooth", [(True, True), (False, True), (True, False)]
)
def test_atten2map_parity(has_gate, smooth):
    rng = np.random.default_rng(6)
    a2m = Atten2Map(
        NG, 4, 2, has_gate=has_gate, smooth=smooth, precision="float64", seed=7
    )
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    ref = a2m.call(g2, h2, mask, sw)  # (nf, nloc, nnei, nnei, nh)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    got = a2m.call_graph(
        g2.reshape(-1, NG),
        h2.reshape(-1, 3),
        sw.reshape(-1),
        q_e,
        k_e,
        pm,
        NF * NLOC * NNEI,
        NNEI,
    )  # (P, nh)
    slot = np.arange(NF * NLOC * NNEI) % NNEI
    ctr = dst[np.asarray(q_e)]
    f, l = ctr // NLOC, ctr % NLOC
    ref_pairs = ref[f, l, slot[np.asarray(q_e)], slot[np.asarray(k_e)], :]
    # NOTE: even for pairs whose query edge is padding, both sides collapse to
    # an exact 0.0 (dense: post-softmax where(mask, 0, ...); graph: pair_mask
    # multiply, and in the smooth branch the fully-masked segment_softmax
    # group also yields exact 0.0), so no restriction to real-query pairs is
    # needed here -- verified empirically before relying on it.
    np.testing.assert_allclose(np.asarray(got), ref_pairs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "has_gate,smooth", [(True, True), (False, True), (True, False)]
)
def test_atten2_mh_apply_parity(has_gate, smooth):
    rng = np.random.default_rng(9)
    nh = 3
    a2m = Atten2Map(
        NG, 4, nh, has_gate=has_gate, smooth=smooth, precision="float64", seed=10
    )
    mha = Atten2MultiHeadApply(NG, nh, precision="float64", seed=11)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    e_tot = NF * NLOC * NNEI
    ref_AA = a2m.call(g2, h2, mask, sw)  # (nf, nloc, nnei, nnei, nh)
    ref = mha.call(ref_AA, g2)  # (nf, nloc, nnei, ng2)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    got_AA = a2m.call_graph(
        g2.reshape(-1, NG), h2.reshape(-1, 3), sw.reshape(-1), q_e, k_e, pm, e_tot, NNEI
    )
    got = mha.call_graph(got_AA, g2.reshape(-1, NG), q_e, k_e, e_tot)
    np.testing.assert_allclose(
        np.asarray(got), ref.reshape(e_tot, NG), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    "has_gate,smooth", [(True, True), (False, True), (True, False)]
)
def test_atten2_ev_apply_parity(has_gate, smooth):
    rng = np.random.default_rng(12)
    nh = 3
    a2m = Atten2Map(
        NG, 4, nh, has_gate=has_gate, smooth=smooth, precision="float64", seed=13
    )
    ev = Atten2EquiVarApply(NG, nh, precision="float64", seed=14)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    e_tot = NF * NLOC * NNEI
    ref_AA = a2m.call(g2, h2, mask, sw)  # (nf, nloc, nnei, nnei, nh)
    ref = ev.call(ref_AA, h2)  # (nf, nloc, nnei, 3)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    got_AA = a2m.call_graph(
        g2.reshape(-1, NG), h2.reshape(-1, 3), sw.reshape(-1), q_e, k_e, pm, e_tot, NNEI
    )
    got = ev.call_graph(got_AA, h2.reshape(-1, 3), q_e, k_e, e_tot)
    np.testing.assert_allclose(
        np.asarray(got), ref.reshape(e_tot, 3), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("smooth", [True, False])
def test_local_atten_parity(smooth):
    la = LocalAtten(8, 4, 2, smooth=smooth, precision="float64", seed=15)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(20)
    g1_ext = g1.reshape(NF, NLOC, 8)
    gg1 = _make_nei_g1(g1_ext, np.where(mask, nlist, 0))
    ref = la.call(g1_ext, gg1, mask, sw)  # (nf, nloc, ng1)
    got = la.call_graph(
        g1,
        np.take(g1, src, axis=0),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        n_total,
        NNEI,
    )
    np.testing.assert_allclose(
        np.asarray(got), ref.reshape(n_total, 8), rtol=1e-12, atol=1e-12
    )


def test_local_atten_below_phantom_dense_parity():
    """OutisLi review: finite valid projection weights can push every smooth
    logit below ``-attnw_shift``.  With ``n_real == sel`` the signed phantom
    count is 0 -- the denominator is a plain positive softmax sum -- and the
    graph route must match dense EXACTLY (the always-on floor used to return
    ~0.0018 where dense gives 1.0).
    """
    la = LocalAtten(1, 1, 1, smooth=True, precision="float64", seed=1)
    la.mapq.w = np.array([[1.0]])
    la.mapkv.w = np.array([[-30.0, 1.0]])  # key -30, value 1
    la.head_map.w = np.array([[1.0]])  # identity head
    la.head_map.b = np.array([0.0])
    nf, nloc, nnei = 1, 1, 2
    g1 = np.ones((nf * nloc, 1))
    gg1 = np.ones((nf, nloc, nnei, 1))
    mask = np.ones((nf, nloc, nnei), dtype=bool)
    sw = np.ones((nf, nloc, nnei))
    ref = la.call(g1.reshape(nf, nloc, 1), gg1, mask, sw)  # == [[[1.0]]]
    dst = np.zeros(nnei, dtype=np.int64)
    got = la.call_graph(
        g1,
        gg1.reshape(-1, 1),
        mask.reshape(-1),
        sw.reshape(-1),
        dst,
        nf * nloc,
        nnei,  # sel == n_real: phantom count 0
    )
    np.testing.assert_allclose(np.asarray(got), ref.reshape(1, 1), rtol=1e-12, atol=0.0)
    # anti-vacuity: the logits really are below -attnw_shift and the dense
    # result is the nontrivial value from the review
    np.testing.assert_allclose(np.asarray(got), [[1.0]], rtol=1e-12)


def test_local_atten_cutoff_crossing_below_phantom_continuous():
    """OutisLi round 5: an edge crossing the cutoff must not JUMP the output
    when the surviving logits sit below ``-attnw_shift``.

    Same LocalAtten as :func:`test_local_atten_below_phantom_dense_parity`
    (every smooth logit ~= -30 < -attnw_shift) but with ``sel = 1``: outside
    the cutoff (one edge, phantom count 0) the output is exactly 1.0; a
    second edge entering with ``sw = s -> 0+`` flips the count to -1.  The
    count-gated denominator floor produced 0.00181599 just inside (limiting
    jump -0.998); the slot-occupancy denominator must converge back to the
    outside value.
    """
    la = LocalAtten(1, 1, 1, smooth=True, precision="float64", seed=1)
    la.mapq.w = np.array([[1.0]])
    la.mapkv.w = np.array([[-30.0, 1.0]])
    la.head_map.w = np.array([[1.0]])
    la.head_map.b = np.array([0.0])
    sel = 1
    n_total = 1

    def run(sw_edges: np.ndarray) -> float:
        nnei = sw_edges.shape[0]
        g1 = np.ones((n_total, 1))
        gg1 = np.ones((nnei, 1))
        mask = np.ones(nnei, dtype=bool)
        dst = np.zeros(nnei, dtype=np.int64)
        out = la.call_graph(g1, gg1, mask, sw_edges, dst, n_total, sel)
        return float(np.asarray(out)[0, 0])

    outside = run(np.array([1.0]))
    np.testing.assert_allclose(outside, 1.0, rtol=1e-12)
    prev_gap = None
    for s in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        inside = run(np.array([1.0, s]))
        assert np.isfinite(inside)
        gap = abs(inside - outside)
        if prev_gap is not None:
            assert gap < prev_gap  # converging, not plateauing at a jump
        prev_gap = gap
    assert prev_gap < 1e-7


def test_atten2map_below_phantom_dense_parity():
    """Same below-``-attnw_shift`` regime for the pair attention map: with
    all key slots real (phantom count 0) graph must equal dense exactly.
    """
    a2m = Atten2Map(1, 1, 1, has_gate=False, smooth=True, precision="float64", seed=2)
    a2m.mapqk.w = np.array([[1.0, -30.0]])  # query 1, key -30 -> logits -30
    nf, nloc, nnei = 1, 1, 2
    g2 = np.ones((nf, nloc, nnei, 1))
    h2 = np.zeros((nf, nloc, nnei, 3))
    h2[..., 0] = 1.0
    mask = np.ones((nf, nloc, nnei), dtype=bool)
    sw = np.ones((nf, nloc, nnei))
    ref = a2m.call(g2, h2, mask, sw)  # (nf, nloc, nnei, nnei, nh)
    n_total = nf * nloc
    dst = np.repeat(np.arange(n_total, dtype=np.int64), nnei)
    q_e, k_e, pm = center_edge_pairs(
        dst,
        mask.reshape(-1),
        n_total,
        include_self=True,
        ordered=True,
        static_nnei=nnei,
    )
    got = a2m.call_graph(
        g2.reshape(-1, 1),
        h2.reshape(-1, 3),
        sw.reshape(-1),
        q_e,
        k_e,
        pm,
        nf * nloc * nnei,
        nnei,
    )
    ref_pairs = ref[0, 0, np.asarray(q_e) % nnei, np.asarray(k_e) % nnei, :]
    np.testing.assert_allclose(np.asarray(got), ref_pairs, rtol=1e-12, atol=0.0)
    # anti-vacuity: weight 0.5 per key slot times h2h2t = 1/sqrt(3)
    np.testing.assert_allclose(np.asarray(got), 0.5 / np.sqrt(3.0), rtol=1e-12)


def test_atten2map_graph_torch():
    import torch

    rng = np.random.default_rng(16)
    a2m = Atten2Map(NG, 4, 2, has_gate=True, smooth=True, precision="float64", seed=17)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    mask = rng.random((NF, NLOC, NNEI)) > 0.3
    sw = rng.random((NF, NLOC, NNEI)) * mask
    n_total = NF * NLOC
    dst = np.repeat(np.arange(n_total, dtype=np.int64), NNEI)
    q_e, k_e, pm = _pairs(mask, dst, n_total)
    e_tot = NF * NLOC * NNEI
    ref = a2m.call_graph(
        g2.reshape(-1, NG), h2.reshape(-1, 3), sw.reshape(-1), q_e, k_e, pm, e_tot, NNEI
    )
    got = a2m.call_graph(
        torch.from_numpy(g2.reshape(-1, NG)),
        torch.from_numpy(h2.reshape(-1, 3)),
        torch.from_numpy(sw.reshape(-1)),
        torch.from_numpy(q_e),
        torch.from_numpy(k_e),
        torch.from_numpy(pm),
        e_tot,
        NNEI,
    )
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Whole-layer parity: RepformerLayer.call_graph vs RepformerLayer.call
# --------------------------------------------------------------------------

# Toggle matrix; each entry is the full set of RepformerLayer kwargs applied
# on top of _mk_layer's base config (g1_out_conv defaults to True unless
# overridden). "conv_only" and "no_g2_attn" additionally exercise
# pairs=None, since neither sets update_g2_has_attn/update_h2 True.
_LAYER_CASES = {
    "defaults": {},
    "no_g2_attn": {"update_g2_has_attn": False},
    "update_h2": {"update_h2": True},
    "conv_only": {
        "update_g1_has_conv": True,
        "g1_out_conv": False,  # feed conv output into g1_mlp (else g1_mlp is empty)
        "update_g1_has_drrd": False,
        "update_g1_has_grrg": False,
        "update_g1_has_attn": False,
        "update_g2_has_g1g1": False,
        "update_g2_has_attn": False,
        "update_h2": False,
    },
    "no_g1_out": {"g1_out_conv": False, "g1_out_mlp": False},
    "no_chnnl_2_no_gg1": {
        # Exercises update_chnnl_2=False (skips g2/h2 updates) and cal_gg1=False (gg1=None).
        # With g1_out_mlp=False, g1_mlp starts with [g1] identity seed for xp.concat.
        "update_chnnl_2": False,
        "update_g1_has_conv": False,
        "update_g1_has_grrg": False,
        "update_g1_has_drrd": False,
        "update_g1_has_attn": False,
        "update_g2_has_g1g1": False,
        "update_g2_has_attn": False,
        "update_h2": False,
        "g1_out_conv": False,
        "g1_out_mlp": False,
    },
}
# cases where pairs=None is exercised (neither update_g2_has_attn nor
# update_h2 is set, so center_edge_pairs is not required)
_PAIRS_NONE_CASES = {"conv_only", "no_g2_attn"}


@pytest.mark.parametrize("case_name", list(_LAYER_CASES))
def test_repformer_layer_call_graph_parity(case_name):
    kwargs = dict(_LAYER_CASES[case_name])
    g1_out_conv = kwargs.pop("g1_out_conv", True)
    layer = _mk_layer(g1_out_conv, seed=21, **kwargs)

    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(23)
    rng = np.random.default_rng(24)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    g1_ext = g1.reshape(NF, NLOC, 8)
    nlist0 = np.where(nlist == -1, 0, nlist)

    ref_g1, ref_g2, ref_h2 = layer.call(g1_ext, g2, h2, nlist0, mask, sw)

    g2_flat = g2.reshape(-1, NG)
    h2_flat = h2.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    sw_flat = sw.reshape(-1)

    if case_name in _PAIRS_NONE_CASES:
        assert not (layer.update_g2_has_attn or layer.update_h2)
        pairs = None
    else:
        pairs = _pairs(mask, dst, n_total)

    got_g1, got_g2, got_h2 = layer.call_graph(
        g1, g2_flat, h2_flat, src, dst, mask_flat, sw_flat, n_total, NNEI, pairs
    )

    np.testing.assert_allclose(
        got_g1, ref_g1.reshape(n_total, -1), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(got_g2, ref_g2.reshape(-1, NG), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_h2, ref_h2.reshape(-1, 3), rtol=1e-12, atol=1e-12)


def test_repformer_layer_call_graph_torch():
    import torch

    layer = _mk_layer(True, seed=25)
    g1, nlist, mask, sw, src, dst, n_total = _mk_g1_nlist(26)
    rng = np.random.default_rng(27)
    g2 = rng.normal(size=(NF, NLOC, NNEI, NG))
    h2 = rng.normal(size=(NF, NLOC, NNEI, 3))
    g2_flat = g2.reshape(-1, NG)
    h2_flat = h2.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    sw_flat = sw.reshape(-1)
    pairs = _pairs(mask, dst, n_total)

    ref = layer.call_graph(
        g1, g2_flat, h2_flat, src, dst, mask_flat, sw_flat, n_total, NNEI, pairs
    )

    t_pairs = tuple(torch.from_numpy(np.asarray(x)) for x in pairs)
    got = layer.call_graph(
        torch.from_numpy(g1),
        torch.from_numpy(g2_flat),
        torch.from_numpy(h2_flat),
        torch.from_numpy(src),
        torch.from_numpy(dst),
        torch.from_numpy(mask_flat),
        torch.from_numpy(sw_flat),
        n_total,
        NNEI,
        t_pairs,
    )
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_allclose(g.numpy(), r, rtol=1e-12, atol=1e-12)


def test_local_atten_gradient_continuous_below_phantom():
    """OutisLi round 6: C1 across the below-phantom surface through the real
    LocalAtten path.  sel=1, sw=[1, 1], logits [-20 +/- eps, -18], values
    [1, 2] (his construction): the relu bracket kept the OUTPUT continuous
    (~2.1353353) but jumped d(output)/d(logit) from -0.009157818652523 to
    -0.153650933331561.  The C1 tail matches the in-design (right) slope
    from both sides.
    """
    la = LocalAtten(1, 1, 1, smooth=True, precision="float64", seed=1)
    la.mapq.w = np.array([[1.0]])
    # key = x (logit = q*k = x); value = 0.5*x + 11 -> v(-20)=1, v(-18)=2
    la.mapkv.w = np.array([[1.0, 0.5]])
    la.mapkv.b = np.array([0.0, 11.0])
    la.head_map.w = np.array([[1.0]])
    la.head_map.b = np.array([0.0])
    n_total, sel = 1, 1

    def out(x1: float) -> float:
        gg1 = np.array([[x1], [-18.0]])
        o = la.call_graph(
            np.ones((n_total, 1)),
            gg1,
            np.ones(2, dtype=bool),
            np.ones(2),
            np.zeros(2, dtype=np.int64),
            n_total,
            sel,
        )
        return float(np.asarray(o)[0, 0])

    # value continuity at the surface (his repro printed ~2.135335)
    np.testing.assert_allclose(out(-20.0), 2.1353, rtol=1e-3)
    d = 1e-6
    left = (out(-20.0 - d) - out(-20.0 - 3.0 * d)) / (2.0 * d)
    right = (out(-20.0 + 3.0 * d) - out(-20.0 + d)) / (2.0 * d)
    # the in-design branch is unchanged, so the right slope equals his
    # measured -0.153650933331561 plus the value-chain term 0.5 * w1
    # (= exp(-2)/2: our value wiring v = 0.5*x + 11 varies with the swept
    # feature); the relu bracket jumped the attention part of the left
    # slope to -0.009157818652523
    np.testing.assert_allclose(
        right, -0.153650933331561 + 0.5 * np.exp(-2.0), rtol=1e-3
    )
    np.testing.assert_allclose(left, right, rtol=5e-3)


def test_atten2map_gradient_continuous_below_phantom():
    """Same C1-at-the-surface guarantee through the pair attention map:
    sweep one edge feature so a single pair logit crosses -attnw_shift while
    the segment stays binding (sel=1 < 2 keys); the FD slope of the
    attention map must match across the crossing.
    """
    a2m = Atten2Map(1, 1, 1, has_gate=False, smooth=True, precision="float64", seed=2)
    # q = g2, k = -g2 -> logit(q,k) = -g_q*g_k: negative-definite pairing
    # keeps every pair logit NEAR the -20 phantom level (a positive pairing
    # pins self-logits >= 0, drowning the crossing pair 20+ units below the
    # segment max where the softmax renders it invisible to FD)
    a2m.mapqk.w = np.array([[1.0, -1.0]])
    n_total, nnei = 1, 2
    dst = np.repeat(np.arange(n_total, dtype=np.int64), nnei)
    mask = np.ones(nnei, dtype=bool)
    q_e, k_e, pm = center_edge_pairs(
        dst, mask, n_total, include_self=True, ordered=True, static_nnei=nnei
    )
    h2 = np.zeros((nnei, 3))
    h2[:, 0] = 1.0

    def out(x: float) -> float:
        # pair logits: {-x*x, -5x, -5x, -25}; -5x crosses -20 at x = 4 with
        # the self pair at -16 in-design and the (1,1) pair statically below
        g2 = np.array([[x], [5.0]])
        att = a2m.call_graph(g2, h2, np.ones(nnei), q_e, k_e, pm, n_total * nnei, 1)
        return float(np.sum(np.asarray(att)))

    d = 1e-6
    left = (out(4.0 - 3.0 * d) - out(4.0 - d)) / (-2.0 * d)
    right = (out(4.0 + d) - out(4.0 + 3.0 * d)) / (-2.0 * d)
    assert abs(left) > 1e-6 and abs(right) > 1e-6
    np.testing.assert_allclose(left, right, rtol=5e-3)


def test_local_atten_float32_high_logit_backward_finite():
    """OutisLi round 7 model-level repro: float32 LocalAtten.call_graph with
    sel == n_real == 2, sw = [1, 1] and raw logits [68, 69] (in-design,
    count 0).  Dense and graph forwards agree (~68.7311) but the graph
    backward produced NaN g1/gg1 gradients through the inactive tail's
    ``1 / ph_e`` overflow; gradients must be finite and match dense.
    """
    import torch

    la = LocalAtten(1, 1, 1, smooth=True, precision="float32", seed=1)
    la.mapq.w = np.array([[1.0]], dtype=np.float32)
    la.mapkv.w = np.array([[1.0, 1.0]], dtype=np.float32)
    la.mapkv.b = np.array([0.0, 0.0], dtype=np.float32)
    la.head_map.w = np.array([[1.0]], dtype=np.float32)
    la.head_map.b = np.array([0.0], dtype=np.float32)
    n_total, nnei, sel = 1, 2, 2

    gg1_np = np.array([[68.0], [69.0]], dtype=np.float32)
    mask_np = np.ones(nnei, dtype=bool)
    sw_np = np.ones(nnei, dtype=np.float32)

    # graph route, torch float32, gradients w.r.t. g1 and gg1
    g1_t = torch.ones(n_total, 1, dtype=torch.float32, requires_grad=True)
    gg1_t = torch.from_numpy(gg1_np).clone().requires_grad_(True)
    out_g = la.call_graph(
        g1_t,
        gg1_t,
        torch.from_numpy(mask_np),
        torch.from_numpy(sw_np),
        torch.zeros(nnei, dtype=torch.int64),
        n_total,
        sel,
    )
    out_g.sum().backward()
    assert torch.all(torch.isfinite(g1_t.grad))
    assert torch.all(torch.isfinite(gg1_t.grad))

    # dense reference gradients on the same weights
    g1_d = torch.ones(1, 1, 1, dtype=torch.float32, requires_grad=True)
    gg1_d = torch.from_numpy(gg1_np.reshape(1, 1, nnei, 1)).clone().requires_grad_(True)
    out_d = la.call(
        g1_d,
        gg1_d,
        torch.from_numpy(mask_np.reshape(1, 1, nnei)),
        torch.from_numpy(sw_np.reshape(1, 1, nnei)),
    )
    out_d.sum().backward()
    np.testing.assert_allclose(float(out_g.sum()), float(out_d.sum()), rtol=1e-6)
    np.testing.assert_allclose(
        g1_t.grad.numpy().reshape(-1), g1_d.grad.numpy().reshape(-1), rtol=1e-4
    )
    np.testing.assert_allclose(
        gg1_t.grad.numpy().reshape(-1), gg1_d.grad.numpy().reshape(-1), rtol=1e-4
    )


def test_local_atten_float32_log_spaced_sw_matches_float64():
    """OutisLi round 8 forward repro: float32 LocalAtten.call_graph with
    smooth cutoff weights spanning decades (sw = [1, 0.1, 0.01, 5e-7]) and
    raw logits -30.  The float32 active-set selection used to pick the wrong
    water-filling cut (~23% forward error, 0.0612 vs 0.0792); float32 must
    now match the float64 evaluation of the same quantized inputs.
    """
    n_total, nnei, sel = 1, 4, 3
    sw32 = np.array([1.0, 0.1, 0.01, 5e-7], dtype=np.float32)

    def run(prec: str) -> float:
        la = LocalAtten(1, 1, 1, smooth=True, precision=prec, seed=1)
        dt = np.float32 if prec == "float32" else np.float64
        la.mapq.w = np.array([[1.0]], dtype=dt)
        la.mapkv.w = np.array([[1.0, 1.0]], dtype=dt)
        la.mapkv.b = np.array([0.0, 0.0], dtype=dt)
        la.head_map.w = np.array([[1.0]], dtype=dt)
        la.head_map.b = np.array([0.0], dtype=dt)
        o = la.call_graph(
            np.ones((n_total, 1), dtype=dt),
            np.full((nnei, 1), -30.0, dtype=dt),
            np.ones(nnei, dtype=bool),
            sw32.astype(dt),
            np.zeros(nnei, dtype=np.int64),
            n_total,
            sel,
        )
        return float(np.asarray(o)[0, 0])

    np.testing.assert_allclose(run("float32"), run("float64"), rtol=1e-3)
