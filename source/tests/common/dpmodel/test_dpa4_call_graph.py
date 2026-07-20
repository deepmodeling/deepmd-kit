# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-route regression tests for the dpmodel DPA4 descriptor."""

import dataclasses

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
    _graph_from_padded_nlist,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)


def build_neighbor_list_np(coord, rcut, nnei):
    """Build a padded, distance-sorted gas-phase neighbor list (no PBC).

    Parameters
    ----------
    coord
        Coordinates with shape (nf, nloc, 3).
    rcut
        Cutoff radius.
    nnei
        Number of neighbor slots; pads with -1.

    Returns
    -------
    np.ndarray
        Neighbor list with shape (nf, nloc, nnei) holding local indices.
    """
    nf, nloc, _ = coord.shape
    nlist = -np.ones((nf, nloc, nnei), dtype=np.int64)
    for f in range(nf):
        dist = np.linalg.norm(coord[f][:, None, :] - coord[f][None, :, :], axis=-1)
        for i in range(nloc):
            neighbors = [
                (dist[i, j], j) for j in range(nloc) if j != i and dist[i, j] < rcut
            ]
            neighbors.sort()
            for slot, (_, j) in enumerate(neighbors[:nnei]):
                nlist[f, i, slot] = j
    return nlist


def build_sparse_edges_from_nlist(coord, nlist):
    """Extract the valid physical edges of a padded neighbor list.

    The padded layout keeps one slot per neighbor (``-1`` marks padding). The
    graph-route edge contract -- edges packed into a :class:`NeighborGraph`
    and consumed by ``call_graph`` -- is one explicit edge per kept slot,
    indexing the flattened frame-major node axis (``node = f * nloc + i``).
    The edge vector points from the center toward the neighbor, matching the
    padded path's ``r_j - r_i``.

    Parameters
    ----------
    coord
        Coordinates with shape (nf, nloc, 3).
    nlist
        Neighbor list with shape (nf, nloc, nnei); -1 marks padding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``edge_index`` with shape (2, E) (rows are src, dst) and ``edge_vec``
        with shape (E, 3), aligned on the same edge axis in row-major
        ``(frame, center, slot)`` order.
    """
    nf, nloc, nnei = nlist.shape
    src, dst, vec = [], [], []
    for f in range(nf):
        for i in range(nloc):
            for s in range(nnei):
                j = int(nlist[f, i, s])
                if j < 0:
                    continue
                src.append(f * nloc + j)
                dst.append(f * nloc + i)
                vec.append(coord[f, j] - coord[f, i])
    edge_index = np.asarray([src, dst], dtype=np.int64)  # (2, E)
    edge_vec = np.asarray(vec, dtype=np.float64)  # (E, 3)
    return edge_index, edge_vec


def make_descriptor() -> DescrptDPA4:
    return DescrptDPA4(
        ntypes=3,
        sel=8,
        rcut=4.0,
        channels=16,
        n_radial=8,
        lmax=2,
        mmax=1,
        n_blocks=2,
        precision="float64",
        seed=7,
        random_gamma=False,
    )


def make_inputs(seed=7, nf=2, nloc=6, rcut=4.0, nnei=8, ntypes=3):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 3.5, size=(nf, nloc, 3))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    nlist = build_neighbor_list_np(coord, rcut, nnei)
    return coord, atype, nlist


def make_graph_from_nlist(coord, nlist):
    """Build a ghost-free NeighborGraph from a gas-phase local nlist."""
    nf, nloc, _ = nlist.shape
    edge_index, edge_vec = build_sparse_edges_from_nlist(coord, nlist)
    return NeighborGraph(
        n_node=np.full((nf,), nloc, dtype=np.int64),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=np.ones(edge_index.shape[1], dtype=bool),
    )


def _run_graph(dd, coord, atype, nlist, permute_seed=None):
    nf, nloc = atype.shape
    graph = make_graph_from_nlist(coord, nlist)
    if permute_seed is not None:
        perm = np.random.default_rng(permute_seed).permutation(
            graph.edge_index.shape[1]
        )
        graph = dataclasses.replace(
            graph,
            edge_index=graph.edge_index[:, perm],
            edge_vec=graph.edge_vec[perm],
            edge_mask=graph.edge_mask[perm],
        )
    out, rot_mat = dd.call_graph(graph, atype.reshape(-1))
    assert rot_mat is None
    return np.asarray(out)


def make_message_sensitive_descriptor(seed: int = 99) -> DescrptDPA4:
    """A ``make_descriptor()`` variant with its zero-init residuals jittered.

    Two calls with the same ``seed`` are bit-identical (deserialize is
    deterministic given the jittered parameter tree), so a pair of
    independently constructed descriptors used to isolate the effect of
    ``exclude_types`` still share every other weight.
    """
    data = make_descriptor().serialize()
    jitter_zero_arrays(data, np.random.default_rng(seed))
    return DescrptDPA4.deserialize(data)


def make_spin_descriptor(seed: int = 99) -> DescrptDPA4:
    """A ``use_spin`` variant of ``make_message_sensitive_descriptor()``.

    Native spin (``use_spin`` on type 0) is enabled on top of the same
    zero-init-residual jitter used by ``make_message_sensitive_descriptor``,
    so the descriptor is sensitive to both edges/messages AND spin -- a bare
    ``make_descriptor()`` variant would be architecturally spin-independent
    for the same reason it is edge-independent (see that helper's
    docstring), which would make a spin-sensitivity check vacuous.
    """
    dd = DescrptDPA4(
        ntypes=3,
        sel=8,
        rcut=4.0,
        channels=16,
        n_radial=8,
        lmax=2,
        mmax=1,
        n_blocks=2,
        precision="float64",
        seed=7,
        random_gamma=False,
        use_spin=[True, False, False],
    )
    data = dd.serialize()
    jitter_zero_arrays(data, np.random.default_rng(seed))
    return DescrptDPA4.deserialize(data)


def test_call_graph_spin_sensitivity() -> None:
    """call_graph(spin=...) must change the output (teeth: spin reaches the trunk)."""
    dd = make_spin_descriptor()
    coord, atype, nlist = make_inputs()
    graph = make_graph_from_nlist(coord, nlist)
    atype_flat = atype.reshape(-1)
    rng = np.random.default_rng(3)
    spin = rng.normal(size=(atype_flat.shape[0], 3))
    out0, _ = dd.call_graph(graph, atype_flat)
    out1, _ = dd.call_graph(graph, atype_flat, spin=spin)
    assert not np.allclose(np.asarray(out0), np.asarray(out1))


def test_call_graph_spin_matches_dense_adapter() -> None:
    """Graph-lower spin path == dense adapter spin path (shared trunk, 1e-12)."""
    dd = make_spin_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    rng = np.random.default_rng(3)
    spin = rng.normal(size=(nf * nloc, 3))
    ref, *_ = dd.call(
        coord.reshape(nf, -1), atype, nlist, spin=spin.reshape(nf, nloc, 3)
    )
    graph, atype_flat = _graph_from_padded_nlist(coord, atype, nlist, None)
    out, _ = dd.call_graph(graph, atype_flat, spin=spin)
    np.testing.assert_allclose(
        np.asarray(out).reshape(nf, nloc, -1),
        np.asarray(ref),
        rtol=1e-12,
        atol=1e-12,
    )


def test_call_graph_matches_dense() -> None:
    # Same physical edges (non-binding sel) => same descriptor within fp64
    # scatter-reassociation tolerance; output is flat (N, C).
    #
    # NOTE: uses make_message_sensitive_descriptor(), not the plain
    # make_descriptor() fixture -- see that helper's docstring. A bare
    # make_descriptor() is architecturally edge-independent (multiple
    # zero-init residual output projections), so this parity check would
    # pass trivially (0.0 == 0.0) regardless of whether call_graph's edge
    # handling is correct.
    dd = make_message_sensitive_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    out_dense = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
    out_graph = _run_graph(dd, coord, atype, nlist)
    assert out_graph.shape == (nf * nloc, dd.get_dim_out())
    np.testing.assert_allclose(
        out_graph.reshape(nf, nloc, -1), out_dense, rtol=1e-10, atol=1e-12
    )


def test_call_graph_matches_dense_permuted_edges() -> None:
    # Arbitrary edge order (the graph contract) must not change the result.
    #
    # NOTE: uses make_message_sensitive_descriptor() -- see
    # test_call_graph_matches_dense's NOTE above; a bare make_descriptor()
    # is edge-independent, so permuting its (irrelevant) edges is vacuous.
    dd = make_message_sensitive_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    out_dense = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
    out_graph = _run_graph(dd, coord, atype, nlist, permute_seed=31)
    np.testing.assert_allclose(
        out_graph.reshape(nf, nloc, -1), out_dense, rtol=1e-10, atol=1e-12
    )


def test_message_sensitive_fixture_is_edge_dependent() -> None:
    """Pin that make_message_sensitive_descriptor() is edge-dependent.

    The two parity tests above (and the exclude_types test below) are only
    meaningful because make_message_sensitive_descriptor() jitters DPA4's
    zero-init residual projections so the output actually depends on
    edges/messages -- a bare make_descriptor() does not (see
    jitter_zero_arrays's docstring in dpa4_fixtures.py). This test durably
    pins that edge-sensitivity: it runs call_graph once as-is and once with every
    edge masked out, and asserts the outputs differ by a non-trivial
    margin. If a future change to the fixture (or to DPA4's zero-init
    scheme) silently made it edge-independent again, this test fails loud
    instead of the parity tests above going quietly vacuous.
    """
    dd = make_message_sensitive_descriptor()
    coord, atype, nlist = make_inputs()
    graph = make_graph_from_nlist(coord, nlist)
    graph_no_edges = dataclasses.replace(
        graph, edge_mask=np.zeros_like(graph.edge_mask)
    )
    out_with_edges, _ = dd.call_graph(graph, atype.reshape(-1))
    out_no_edges, _ = dd.call_graph(graph_no_edges, atype.reshape(-1))
    out_with_edges = np.asarray(out_with_edges)
    out_no_edges = np.asarray(out_no_edges)
    assert np.max(np.abs(out_with_edges - out_no_edges)) > 1e-6


def test_call_graph_exclude_types_matches_dense() -> None:
    # Pair exclusion: canonical apply_pair_exclusion on the graph must equal
    # the dense build_type_exclude_mask route. Also pins the empty-exclusion
    # branch via the tests above.
    #
    # NOTE: uses make_message_sensitive_descriptor(), not the plain
    # make_descriptor() fixture, so the anti-vacuity assertion below is
    # meaningful -- see that helper's docstring. A bare make_descriptor()
    # is architecturally edge-independent (multiple zero-init residual
    # output projections), so exclude_types provably cannot change its
    # output; asserting non-vacuity against it would always fail regardless
    # of whether exclusion is correctly wired.
    dd = make_message_sensitive_descriptor()
    dd.reinit_exclude([(0, 1)])
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    out_dense = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
    out_graph = _run_graph(dd, coord, atype, nlist)
    np.testing.assert_allclose(
        out_graph.reshape(nf, nloc, -1), out_dense, rtol=1e-10, atol=1e-12
    )
    # anti-vacuity: exclusion must actually change the descriptor
    dd2 = make_message_sensitive_descriptor()
    out_noexcl = np.asarray(dd2.call(coord.reshape(nf, -1), atype, nlist)[0])
    assert not np.allclose(out_dense, out_noexcl, rtol=1e-6, atol=1e-8)


def test_call_graph_comm_dict_raises() -> None:
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    graph = make_graph_from_nlist(coord, nlist)
    with pytest.raises(NotImplementedError, match="comm_dict"):
        dd.call_graph(graph, atype.reshape(-1), comm_dict={"dummy": None})


def test_capability_flags() -> None:
    dd = make_descriptor()
    assert dd.uses_graph_lower() is True
    assert dd.uses_compact_edge_pairs() is False
    assert dd.graph_type_embedding_table() is None
    dd.disable_graph_lower()
    assert dd.uses_graph_lower() is False


def test_supports_native_spin_capability_gate() -> None:
    """Pin the explicit ``supports_native_spin`` capability method (not duck-typed).

    ``DPAtomicModel.supports_native_spin`` is cached from the descriptor's
    ``supports_native_spin()`` method (see ``DPAtomicModel.__init__``), not
    from a `hasattr` probe on an internal descriptor attribute. A DPA4
    descriptor reports the capability explicitly; a descriptor lacking the
    method (or lacking native spin support altogether) must default to
    ``False`` via the defensive ``getattr(..., lambda: False)()`` call.
    """
    # DPA4 explicitly declares native spin support.
    dd = make_descriptor()
    assert dd.supports_native_spin() is True
    ft = SeZMEnergyFittingNet(
        ntypes=3,
        dim_descrpt=dd.get_dim_out(),
        neuron=[16],
        precision="float64",
        seed=5,
    )
    dpa4_model = DPAtomicModel(dd, ft, type_map=["A", "B", "C"])
    assert dpa4_model.supports_native_spin is True

    # A non-DPA4 descriptor has no supports_native_spin method at all; the
    # defensive getattr must fall back to False rather than raising.
    assert (
        getattr(dd, "not_a_real_method", lambda: False)() is False
    )  # sanity: getattr fallback semantics
    se_a = DescrptSeA(4.0, 3.5, [8, 8, 8])
    assert not hasattr(se_a, "supports_native_spin")
    assert getattr(se_a, "supports_native_spin", lambda: False)() is False
    ft_se_a = InvarFitting(
        "energy",
        3,
        se_a.get_dim_out(),
        1,
        mixed_types=se_a.mixed_types(),
    )
    se_a_model = DPAtomicModel(se_a, ft_se_a, type_map=["A", "B", "C"])
    assert se_a_model.supports_native_spin is False


def test_uses_graph_lower_feature_gates() -> None:
    # Conditioning inputs that still ride only the dense call signature must
    # gate the graph route off. Exercise each gate attribute directly (the
    # constructor kwargs for charge-spin/bridging are heavyweight).
    for attr in ("charge_spin_embedding", "bridging_switch"):
        dd = make_descriptor()
        assert dd.uses_graph_lower() is True
        setattr(dd, attr, object())  # any non-None sentinel
        assert dd.uses_graph_lower() is False, attr
    # native spin is now threaded through the graph lower (this task); a
    # non-None spin_embedding must NOT disable it any more.
    dd = make_descriptor()
    assert dd.uses_graph_lower() is True
    dd.spin_embedding = object()  # any non-None sentinel
    assert dd.uses_graph_lower() is True


def test_dpa4_ener_fitting_call_graph_matches_dense() -> None:
    # The inherited flat-N call_graph must be bit-identical to the dense
    # call for the custom GLU fitting nets.
    from deepmd.dpmodel.fitting.dpa4_ener import (
        SeZMEnergyFittingNet,
    )

    rng = np.random.default_rng(11)
    ntypes, nf, nloc, nd = 3, 2, 6, 16
    ft = SeZMEnergyFittingNet(
        ntypes=ntypes,
        dim_descrpt=nd,
        neuron=[24, 24],
        precision="float64",
        seed=5,
    )
    dd = rng.standard_normal((nf, nloc, nd))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    ref = np.asarray(ft(dd, atype)["energy"])
    got = np.asarray(
        ft.call_graph(dd.reshape(nf * nloc, nd), atype.reshape(-1))["energy"]
    )
    np.testing.assert_allclose(got.reshape(ref.shape), ref, rtol=1e-12, atol=1e-14)


# Golden values pinned at the pre-reroute dense ``call`` (Task 7 controller
# Step 1.5). Generated at commit 12fe36ce (before the dense body became a
# NeighborGraph adapter) on an edge-sensitive descriptor. These MUST stay green
# after the reroute: a failure is a real adapter bug (edge-enumeration order,
# masking semantics, or precision-cast placement) -- never regenerate them and
# never loosen the tolerance.
_GOLDEN_CALL_DENSE_A = np.array(
    [
        -0.08188959018201433,
        -1.6401098493602166,
        0.8980097439814706,
        0.6061720320675933,
        0.03693486789275601,
        -0.46056159178378686,
        0.26637207133459684,
        -1.0019824974075395,
        0.9408144542800043,
        -0.6099638888595346,
        2.4055685026408096,
        0.6359644277448068,
        0.6903588279790595,
        -0.9224964849664444,
        0.4990734718472053,
        -1.02895503646835,
        -0.05256005016309175,
        -1.1718280288126839,
        -0.43993272700594266,
        0.9966067345185069,
        1.5687440259194865,
        -3.030680888851244,
        1.3242653547772454,
        -0.9019250810694728,
        1.252395312256215,
        -0.5143223479381809,
        3.040128995265687,
        0.457988498855213,
        -1.3940405505600324,
        0.8850637922824572,
        1.037290726562649,
        -0.451579204316771,
        -1.6828699209445968,
        -0.017535015972101775,
        1.3975042439037813,
        0.83057838943815,
        0.017013404906080956,
        -0.047541411898216854,
        1.7317295348980857,
        0.26462100277272577,
        1.7083324216568532,
        -0.45813464799200393,
        2.597544279956864,
        -0.12463014959818852,
        -0.6443660165322296,
        -2.0713562544165725,
        2.5475184469030716,
        -3.261424700951668,
        0.7055742911945262,
        0.5939618372377968,
        -0.5648582648639231,
        1.0694437438105542,
        0.3792834930442088,
        -1.509222936665205,
        1.6931192101329084,
        -0.11732801383346884,
        -0.38258844981491313,
        -0.4160474995360459,
        0.6259875983844945,
        0.7919342487823745,
        -0.9272238100862809,
        -0.515267275914841,
        -0.23177697405554906,
        -0.6026480300696168,
        -0.7302717754794611,
        -2.0898379547118577,
        -0.24268458905721696,
        1.0001761022194025,
        1.6824564434754978,
        -2.633534959892684,
        0.8333379958024834,
        -1.109868239851361,
        1.969652575890085,
        -1.209619429600107,
        3.8541594041939087,
        0.6124194452723404,
        -0.9192570946559506,
        0.6938046845346053,
        1.8461036469106937,
        -1.013885986394464,
        -0.283395639887894,
        -1.217166027784505,
        1.1321978535280568,
        0.3540199411014809,
        -0.010795167604265472,
        -0.430970906188755,
        -0.17835625718126516,
        -0.5311504953096852,
        1.3411563831235733,
        -0.6627921066052314,
        1.9435706605008385,
        0.43193648491861686,
        1.2552004893220838,
        -1.3657297709949308,
        -0.013361990610621991,
        -0.33930694168302505,
        0.1088469407013412,
        -0.8166380356999482,
        -0.3706884827706439,
        0.5737376230333425,
        0.8648002143180473,
        -0.03232429036781902,
        0.5307182990446323,
        -0.8630078731724977,
        0.6021562456701433,
        -0.8453025424031696,
        1.8265839814217595,
        0.5628498717136691,
        0.2439306006638069,
        -0.0736639864300317,
        1.270061361051658,
        -0.6369663403612221,
        0.7182610271779691,
        -1.2498111641611587,
        0.11839551453786924,
        0.6732054967935062,
        0.8361026900297994,
        0.060824893598470195,
        -0.6875845049226071,
        -0.763183387458827,
        0.9760045122171771,
        -0.7764955748261451,
        1.76293297140091,
        -0.35504472767889006,
        0.21253553533723282,
        -1.0526008763609391,
        -0.1665626764854901,
        0.42925123110035823,
        1.064734842547012,
        -0.703125593916523,
        0.07276190724126394,
        0.3076641202491557,
        1.3224517640921287,
        -1.3962031133037114,
        0.37263550182784877,
        -0.3339235740697669,
        -0.1659563303062172,
        0.04099500531462166,
        1.898240916432775,
        1.0099610723769925,
        -1.4778201613232989,
        1.3871987774050112,
        -1.0213687442129757,
        0.686007686226708,
        -0.4399560418511054,
        0.11421268118318455,
        -0.07694608046532574,
        0.6033915797317563,
        0.2905546324808915,
        -0.351225105245204,
        0.42354548538349407,
        0.34984439358620534,
        0.6192799537531579,
        -0.12000594483636648,
        0.3763848085856086,
        -0.07361656061206275,
        0.1594170793656668,
        -1.1853656424153713,
        0.4255844333007695,
        -0.3040611162265903,
        -0.300791301906481,
        -0.5506951498905596,
        0.36012535786016303,
        0.5901188093406007,
        -0.11362248865107621,
        0.06574730595396804,
        -0.039340963126774875,
        -0.20957252676873592,
        0.5284990849697825,
        0.3279675804392425,
        0.6123515428161475,
        0.532812599642107,
        0.3274602744373568,
        -0.7679341935531645,
        -0.03278907210837387,
        0.20272032964433895,
        -0.4361103067776296,
        -1.6116068028821415,
        0.2234266625551761,
        0.7715215732147799,
        -0.1459405350043001,
        -1.2414697715592458,
        -0.09136738391038945,
        -1.2382383040771507,
        1.541171461220729,
        -0.6890339648617374,
        2.1760693147131955,
        0.6673206231848172,
        -0.18356563793714736,
        1.0249648836738674,
        -0.2139377206453627,
        0.3847415740416047,
    ]
).reshape((2, 6, 16))

_GOLDEN_CALL_DENSE_B = np.array(
    [
        -0.04846433809853028,
        0.12578811680620294,
        -0.502354031810507,
        0.1023746300424693,
        0.9474243712803286,
        0.6389272818246459,
        -0.37506873916066863,
        0.095383961266827,
        0.20579108233831206,
        -0.3276132151157352,
        -0.8137945688101983,
        0.05810457548131772,
        0.6878260670880192,
        0.1279947655103774,
        -0.13265038479198168,
        -0.24167288589339592,
        0.01112900444112332,
        -1.267334043258829,
        0.2142201939814206,
        0.23762263686923288,
        1.046386199811448,
        0.19242219347241613,
        -0.49840402577764353,
        -0.6985139062689419,
        0.6156564212081427,
        -0.2490472904444329,
        1.7735818652009143,
        0.9364817791001074,
        -0.10997362146753191,
        0.5498063052071968,
        0.2434448474115196,
        -0.1835268485901494,
        -0.5207070411295761,
        -0.0005716825111456957,
        0.43636630054153547,
        0.5333429882192071,
        -0.26341743971033177,
        -0.026220004855314442,
        0.3660861333059606,
        0.05434807756690765,
        0.9332986182814297,
        -0.38921117138952627,
        0.9892755054085703,
        -0.3914065747042522,
        0.6289576233357136,
        -1.2955038542957107,
        0.8427744955990363,
        -0.11360380301169497,
        -0.2606945308121848,
        -0.2262084317032449,
        0.18969602241467,
        0.2615142046127337,
        0.7136884578705122,
        0.5763061017240791,
        -0.773810816225985,
        0.49581953193264067,
        0.21465711080571293,
        0.24163022457103087,
        -0.601568735303618,
        0.3256737855053371,
        0.1631571118756852,
        -0.020323905852621227,
        -0.6476449636189504,
        -0.16119265021468507,
    ]
).reshape((1, 4, 16))


def test_call_dense_golden() -> None:
    """Pin the dense ``call`` outputs across the Task 7 graph-adapter reroute.

    Two fixtures exercise the two dense entry contracts on an edge-sensitive
    (jittered zero-init) descriptor so the pin is not vacuous:

    * Fixture A -- ``mapping=None`` gas-phase local indices (``nall == nloc``).
    * Fixture B -- a real periodic ~4-atom box with ghosts folded through an
      explicit extended->local ``mapping`` (``nall > nloc``), so the
      ghost-source scatter is actually exercised.

    The reroute (dense ``call`` -> ``graph_from_dense_quartet`` ->
    ``_call_graph_impl``) must preserve these values bit-for-bit within fp64
    scatter-reassociation tolerance.
    """
    # Fixture A: mapping=None local indices
    ddA = make_message_sensitive_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    outA = np.asarray(ddA.call(coord.reshape(nf, -1), atype, nlist)[0])
    np.testing.assert_allclose(outA, _GOLDEN_CALL_DENSE_A, rtol=1e-10, atol=1e-12)

    # Fixture B: real periodic ghosts + explicit mapping
    ddB = make_message_sensitive_descriptor()
    box = np.eye(3, dtype=np.float64)[None] * 6.0
    rng = np.random.default_rng(3)
    coord_b = rng.uniform(0.0, 6.0, size=(1, 4, 3))
    atype_b = np.array([[0, 1, 2, 0]], dtype=np.int64)
    ext_coord, ext_atype, mapping, nlist_b = extend_input_and_build_neighbor_list(
        coord_b,
        atype_b,
        ddB.get_rcut(),
        ddB.get_sel(),
        mixed_types=ddB.mixed_types(),
        box=box,
    )
    assert ext_atype.shape[1] > coord_b.shape[1]  # ghosts present
    outB = np.asarray(ddB.call(ext_coord, ext_atype, nlist_b, mapping=mapping)[0])
    np.testing.assert_allclose(outB, _GOLDEN_CALL_DENSE_B, rtol=1e-10, atol=1e-12)


def test_dense_call_comm_dict_raises() -> None:
    # The dense lower has no comm implementation; the dense adapter is the
    # one owner of that rejection.
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    with pytest.raises(NotImplementedError, match="dense"):
        dd.call(coord.reshape(nf, -1), atype, nlist, comm_dict={"dummy": None})


def test_call_graph_comm_dict_reaches_leaf_stub() -> None:
    # The graph trunk now threads comm_dict; in the pure-dpmodel backend the
    # per-block exchange leaf is the guard (mirrors dpa2's
    # _exchange_ghosts_graph base). ``make_descriptor()`` leaves
    # ``use_env_seed`` at its class default (True), so ``_block_comm``
    # forwards comm_dict starting at block 0 already (block 0 is only
    # skipped when ``use_env_seed=False``) -- the leaf raise fires on the
    # first block regardless of ``n_blocks``. ``n_blocks=2`` is not load
    # bearing for reaching the leaf here; it just matches the shared
    # ``make_descriptor()`` fixture used across this file.
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    graph = make_graph_from_nlist(coord, nlist)
    fake_comm = dict.fromkeys(
        (
            "send_list",
            "send_proc",
            "recv_proc",
            "send_num",
            "recv_num",
            "communicator",
            "nlocal",
            "nghost",
        )
    )
    with pytest.raises(NotImplementedError, match="dpmodel backend"):
        dd.call_graph(graph, atype.reshape(-1), comm_dict=fake_comm)
