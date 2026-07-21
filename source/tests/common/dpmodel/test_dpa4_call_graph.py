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
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
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
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
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


def make_charge_spin_descriptor(seed: int = 99) -> DescrptDPA4:
    """An ``add_chg_spin_ebd`` variant of ``make_message_sensitive_descriptor()``.

    Charge/spin FiLM conditions the type embedding directly (see
    ``_apply_charge_spin_embedding``), which -- per
    ``jitter_zero_arrays``'s docstring -- IS the fresh descriptor's scalar
    read-out (zero-init residuals make the blocks near-identity), so even a
    bare ``make_descriptor(add_chg_spin_ebd=True)`` would already be
    charge_spin-sensitive. Jittered anyway for consistency with every other
    graph-route fixture in this module (project convention: all
    sensitivity/parity fixtures jitter).
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
        add_chg_spin_ebd=True,
    )
    data = dd.serialize()
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
    return DescrptDPA4.deserialize(data)


def test_call_graph_charge_spin_sensitivity() -> None:
    """call_graph(charge_spin=...) must change the output (teeth: charge_spin reaches the trunk)."""
    dd = make_charge_spin_descriptor()
    coord, atype, nlist = make_inputs()
    graph = make_graph_from_nlist(coord, nlist)
    atype_flat = atype.reshape(-1)
    nf = atype.shape[0]
    cs0 = np.tile(np.array([[0.0, 1.0]]), (nf, 1))
    cs1 = np.tile(np.array([[1.0, 1.0]]), (nf, 1))
    out0, _ = dd.call_graph(graph, atype_flat, charge_spin=cs0)
    out1, _ = dd.call_graph(graph, atype_flat, charge_spin=cs1)
    assert not np.allclose(np.asarray(out0), np.asarray(out1))


def test_call_graph_charge_spin_matches_dense_adapter() -> None:
    """Graph-lower charge_spin path == dense adapter charge_spin path (shared trunk, 1e-12).

    Uses DISTINCT charge_spin values per frame (nf=2) so the test also pins
    the per-frame (nf, 2) -> flat-N association: a wrong nf threaded into
    ``_apply_charge_spin_embedding`` would either shape-error or silently mix
    the two frames' conditioning.
    """
    dd = make_charge_spin_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    cs = np.array([[0.3, -0.7], [1.0, 0.2]])
    ref, *_ = dd.call(coord.reshape(nf, -1), atype, nlist, charge_spin=cs)
    graph, atype_flat = _graph_from_padded_nlist(coord, atype, nlist, None)
    out, _ = dd.call_graph(graph, atype_flat, charge_spin=cs)
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
    # Every conditioning input DPA4 supports now rides the graph lower: native
    # spin, charge/spin FiLM, and SFPG bridging. Only the explicit escape
    # hatch (disable_graph_lower / _graph_lower_disabled) gates it off.
    for attr in ("charge_spin_embedding", "bridging_switch", "spin_embedding"):
        dd = make_descriptor()
        assert dd.uses_graph_lower() is True
        setattr(dd, attr, object())  # any non-None sentinel
        assert dd.uses_graph_lower() is True, attr
    dd = make_descriptor()
    assert dd.uses_graph_lower() is True
    dd.disable_graph_lower()
    assert dd.uses_graph_lower() is False


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


def make_bridging_descriptor(seed: int = 99) -> DescrptDPA4:
    """A SFPG-bridging variant of ``make_message_sensitive_descriptor()``.

    ``inner_clamp_r_inner``/``inner_clamp_r_outer`` build ``InnerClamp`` (edge
    distance freeze) and ``BridgingSwitch`` (per-source edge gate) -- see
    ``test_message_passing_semantics`` in ``test_descrpt_dpa4.py`` for the
    same construction. Jittered for the same message-sensitivity reason as
    ``make_message_sensitive_descriptor``.
    """
    dd = DescrptDPA4(
        ntypes=2,
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
        inner_clamp_r_inner=0.8,
        inner_clamp_r_outer=1.2,
    )
    data = dd.serialize()
    data = jitter_zero_arrays(data, np.random.default_rng(seed))
    return DescrptDPA4.deserialize(data)


def _sphere_points(n_points: int, radius: float) -> np.ndarray:
    """Deterministic Fibonacci-like sphere sampling around the origin."""
    idx = np.arange(n_points, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden-angle step
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_points)
    rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = phi * idx
    return np.stack(
        [radius * rho * np.cos(theta), radius * rho * np.sin(theta), radius * z],
        axis=-1,
    )  # (n_points, 3)


def _frozen_sphere_descriptor_outputs(
    dd: DescrptDPA4, near_distance: float, n_points: int = 12
) -> np.ndarray:
    """3-atom probe (A, B, C): A fixed at the origin, B rigidly slides on a
    sphere of radius ``near_distance`` around A (inside the frozen zone), C
    anchored well outside the bridging window as an ordinary GNN neighbor of
    both. Returns the flat call_graph output reshaped to (n_points, 3, C).

    Mirrors pt's ``TestSourceFreezePropagationGate._build_three_atom_box`` /
    ``_evaluate_frozen_sphere_atom_energies``, at the descriptor level (no
    fitting net needed: if the descriptor feature of A is invariant, any
    downstream per-atom fitting net -- itself a function of A's fixed
    descriptor and fixed type only -- is trivially invariant too).
    """
    directions = _sphere_points(n_points, near_distance)
    coord = np.zeros((n_points, 3, 3), dtype=np.float64)
    coord[:, 1, :] = directions  # B rotates around A, radius fixed
    coord[:, 2, :] = np.array([2.4, 0.0, 0.0])  # C: ordinary neighbor, static
    atype = np.tile(np.array([[0, 1, 0]], dtype=np.int64), (n_points, 1))
    nlist = build_neighbor_list_np(coord, dd.get_rcut(), nnei=4)
    graph = make_graph_from_nlist(coord, nlist)
    out, _ = dd.call_graph(graph, atype.reshape(-1))
    return np.asarray(out).reshape(n_points, 3, -1)


def test_call_graph_bridging_frozen_sphere_invariance() -> None:
    """SFPG bridging on the graph route: A's descriptor must be invariant to
    the rigid motion of its frozen partner B (inside r_inner=0.8).
    """
    dd = make_bridging_descriptor()
    out = _frozen_sphere_descriptor_outputs(dd, near_distance=0.5)
    span_a = np.max(np.abs(out[:, 0, :] - out[0:1, 0, :]))
    assert span_a < 1e-10, span_a


def test_call_graph_bridging_leak_reopens_when_gate_disabled() -> None:
    """Ablation: clearing bridging_switch (InnerClamp stays active) must
    reopen the direction/multi-hop leak on the graph route -- pins that SFPG,
    not InnerClamp alone, owns the invariance above.
    """
    dd = make_bridging_descriptor()
    dd.bridging_switch = None
    out = _frozen_sphere_descriptor_outputs(dd, near_distance=0.5)
    span_a = np.max(np.abs(out[:, 0, :] - out[0:1, 0, :]))
    assert span_a > 1e-6, span_a


def test_charge_spin_model_routes_through_graph_lower() -> None:
    """A native charge_spin DPA4 model must reach ``call_graph`` (not the old
    ``cs -> dense`` gate) when a graph ``neighbor_graph_method`` is requested,
    and the output must still be charge_spin-sensitive.
    """
    from unittest.mock import (
        patch,
    )

    from deepmd.dpmodel.model.model import (
        get_model,
    )

    config = {
        "type_map": ["Ni", "O"],
        "descriptor": {
            "type": "dpa4",
            "rcut": 4.0,
            "sel": 8,
            "channels": 16,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "precision": "float64",
            "seed": 7,
            "random_gamma": False,
            "add_chg_spin_ebd": True,
        },
        "fitting_net": {"type": "dpa4_ener", "neuron": [8, 8]},
    }
    model = get_model(config)
    data = model.serialize()
    data = jitter_zero_arrays(data, np.random.default_rng(17))
    model = type(model).deserialize(data)

    rng = np.random.default_rng(23)
    nf, nloc = 1, 6
    coord = rng.uniform(0.5, 3.5, size=(nf, nloc, 3))
    atype = rng.integers(0, 2, size=(nf, nloc))
    box = 8.0 * np.eye(3, dtype=np.float64)[None]
    cs0 = np.array([[0.0, 1.0]])
    cs1 = np.array([[1.0, 1.0]])

    # Spy on the INSTANCE's bound method (wrapping the bound method), not the
    # class with autospec: ``autospec=True`` + ``wraps=<unbound>`` mis-binds
    # ``self`` across Python/mock versions (green on 3.13, "not enough values
    # to unpack" on 3.10 CI). Patching the instance and wrapping its bound
    # method is version-stable.
    descriptor = model.atomic_model.descriptor
    with patch.object(
        descriptor,
        "call_graph",
        wraps=descriptor.call_graph,
    ) as spy:
        out0 = model.call_common(
            coord, atype, box, charge_spin=cs0, neighbor_graph_method="dense"
        )
        assert spy.call_count >= 1, (
            "charge_spin must not force the model back onto the dense lower"
        )
    out1 = model.call_common(
        coord, atype, box, charge_spin=cs1, neighbor_graph_method="dense"
    )
    assert not np.allclose(out0["energy_redu"], out1["energy_redu"])


# Golden values pinned at the pre-reroute dense ``call`` (Task 7 controller
# Step 1.5). Generated at commit 12fe36ce (before the dense body became a
# NeighborGraph adapter) on an edge-sensitive descriptor. These MUST stay green
# after the reroute: a failure is a real adapter bug (edge-enumeration order,
# masking semantics, or precision-cast placement) -- never regenerate them and
# never loosen the tolerance.
_GOLDEN_CALL_DENSE_A = np.array(
    [
        -0.08188957220398312,
        -1.640109878466157,
        0.8980097395843406,
        0.6061720338823796,
        0.0369348802764473,
        -0.4605616073317207,
        0.2663720748033271,
        -1.0019825204289972,
        0.9408144687458523,
        -0.6099639089284248,
        2.4055685430467664,
        0.6359644283661099,
        0.6903588394569702,
        -0.9224964930776121,
        0.4990734769035464,
        -1.0289550441210629,
        -0.052560049562915594,
        -1.17182802800252,
        -0.4399327261943537,
        0.9966067341202732,
        1.5687440250412155,
        -3.0306808891197763,
        1.3242653552348236,
        -0.9019250803105836,
        1.252395311571487,
        -0.514322347326694,
        3.040128995033731,
        0.45798849935105596,
        -1.3940405513593772,
        0.8850637923045088,
        1.0372907246516128,
        -0.4515792038039226,
        -1.6828699160972829,
        -0.01753501653910832,
        1.3975042415458003,
        0.8305783891840123,
        0.017013409258949445,
        -0.047541417689215784,
        1.7317295338479273,
        0.2646210000936154,
        1.7083324210702961,
        -0.458134646915977,
        2.59754428069986,
        -0.12463014845722649,
        -0.6443660208642956,
        -2.071356246344078,
        2.5475184391276517,
        -3.261424696743016,
        0.7055742966569417,
        0.5939618410270757,
        -0.56485826918787,
        1.0694437511308956,
        0.37928349246064613,
        -1.5092229530421246,
        1.693119227855766,
        -0.11732801749609194,
        -0.3825884526212911,
        -0.4160475055486212,
        0.6259876087807682,
        0.7919342525125217,
        -0.927223820144914,
        -0.5152672801477078,
        -0.23177697201360192,
        -0.6026480324401201,
        -0.7302717757588707,
        -2.0898379534230243,
        -0.24268458779685936,
        1.0001761022724518,
        1.6824564427885655,
        -2.6335349596487907,
        0.8333379970412093,
        -1.10986823930664,
        1.96965257621882,
        -1.2096194291700268,
        3.8541594037592026,
        0.6124194446383112,
        -0.9192570952710638,
        0.6938046842943415,
        1.846103645926,
        -1.0138859880807036,
        -0.28339562628465337,
        -1.217166056202735,
        1.1321978569539521,
        0.3540199393504897,
        -0.010795152497377874,
        -0.43097091905968254,
        -0.17835626273011676,
        -0.5311505123026412,
        1.3411564031588632,
        -0.6627921275277259,
        1.943570693424962,
        0.4319364830723235,
        1.255200508583198,
        -1.3657297839233669,
        -0.01336199019878765,
        -0.3393069462873377,
        0.10884694751201296,
        -0.8166380331391334,
        -0.3706884901668285,
        0.573737624799559,
        0.8648002171342688,
        -0.03232429463294054,
        0.5307183066940667,
        -0.8630078781729218,
        0.6021562424905322,
        -0.8453025461956412,
        1.82658398151856,
        0.5628498724685331,
        0.2439306013273806,
        -0.07366398394503038,
        1.2700613621652759,
        -0.6369663434120979,
        0.7182610420412423,
        -1.2498111948276467,
        0.11839550585559121,
        0.6732055094305546,
        0.8361026952993864,
        0.060824890674981126,
        -0.6875845134943815,
        -0.7631834108102469,
        0.9760045290068342,
        -0.7764955885467189,
        1.7629330045268279,
        -0.35504472682736055,
        0.21253553542145115,
        -1.0526008881405187,
        -0.16656267846511705,
        0.4292512537198382,
        1.0647348591723098,
        -0.7031255947553559,
        0.0727618965092522,
        0.30766412122713804,
        1.3224517746204512,
        -1.3962031323934576,
        0.3726355110942625,
        -0.3339235817057772,
        -0.16595633888865696,
        0.040995004058702995,
        1.898240920028487,
        1.0099610840499202,
        -1.4778201749113464,
        1.3871987975207034,
        -1.02136875967816,
        0.6860076952082405,
        -0.43995605826138,
        0.114212679611265,
        -0.07694607628767526,
        0.6033915854223885,
        0.2905546252040811,
        -0.3512251163848229,
        0.423545494803197,
        0.34984439544936924,
        0.6192799614319217,
        -0.12000594508590802,
        0.37638481745100344,
        -0.07361656093975473,
        0.1594170731485163,
        -1.1853656577148597,
        0.425584449110405,
        -0.3040611152369916,
        -0.3007913065612787,
        -0.5506951605940389,
        0.36012535729977696,
        0.5901188186270819,
        -0.11362249148148017,
        0.06574730424045437,
        -0.03934096367717334,
        -0.20957253183767113,
        0.5284990952855384,
        0.3279675860082166,
        0.6123515557513174,
        0.5328126070541334,
        0.32746026978870224,
        -0.7679342108202849,
        -0.032789070637326424,
        0.2027203335513228,
        -0.4361103109207129,
        -1.611606821198701,
        0.22342666473445216,
        0.7715215838425302,
        -0.14594054046534827,
        -1.241469779704863,
        -0.09136738204622909,
        -1.2382383210774293,
        1.5411714827503882,
        -0.6890339762040898,
        2.1760693406110523,
        0.6673206230235658,
        -0.1835656432982903,
        1.0249648846349062,
        -0.21393771509994333,
        0.3847415726849628,
    ]
).reshape((2, 6, 16))

_GOLDEN_CALL_DENSE_B = np.array(
    [
        -0.048464349831002264,
        0.12578814100862684,
        -0.5023540889336157,
        0.10237461495825154,
        0.9474244082687194,
        0.6389273077690177,
        -0.3750687372458193,
        0.09538392495089612,
        0.2057911146693594,
        -0.32761325570956845,
        -0.8137945977073056,
        0.058104558306177854,
        0.6878261146120073,
        0.12799479249653034,
        -0.13265034854586666,
        -0.24167290953944234,
        0.011129002879949498,
        -1.2673340880838346,
        0.21422018979845414,
        0.2376226530727919,
        1.0463862207314913,
        0.19242216959898456,
        -0.49840404243215153,
        -0.6985139237420458,
        0.6156564351614421,
        -0.24904728503654996,
        1.7735818867149378,
        0.9364818091277682,
        -0.10997363505719217,
        0.5498063302746796,
        0.24344482865921666,
        -0.18352683867020564,
        -0.5207070541287808,
        -0.0005716822472628322,
        0.43636630351918926,
        0.5333429921068772,
        -0.263417441450637,
        -0.026220016900765974,
        0.3660861465815261,
        0.0543480783146649,
        0.9332986431668936,
        -0.3892111895434158,
        0.9892755313166647,
        -0.3914065926656158,
        0.628957621557961,
        -1.2955038838182076,
        0.8427745190841428,
        -0.11360380972926276,
        -0.2606945801433464,
        -0.2262084619051,
        0.18969605423739705,
        0.2615142011677701,
        0.7136884709209146,
        0.5763061230504399,
        -0.7738108644417824,
        0.49581954056878286,
        0.21465715021208298,
        0.2416302457608278,
        -0.6015687398551549,
        0.32567380127953915,
        0.16315710937654773,
        -0.02032387835522999,
        -0.6476449833730196,
        -0.1611926657032835,
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
    ``_run_graph``) must preserve these values bit-for-bit within fp64
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
