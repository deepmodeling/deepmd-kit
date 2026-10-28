# SPDX-License-Identifier: LGPL-3.0-or-later
"""Carry-all graph energy forward via ``neighbor_graph_method`` (Option B).

``CM.call_common`` routes a graph-eligible dpa1(``attn_layer == 0``) ENERGY
forward through the carry-all graph builder + ``call_lower_graph``. Per the
default-flip (decision #17) this is now the DEFAULT for eligible models;
``neighbor_graph_method="legacy"`` opts out to the truncating dense nlist path,
and ``"dense"``/``"ase"`` force the carry-all graph with that builder.

Option-B behavior (decision #17 / spec_unified_edge_nlist):

* non-binding ``sel`` -- the carry-all graph and the legacy dense path see the
  SAME neighbors, so ``energy``/``atom_energy`` are EXACTLY equal;
* binding ``sel`` -- the carry-all graph keeps neighbors the legacy dense path
  truncates, so energy DIFFERS (intended).
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)


def _make_model(sel):
    ds = DescrptDPA1(
        rcut=4.0,
        rcut_smth=0.5,
        sel=sel,
        ntypes=2,
        attn_layer=0,
        axis_neuron=2,
        neuron=[6, 12],
    )
    ft = InvarFitting(
        "energy",
        2,
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
    )
    return EnergyModel(ds, ft, type_map=["foo", "bar"])


@pytest.mark.parametrize("method", ["dense", "ase"])  # in-tree carry-all AND ase
@pytest.mark.parametrize("periodic", [True, False])  # PBC and non-PBC
def test_energy_parity_non_binding_sel(method, periodic) -> None:
    """At non-binding sel the carry-all graph and the dense path see the SAME
    neighbors, so model energy is exactly equal.
    """
    if method == "ase":
        pytest.importorskip("ase")
    rng = np.random.default_rng(0)
    nloc = 6
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
    box = None
    if periodic:
        # large box so the cell is essentially non-periodic for rcut=4.0
        box = np.eye(3).reshape(1, 9) * 20.0
    # LARGE sel -> non-binding (no truncation)
    model = _make_model([200])

    dense = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    graph = model.call_common(coord, atype, box, neighbor_graph_method=method)

    # dense energy keys: ``energy_redu`` (reduced, nf x 1) and ``energy``
    # (per-atom, nf x nloc x 1). Compare matching keys.
    np.testing.assert_allclose(
        graph["energy_redu"], dense["energy_redu"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(graph["energy"], dense["energy"], rtol=1e-12, atol=1e-12)
    # mask must match the dense all-ones (nf, nloc) int mask
    np.testing.assert_array_equal(graph["mask"], dense["mask"])


@pytest.mark.parametrize("method", ["dense", "ase"])  # in-tree carry-all AND ase
def test_energy_parity_multiframe_periodic(method) -> None:
    """Multi-frame (nf=2) PERIODIC energy parity at non-binding sel.

    Exercises the nf>1 graph reductions (``frame_id = repeat(arange(nf),
    n_node)`` energy segment-sum and the ``frame * nloc`` node offsetting in
    ``from_dense_quartet``) with DIFFERENT per-frame coordinates and a box.
    At non-binding sel the carry-all graph and the dense path see the SAME
    neighbors, so ``energy_redu``/``energy`` are EXACTLY equal per frame.
    """
    if method == "ase":
        pytest.importorskip("ase")
    rng = np.random.default_rng(3)
    nf, nloc = 2, 6
    # distinct coordinates per frame (not a broadcast of one frame)
    coord = rng.normal(size=(nf, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0, 1]] * nf, dtype=np.int64)
    # large box so the cell is essentially non-periodic for rcut=4.0
    box = np.tile(np.eye(3).reshape(1, 9) * 20.0, (nf, 1))
    # LARGE sel -> non-binding (no truncation)
    model = _make_model([200])

    dense = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    graph = model.call_common(coord, atype, box, neighbor_graph_method=method)

    np.testing.assert_allclose(
        graph["energy_redu"], dense["energy_redu"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(graph["energy"], dense["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(graph["mask"], dense["mask"])
    # the two frames must produce DIFFERENT energies (genuine nf>1 test, not a
    # broadcast of one frame); they differ here by ~1e-5.
    assert not np.array_equal(dense["energy_redu"][0], dense["energy_redu"][1])


def test_virtual_atom_masked() -> None:
    """A virtual atom (``atype == -1``) must contribute ZERO energy and have a
    ZERO mask in the carry-all graph path, matching the dense path exactly.

    Regression for the leak where the graph path fed the raw (negative) atype
    to the descriptor/fitting and stamped an all-ones mask, so virtual atoms
    picked up a type-embedding + bias energy that the dense path masks out.

    Uses the in-tree ``"dense"`` builder, which shares the EXACT same quartet
    neighbor list as the ``"legacy"`` dense path, so the parity is bit-tight
    (the ``"ase"`` builder has its own near-cutoff boundary quirks, covered by
    the other tests).
    """
    method = "dense"
    rng = np.random.default_rng(7)
    nloc = 6
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    # one local virtual atom (atype == -1); the rest are real
    atype = np.array([[0, 1, -1, 1, 0, 1]], dtype=np.int64)
    box = None
    # LARGE sel -> non-binding (no truncation) so dense == graph on real atoms
    model = _make_model([200])

    dense = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    graph = model.call_common(coord, atype, box, neighbor_graph_method=method)

    # graph energy (reduced + per-atom) must match the dense path exactly
    np.testing.assert_allclose(
        graph["energy_redu"], dense["energy_redu"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(graph["energy"], dense["energy"], rtol=1e-12, atol=1e-12)
    # the virtual atom (index 2) contributes ZERO per-atom energy
    np.testing.assert_allclose(graph["energy"][0, 2], 0.0, rtol=0, atol=0)
    # mask must be 0 at the virtual atom and match the dense int mask
    assert int(graph["mask"][0, 2]) == 0
    np.testing.assert_array_equal(graph["mask"], dense["mask"])
    expected_mask = np.array([[1, 1, 0, 1, 1, 1]], dtype=np.int32)
    np.testing.assert_array_equal(graph["mask"], expected_mask)


def test_binding_sel_carries_more_than_dense() -> None:
    """At binding sel the carry-all graph includes neighbors the dense path
    truncates, so energy DIFFERS (intended, decision #17 / Option B).
    """
    rng = np.random.default_rng(1)
    nloc = 14
    # a dense cluster: many atoms well within rcut=4.0 of each other
    coord = rng.normal(size=(1, nloc, 3)) * 0.8
    atype = np.array([[0, 1] * 7], dtype=np.int64)
    box = None
    # binding sel -> dense path truncates to 4 neighbors per atom
    model = _make_model([4])

    dense = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    graph = model.call_common(coord, atype, box, neighbor_graph_method="dense")

    assert not np.allclose(graph["energy_redu"], dense["energy_redu"])


def test_neighbor_list_conflicts_with_graph_method() -> None:
    """An explicit ``neighbor_list`` (a dense-nlist strategy) cannot be combined
    with an explicit graph ``neighbor_graph_method``; passing both raises.
    """
    from deepmd.dpmodel.utils.default_neighbor_list import (
        DefaultNeighborList,
    )

    rng = np.random.default_rng(2)
    nloc = 6
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
    model = _make_model([200])

    with pytest.raises(ValueError, match="cannot be combined"):
        model.call_common(
            coord,
            atype,
            None,
            neighbor_list=DefaultNeighborList(),
            neighbor_graph_method="dense",
        )


def test_neighbor_list_takes_dense_route() -> None:
    """Supplying ``neighbor_list`` (without an explicit graph method) takes the
    dense route -- it is NOT silently ignored by the graph path. With the
    default builder the result matches the legacy dense path exactly.
    """
    from deepmd.dpmodel.utils.default_neighbor_list import (
        DefaultNeighborList,
    )

    rng = np.random.default_rng(3)
    nloc = 6
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
    box = np.eye(3).reshape(1, 9) * 20.0
    model = _make_model([200])

    legacy = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    with_nlist = model.call_common(
        coord, atype, box, neighbor_list=DefaultNeighborList()
    )
    np.testing.assert_allclose(
        with_nlist["energy_redu"], legacy["energy_redu"], rtol=1e-12, atol=1e-12
    )


def test_graph_lower_invariant_to_charge_spin() -> None:
    """dpa1 does NOT consume charge_spin (``get_dim_chg_spin() == 0``); the dense
    atomic model passes ``None`` to the dpa1 descriptor regardless. The graph
    lower accepts ``charge_spin`` for descriptors with charge/spin
    conditioning, so dpa1 output must be invariant to it.

    Combined with the graph==dense parity at non-binding sel
    (:func:`test_energy_parity_non_binding_sel`), this gives the full claim:
    ``graph(charge_spin) == graph(None) == dense``.
    """
    from deepmd.dpmodel.utils.neighbor_graph import (
        build_neighbor_graph,
    )

    rng = np.random.default_rng(4)
    nloc = 6
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
    box = np.eye(3).reshape(1, 9) * 20.0
    model = _make_model([200])
    assert model.get_descriptor().get_dim_chg_spin() == 0  # dpa1: no chg/spin

    ng = build_neighbor_graph(coord, atype, box, model.get_rcut())
    atype_flat = atype.reshape(-1)
    base = model.call_common_lower_graph(
        atype_flat, ng.n_node, ng.edge_index, ng.edge_vec, ng.edge_mask
    )
    # arbitrary non-None charge/spin -> must NOT change the dpa1 graph output
    cs = np.array([[1.0, 2.0]], dtype=coord.dtype)
    with_cs = model.call_common_lower_graph(
        atype_flat,
        ng.n_node,
        ng.edge_index,
        ng.edge_vec,
        ng.edge_mask,
        charge_spin=cs,
    )
    assert set(base) == set(with_cs)
    for k, v in base.items():
        if v is None:
            assert with_cs[k] is None
        else:
            np.testing.assert_array_equal(with_cs[k], v)


def test_graph_type_embedding_table_matches_type_embedding() -> None:
    # The seam hook must return exactly the descriptor's full tebd table.
    dd = _make_model([200]).get_descriptor()
    np.testing.assert_array_equal(
        np.asarray(dd.graph_type_embedding_table()),
        np.asarray(dd.type_embedding.call()),
    )
