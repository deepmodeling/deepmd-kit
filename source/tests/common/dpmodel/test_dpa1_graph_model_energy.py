# SPDX-License-Identifier: LGPL-3.0-or-later
"""Opt-in carry-all graph energy forward via ``neighbor_graph_method`` (Option B).

PR-A 5c: ``CM.call_common`` gains a ``neighbor_graph_method`` keyword that,
when set, routes a dpa1(``attn_layer == 0``) ENERGY forward through the
carry-all graph builder + ``call_lower_graph`` instead of the dense nlist path.

Option-B behavior (decision #17 / spec_unified_edge_nlist):

* non-binding ``sel`` -- the carry-all graph and the dense path see the SAME
  neighbors, so ``energy``/``atom_energy`` are EXACTLY equal;
* binding ``sel`` -- the carry-all graph keeps neighbors the dense path
  truncates, so energy DIFFERS (intended).

The DEFAULT (``neighbor_graph_method=None``) keeps the dense path unchanged.
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

    dense = model.call_common(coord, atype, box)
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

    dense = model.call_common(coord, atype, box)
    graph = model.call_common(coord, atype, box, neighbor_graph_method=method)

    np.testing.assert_allclose(
        graph["energy_redu"], dense["energy_redu"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(graph["energy"], dense["energy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(graph["mask"], dense["mask"])
    # the two frames must produce DIFFERENT energies (genuine nf>1 test, not a
    # broadcast of one frame); they differ here by ~1e-5.
    assert not np.array_equal(dense["energy_redu"][0], dense["energy_redu"][1])


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

    dense = model.call_common(coord, atype, box)
    graph = model.call_common(coord, atype, box, neighbor_graph_method="dense")

    assert not np.allclose(graph["energy_redu"], dense["energy_redu"])
