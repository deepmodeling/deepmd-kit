# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel
from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1
from deepmd.dpmodel.fitting import InvarFitting
from deepmd.dpmodel.model.ener_model import EnergyModel
from deepmd.dpmodel.utils.neighbor_graph import from_dense_quartet
from deepmd.dpmodel.utils.nlist import extend_input_and_build_neighbor_list


def _atomic_model(sel=(30,), **kw):
    ds = DescrptDPA1(
        rcut=4.0, rcut_smth=0.5, sel=list(sel), ntypes=2, attn_layer=0, **kw
    )
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    return DPAtomicModel(ds, ft, type_map=["a", "b"])


def test_forward_atomic_graph_matches_dense():
    rng = np.random.default_rng(0)
    coord = rng.normal(size=(1, 5, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)
    am = _atomic_model()
    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, 4.0, [30], mixed_types=True, box=None
    )
    dense = am.forward_atomic(ext_coord, ext_atype, nlist, mapping=mapping)
    ng = from_dense_quartet(ext_coord, nlist, mapping)
    graph = am.forward_atomic_graph(ng, atype.reshape(-1))
    np.testing.assert_allclose(graph["energy"], dense["energy"], rtol=1e-12, atol=1e-12)


def test_forward_common_atomic_graph_matches_dense():
    rng = np.random.default_rng(1)
    coord = rng.normal(size=(1, 5, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)
    am = _atomic_model()
    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, 4.0, [30], mixed_types=True, box=None
    )
    dense = am.forward_common_atomic(ext_coord, ext_atype, nlist, mapping=mapping)
    ng = from_dense_quartet(ext_coord, nlist, mapping)
    graph = am.forward_common_atomic_graph(ng, atype.reshape(-1))
    for k in ("energy", "mask"):
        np.testing.assert_allclose(
            np.asarray(graph[k]), np.asarray(dense[k]), rtol=1e-12, atol=1e-12
        )


# ── Feature-flag parity matrix (Task 6) ──────────────────────────────────────


def _ener_model(sel, type_one_side=False, exclude_types=None):
    ds = DescrptDPA1(
        rcut=4.0,
        rcut_smth=0.5,
        sel=list(sel),
        ntypes=2,
        attn_layer=0,
        type_one_side=type_one_side,
        exclude_types=exclude_types or [],
    )
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    return EnergyModel(ds, ft, type_map=["a", "b"])


@pytest.mark.parametrize("virtual", [False, True])  # one local atype == -1
@pytest.mark.parametrize("type_one_side", [False, True])  # tebd concat content
@pytest.mark.parametrize("nf", [1, 2])  # single- and multi-frame
def test_graph_matches_dense_over_flags(virtual, type_one_side, nf):
    rng = np.random.default_rng(2)
    nloc = 6
    coord = rng.normal(size=(nf, nloc, 3)) * 1.5
    atype = np.tile(np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64), (nf, 1))
    if virtual:
        atype[:, -1] = -1  # mark one local atom virtual
    box = np.tile(np.eye(3).reshape(1, 9) * 20.0, (nf, 1))
    model = _ener_model([200], type_one_side=type_one_side)  # non-binding sel
    g = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    d = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    for k in ("energy", "energy_redu", "mask"):
        np.testing.assert_allclose(
            np.asarray(g[k]), np.asarray(d[k]), rtol=1e-12, atol=1e-12
        )
    if virtual:
        assert int(np.asarray(g["mask"])[0, -1]) == 0  # virtual atom masked


def test_pair_exclude_types_falls_back_to_dense():
    """Pair exclude_types is unsupported on the graph -> uses_graph_lower False."""
    m = _ener_model([30], exclude_types=[(0, 1)])
    assert m.atomic_model.descriptor.uses_graph_lower() is False


def test_graph_matches_dense_with_out_bias():
    """The graph path applies apply_out_stat (per-type out-bias) identically
    to the dense path. With a non-zero bias, graph == dense at 1e-12, and the
    bias actually shifts the graph energy (non-vacuous).
    """
    rng = np.random.default_rng(3)
    nloc = 5
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)
    box = np.eye(3).reshape(1, 9) * 20.0
    model = _ener_model([200])
    # energy BEFORE setting bias (zero out-bias), graph path
    g_zero = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    # set a non-zero per-type energy out-bias
    model.atomic_model.out_bias[0, :, 0] = np.array([0.3, -0.7])
    g = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    d = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    # graph applies out-stat exactly like dense
    for k in ("energy", "energy_redu"):
        np.testing.assert_allclose(
            np.asarray(g[k]), np.asarray(d[k]), rtol=1e-12, atol=1e-12
        )
    # non-vacuous: the bias actually shifted the graph energy
    assert not np.allclose(np.asarray(g["energy"]), np.asarray(g_zero["energy"]))
