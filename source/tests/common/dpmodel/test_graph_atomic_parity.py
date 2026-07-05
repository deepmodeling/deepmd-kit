# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    apply_pair_exclusion,
    from_dense_quartet,
)
from deepmd.dpmodel.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


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
    np.testing.assert_allclose(
        graph["energy"], dense["energy"].reshape(-1, 1), rtol=1e-12, atol=1e-12
    )


def test_forward_atomic_graph_flat_shape_and_parity():
    """Flat (N, *) output, matching dense forward_atomic raveled over (nf, nloc)."""
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
    assert graph["energy"].shape == (5, 1)  # FLAT (N, 1)
    np.testing.assert_allclose(
        graph["energy"], dense["energy"].reshape(5, 1), rtol=1e-12, atol=1e-12
    )


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
    # graph returns flat (N,*); reshape dense (nf,nloc,*) -> flat for comparison
    for k in ("energy", "mask"):
        g_arr = np.asarray(graph[k])
        d_arr = np.asarray(dense[k]).reshape(g_arr.shape)
        np.testing.assert_allclose(g_arr, d_arr, rtol=1e-12, atol=1e-12)


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


def test_descriptor_exclude_types_is_graph_eligible():
    """Descriptor-level exclude_types (Task 3): uses_graph_lower() is True."""
    m = _ener_model([30], exclude_types=[(0, 1)])
    assert m.atomic_model.descriptor.uses_graph_lower() is True


def test_model_pair_exclude_types_graph_matches_dense():
    """Model-level pair_exclude_types is now graph-native (edge mask): graph ==
    dense at 1e-12 (was: gated to dense / raises NotImplementedError).
    """
    rng = np.random.default_rng(4)
    nloc = 6
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
    box = np.eye(3).reshape(1, 9) * 20.0
    ds = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[200], ntypes=2, attn_layer=0)
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    model = EnergyModel(ds, ft, type_map=["a", "b"], pair_exclude_types=[(0, 1)])
    assert model.atomic_model.pair_excl is not None
    g = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    d = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    for k in ("energy", "energy_redu", "mask"):
        np.testing.assert_allclose(
            np.asarray(g[k]), np.asarray(d[k]), rtol=1e-12, atol=1e-12
        )
    # non-vacuous: toggle pair exclusion OFF on the SAME model (same weights),
    # so any energy difference is due solely to the exclusion (not weights).
    g_excl = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    model.atomic_model.reinit_pair_exclude([])  # clear pair exclusion
    assert model.atomic_model.pair_excl is None
    g_noexcl = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    # tight tolerance: the excluded (0,1) pairs contribute a small but real
    # amount; default rtol=1e-5 is too loose to register it.
    assert not np.allclose(
        np.asarray(g_excl["energy_redu"]),
        np.asarray(g_noexcl["energy_redu"]),
        rtol=1e-9,
        atol=1e-9,
    ), "pair exclusion must change the graph energy (same weights)"


def test_graph_matches_dense_with_fparam():
    """Frame parameter is gathered to nodes by frame_id in forward_atomic_graph
    and fed to the fitting's call_graph; the graph path must match dense at 1e-12
    with a non-zero fparam (exercises the frame_id gather + xp.take dispatch).
    """
    rng = np.random.default_rng(7)
    nf, nloc, ndf = 2, 5, 3
    coord = rng.normal(size=(nf, nloc, 3)) * 1.5
    atype = np.tile(np.array([[0, 1, 0, 1, 0]], dtype=np.int64), (nf, 1))
    box = np.tile(np.eye(3).reshape(1, 9) * 20.0, (nf, 1))
    fparam = rng.normal(size=(nf, ndf))  # per-frame, differs across frames
    ds = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[200], ntypes=2, attn_layer=0)
    ft = InvarFitting(
        "energy", 2, ds.get_dim_out(), 1, mixed_types=True, numb_fparam=ndf
    )
    model = EnergyModel(ds, ft, type_map=["a", "b"])
    g = model.call_common(
        coord, atype, box, fparam=fparam, neighbor_graph_method="dense"
    )
    d = model.call_common(
        coord, atype, box, fparam=fparam, neighbor_graph_method="legacy"
    )
    for k in ("energy", "energy_redu"):
        np.testing.assert_allclose(
            np.asarray(g[k]), np.asarray(d[k]), rtol=1e-12, atol=1e-12
        )
    # non-vacuous: each frame's fparam differs, so a mis-gathered fparam (e.g.
    # every node given frame 0's fparam) would make the two frames' energies equal.
    assert not np.allclose(
        np.asarray(g["energy_redu"][0]), np.asarray(g["energy_redu"][1])
    )


def test_graph_matches_dense_with_atom_exclude():
    """Model-level atom_exclude_types IS supported on the graph path (applied
    via _finalize_atomic_ret's atom_excl).  Graph == dense at rtol/atol 1e-12.
    Also proves atom-level exclusion is correctly inherited and non-vacuous.
    """
    rng = np.random.default_rng(11)
    nloc = 6
    coord = rng.normal(size=(1, nloc, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
    box = np.eye(3).reshape(1, 9) * 20.0
    ds = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[200], ntypes=2, attn_layer=0)
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    am = DPAtomicModel(ds, ft, type_map=["a", "b"], atom_exclude_types=[0])
    model = EnergyModel(atomic_model_=am)
    g = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    d = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    for k in ("energy", "energy_redu", "mask"):
        g_arr = np.asarray(g[k])
        d_arr = np.asarray(d[k])
        max_diff = float(np.max(np.abs(g_arr - d_arr)))
        np.testing.assert_allclose(
            g_arr,
            d_arr,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"graph vs dense mismatch for '{k}': max_diff={max_diff}",
        )
    # non-vacuous: type-0 atoms have zero energy (excluded), type-1 have nonzero
    g_energy = np.asarray(g["energy"])
    g_mask = np.asarray(g["mask"])
    type0_indices = atype[0] == 0
    assert np.allclose(g_energy[0, type0_indices], 0.0), (
        "excluded type-0 atoms must have zero energy"
    )
    assert not np.allclose(g_energy[0, ~type0_indices], 0.0), (
        "non-excluded type-1 atoms must have nonzero energy"
    )
    # also check mask: excluded type-0 atoms should have mask==0
    assert np.all(g_mask[0, type0_indices] == 0), (
        "excluded type-0 atoms must have mask==0"
    )


def test_forward_common_atomic_graph_flat_shape():
    rng = np.random.default_rng(1)
    coord = rng.normal(size=(1, 5, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)
    am = _atomic_model()
    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, 4.0, [30], mixed_types=True, box=None
    )
    ng = from_dense_quartet(ext_coord, nlist, mapping)
    out = am.forward_common_atomic_graph(ng, atype.reshape(-1))
    assert out["energy"].shape == (5, 1)  # flat (N, 1)
    assert out["mask"].shape == (5,)  # flat (N,)


def test_graph_nloc1_unravel_shapes():
    """Regression: when nloc==1, N==nf so per-frame _redu keys must NOT be
    reshaped to (nf,1,*).  Before the fix, energy_redu came out (nf,1,1) instead
    of (nf,1).  Checks both shapes and value parity against the dense (legacy) path.
    """
    nf = 2
    rng = np.random.default_rng(42)
    coord = rng.normal(size=(nf, 1, 3)) * 1.5
    atype = np.zeros((nf, 1), dtype=np.int64)
    box = np.tile(np.eye(3).reshape(1, 9) * 20.0, (nf, 1))
    model = _ener_model([200])  # non-binding sel
    g = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    d = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    # shape assertions — the critical regression check
    assert g["energy"].shape == (nf, 1, 1), f"energy shape {g['energy'].shape}"
    assert g["energy_redu"].shape == (nf, 1), (
        f"energy_redu shape {g['energy_redu'].shape}"
    )
    assert g["mask"].shape == (nf, 1), f"mask shape {g['mask'].shape}"
    # value parity with the dense path
    for k in ("energy", "energy_redu", "mask"):
        np.testing.assert_allclose(
            np.asarray(g[k]),
            np.asarray(d[k]),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"graph vs legacy mismatch for key '{k}'",
        )


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


# ── apply_pair_exclusion idempotence (Task 2) ─────────────────────────────────


@pytest.mark.parametrize(
    "pair_exclude_types", [[], [(0, 1)]]
)  # empty branch AND non-empty branch
def test_apply_pair_exclusion_idempotent(pair_exclude_types):
    """Applying apply_pair_exclusion twice gives the same edge_mask as once.

    Covers both the empty pair_excl branch (identity) and non-empty branch.
    """
    rng = np.random.default_rng(42)
    coord = rng.normal(size=(1, 5, 3)) * 1.5
    atype = np.array([[0, 1, 0, 1, 0]], dtype=np.int64)
    ds = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[200], ntypes=2, attn_layer=0)
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    am = DPAtomicModel(ds, ft, type_map=["a", "b"])
    ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, 4.0, [200], mixed_types=True, box=None
    )
    ng = from_dense_quartet(ext_coord, nlist, mapping)
    pair_excl = PairExcludeMask(2, pair_exclude_types) if pair_exclude_types else None
    atype_flat = atype.reshape(-1)
    once = apply_pair_exclusion(ng, atype_flat, pair_excl)
    twice = apply_pair_exclusion(once, atype_flat, pair_excl)
    # Masks must be exactly equal (AND-idempotent for 0/1 values)
    np.testing.assert_array_equal(
        np.asarray(once.edge_mask),
        np.asarray(twice.edge_mask),
    )
