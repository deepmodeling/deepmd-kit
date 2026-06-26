# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flat-N ragged-native graph path: nodes on a flat (N,) axis, N = sum(n_node);
per-frame reductions use segment_sum over frame_id. UNEQUAL per-frame node counts
(ragged) -- the case the old rectangular (nf,nloc) path could not represent.
"""

import numpy as np

from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1
from deepmd.dpmodel.fitting import InvarFitting
from deepmd.dpmodel.model.ener_model import EnergyModel
from deepmd.dpmodel.utils.neighbor_graph import frame_id_from_n_node


def test_frame_id_ragged():
    fid = frame_id_from_n_node(np.array([3, 5, 2], dtype=np.int64))  # N=10
    np.testing.assert_array_equal(
        fid, np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=fid.dtype)
    )


def test_forward_common_atomic_graph_ragged():
    """Two frames with DIFFERENT node counts (3 and 2) share one flat node axis.

    The old rectangular path (nloc = N // nf) could not represent this.
    """
    import numpy as np

    from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel
    from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1
    from deepmd.dpmodel.fitting import InvarFitting
    from deepmd.dpmodel.utils.neighbor_graph import NeighborGraph

    ds = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[30], ntypes=2, attn_layer=0)
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    am = DPAtomicModel(ds, ft, type_map=["a", "b"])
    n_node = np.array([3, 2], dtype=np.int64)  # RAGGED, N=5
    atype = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    edge_index = np.array([[1, 0, 4], [0, 1, 3]], dtype=np.int64)  # within-frame
    edge_vec = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0.5, 0, 0]], dtype=np.float64)
    edge_mask = np.array([True, True, True])
    g = NeighborGraph(
        n_node=n_node, edge_index=edge_index, edge_vec=edge_vec, edge_mask=edge_mask
    )
    out = am.forward_common_atomic_graph(g, atype)
    assert out["energy"].shape == (5, 1) and out["mask"].shape == (5,)
    assert np.all(np.isfinite(out["energy"]))


def test_frame_id_rectangular():
    fid = frame_id_from_n_node(np.array([4, 4], dtype=np.int64))
    np.testing.assert_array_equal(
        fid, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=fid.dtype)
    )


def test_call_lower_graph_ragged_energy_reduction():
    """Per-frame energy_redu = segment_sum of the frame's atom energies; ragged."""
    import numpy as np

    from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1
    from deepmd.dpmodel.fitting import InvarFitting
    from deepmd.dpmodel.model.ener_model import EnergyModel

    ds = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=[30], ntypes=2, attn_layer=0)
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    m = EnergyModel(ds, ft, type_map=["a", "b"])
    n_node = np.array([3, 2], dtype=np.int64)
    atype = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    edge_index = np.array([[1, 0, 4], [0, 1, 3]], dtype=np.int64)
    edge_vec = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0.5, 0, 0]], dtype=np.float64)
    edge_mask = np.array([True, True, True])
    out = m.call_lower_graph(
        atype=atype,
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
    )
    assert out["energy"].shape == (5, 1)  # flat node energy
    assert out["energy_redu"].shape == (2, 1)  # per-FRAME reduced
    np.testing.assert_allclose(
        out["energy_redu"][0, 0],
        out["energy"][0:3, 0].sum(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out["energy_redu"][1, 0],
        out["energy"][3:5, 0].sum(),
        rtol=1e-12,
        atol=1e-12,
    )


def _ener_model_ragged(sel=(200,)):
    """Build a dpa1(attn_layer=0) EnergyModel for gate tests."""
    ds = DescrptDPA1(rcut=4.0, rcut_smth=0.5, sel=list(sel), ntypes=2, attn_layer=0)
    ft = InvarFitting("energy", 2, ds.get_dim_out(), 1, mixed_types=True)
    return EnergyModel(ds, ft, type_map=["a", "b"])


def test_rectangular_free_view_equivalence():
    """GATE: rectangular nf=2, nloc=5 graph path == legacy dense path bit-identical.

    Proves the flat-N rewrite does not perturb the rectangular special case.
    public call_common with neighbor_graph_method='dense' must match 'legacy'
    on energy / energy_redu / mask at rtol/atol 1e-12 (non-binding sel=[200]).
    """
    nf, nloc = 2, 5
    rng = np.random.default_rng(7)
    coord = rng.normal(size=(nf, nloc, 3)) * 1.5
    atype = np.tile(np.array([[0, 1, 0, 1, 0]], dtype=np.int64), (nf, 1))
    box = np.tile(np.eye(3).reshape(1, 9) * 20.0, (nf, 1))  # large PBC box
    model = _ener_model_ragged(sel=[200])  # non-binding sel
    g = model.call_common(coord, atype, box, neighbor_graph_method="dense")
    d = model.call_common(coord, atype, box, neighbor_graph_method="legacy")
    for k in ("energy", "energy_redu", "mask"):
        np.testing.assert_allclose(
            np.asarray(g[k]),
            np.asarray(d[k]),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"graph vs legacy mismatch for key '{k}'",
        )


def test_ragged_frames_independent():
    """GATE: ragged n_node=[3,2] per-frame energies equal two single-frame runs.

    Proves frames do not leak through segment_sum on the flat axis: the ragged
    energy_redu[i] must match running the i-th frame's atoms+edges in isolation
    through call_lower_graph.  The SAME model weights are used for all three
    calls so the comparison is meaningful.

    Frame 0: nodes 0-2 (atype [0,1,0]), edges 0<->1, 1<->2.
    Frame 1: nodes 3-4 (atype [1,0]), edges 3<->4 (global) = 0<->1 (local).
    """
    model = _ener_model_ragged()

    # ── Ragged graph (both frames in one flat call) ────────────────────────
    atype5 = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    #  frame-0 edges (global indices 0,1,2): 0↔1, 1↔2
    #  frame-1 edges (global indices 3,4):  3↔4
    edge_index_rag = np.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=np.int64)
    edge_vec_rag = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    edge_mask_rag = np.ones(6, dtype=bool)
    ragged = model.call_lower_graph(
        atype=atype5,
        n_node=np.array([3, 2], dtype=np.int64),
        edge_index=edge_index_rag,
        edge_vec=edge_vec_rag,
        edge_mask=edge_mask_rag,
    )

    # ── Single-frame 0 (nodes 0-2) ─────────────────────────────────────────
    atype_f0 = atype5[:3]  # [0, 1, 0]
    edge_index_f0 = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    edge_vec_f0 = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    edge_mask_f0 = np.ones(4, dtype=bool)
    f0 = model.call_lower_graph(
        atype=atype_f0,
        n_node=np.array([3], dtype=np.int64),
        edge_index=edge_index_f0,
        edge_vec=edge_vec_f0,
        edge_mask=edge_mask_f0,
    )

    # ── Single-frame 1 (nodes 3-4, remapped to local indices 0-1) ──────────
    atype_f1 = atype5[3:]  # [1, 0]  (atype of global nodes 3,4)
    # global edge 3→4 becomes local 0→1; global 4→3 becomes local 1→0
    edge_index_f1 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_vec_f1 = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=np.float64)
    edge_mask_f1 = np.ones(2, dtype=bool)
    f1 = model.call_lower_graph(
        atype=atype_f1,
        n_node=np.array([2], dtype=np.int64),
        edge_index=edge_index_f1,
        edge_vec=edge_vec_f1,
        edge_mask=edge_mask_f1,
    )

    # ── Gate assertions ────────────────────────────────────────────────────
    np.testing.assert_allclose(
        np.asarray(ragged["energy_redu"][0]),
        np.asarray(f0["energy_redu"][0]),
        rtol=1e-12,
        atol=1e-12,
        err_msg="ragged frame-0 energy_redu must equal single-frame-0 energy_redu",
    )
    np.testing.assert_allclose(
        np.asarray(ragged["energy_redu"][1]),
        np.asarray(f1["energy_redu"][0]),
        rtol=1e-12,
        atol=1e-12,
        err_msg="ragged frame-1 energy_redu must equal single-frame-1 energy_redu",
    )
