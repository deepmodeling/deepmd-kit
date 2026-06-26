# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flat-N ragged-native graph path: nodes on a flat (N,) axis, N = sum(n_node);
per-frame reductions use segment_sum over frame_id. UNEQUAL per-frame node counts
(ragged) -- the case the old rectangular (nf,nloc) path could not represent.
"""

import numpy as np

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
