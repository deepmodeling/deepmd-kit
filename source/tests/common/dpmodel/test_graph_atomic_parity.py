# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel
from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1
from deepmd.dpmodel.fitting import InvarFitting
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
