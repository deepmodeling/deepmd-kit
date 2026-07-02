# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import build_neighbor_graph

vesin_builder = pytest.importorskip("deepmd.pt_expt.utils.vesin_graph_builder")
from deepmd.pt_expt.utils.vesin_neighbor_list import is_vesin_torch_available

pytestmark = pytest.mark.skipif(
    not is_vesin_torch_available(), reason="vesin[torch] not installed"
)


def _sets(ng, nloc):
    """Per-center set of (src_local, rounded edge_vec) over real edges."""
    ei = np.asarray(ng.edge_index)
    ev = np.asarray(ng.edge_vec)
    em = np.asarray(ng.edge_mask)
    out = {c: set() for c in range(nloc)}
    for e in range(ei.shape[1]):
        if not em[e]:
            continue
        src, dst = int(ei[0, e]), int(ei[1, e])
        out[dst].add((src, tuple(np.round(ev[e], 6))))
    return out


def _system(periodic):
    coord = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]],
        dtype=torch.float64,
    )
    box = torch.eye(3, dtype=torch.float64) * 3.0 if periodic else None
    # atype is (nf, nloc) = (1, 4); build_neighbor_graph requires 2-D atype
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
    return coord, atype, box


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC and PBC
def test_vesin_matches_intree_carry_all(periodic):
    coord, atype, box = _system(periodic)
    rcut = 2.0
    ng_ref = build_neighbor_graph(
        coord.reshape(1, 4, 3),
        atype,
        None if box is None else box.reshape(1, 3, 3),
        rcut,
    )
    ng = vesin_builder.build_neighbor_graph_vesin(
        coord.reshape(1, 4, 3),
        atype,
        None if box is None else box.reshape(1, 3, 3),
        rcut,
    )
    assert _sets(ng, 4) == _sets(ng_ref, 4)


def test_vesin_outputs_on_input_device():
    coord, atype, box = _system(True)
    ng = vesin_builder.build_neighbor_graph_vesin(
        coord.reshape(1, 4, 3), atype, box.reshape(1, 3, 3), 2.0
    )
    assert ng.edge_index.device.type == coord.device.type
    assert ng.edge_vec.device.type == coord.device.type


def test_vesin_empty_system():
    coord = torch.zeros((1, 0, 3), dtype=torch.float64)
    atype = torch.zeros((0,), dtype=torch.int64)
    ng = vesin_builder.build_neighbor_graph_vesin(coord, atype, None, 2.0)
    assert bool(ng.edge_mask.any()) is False  # only min_edges guard edges


def test_vesin_edge_vec_is_differentiable():
    coord, atype, box = _system(True)
    coord = coord.reshape(1, 4, 3).requires_grad_(True)
    ng = vesin_builder.build_neighbor_graph_vesin(
        coord, atype, box.reshape(1, 3, 3), 2.0
    )
    # Use squared sum: with full_list=True every edge (i,j,S) has a reverse (j,i,-S)
    # so edge_vec.sum() = 0 and its gradient is identically zero.  The squared
    # loss is asymmetric and gives a non-trivial, non-cancelling gradient.
    (ng.edge_vec**2).sum().backward()
    assert coord.grad is not None and torch.any(coord.grad != 0)
