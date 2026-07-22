# SPDX-License-Identifier: LGPL-3.0-or-later
"""nvalchemiops carry-all NeighborGraph builder: neighbor SET must equal the
in-tree ``dense`` carry-all reference. CUDA + nvalchemi-toolkit-ops only.
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)

nv_builder = pytest.importorskip("deepmd.pt_expt.utils.nv_graph_builder")
from deepmd.pt.utils.nv_nlist import (
    is_nv_available,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_nv_available()),
    reason="nvalchemiops requires CUDA + nvalchemi-toolkit-ops",
)


def _edge_multiset(ng):
    edge_index = np.asarray(ng.edge_index.cpu())
    edge_vec = np.asarray(ng.edge_vec.detach().cpu())
    edge_mask = np.asarray(ng.edge_mask.cpu())
    return sorted(
        (
            int(edge_index[0, edge]),
            int(edge_index[1, edge]),
            tuple(np.round(edge_vec[edge], 10)),
        )
        for edge in range(edge_index.shape[1])
        if edge_mask[edge]
    )


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC and PBC
def test_nv_matches_intree_carry_all(periodic):
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]]],
        dtype=torch.float64,
        device=dev,
    )
    box = (
        (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
        if periodic
        else None
    )
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64, device=dev)
    ng_ref = build_neighbor_graph(coord, atype, box, 2.0)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    assert _edge_multiset(ng) == _edge_multiset(ng_ref)


def test_nv_batches_frames_without_python_loop():
    """Multi-frame: nv searches all frames in one kernel (no per-frame loop)."""
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    coord = torch.tensor(rng.random((3, 5, 3)) * 3.0, dtype=torch.float64, device=dev)
    box = (
        (torch.eye(3, dtype=torch.float64, device=dev) * 4.0)
        .reshape(1, 3, 3)
        .repeat(3, 1, 1)
    )
    atype = torch.tensor(
        [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0]],
        dtype=torch.int64,
        device=dev,
    )
    ng_ref = build_neighbor_graph(coord, atype, box, 2.0)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    assert _edge_multiset(ng) == _edge_multiset(ng_ref)


def test_nv_edge_vec_is_differentiable():
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]]],
        dtype=torch.float64,
        device=dev,
    ).requires_grad_(True)
    box = (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64, device=dev)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    (ng.edge_vec**2).sum().backward()
    assert coord.grad is not None and torch.any(coord.grad != 0)


def test_nv_excludes_virtual_atoms_like_dense():
    """Virtual atoms (atype < 0) excluded as center AND neighbor (dense contract)."""
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]]],
        dtype=torch.float64,
        device=dev,
    )
    box = (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
    atype = torch.tensor([[0, -1, 0, 1]], dtype=torch.int64, device=dev)  # 1 virtual
    ng_ref = build_neighbor_graph(coord, atype, box, 2.0)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    assert _edge_multiset(ng) == _edge_multiset(ng_ref)
    ei = np.asarray(ng.edge_index.cpu())[:, np.asarray(ng.edge_mask.cpu())]
    at = atype.reshape(-1).cpu().numpy()
    assert np.all(at[ei[0]] >= 0) and np.all(at[ei[1]] >= 0)


def test_nv_pair_exclusion_matches_dense():
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 1.1, 0.0], [1.2, 1.0, 0.0]]],
        dtype=torch.float64,
        device=dev,
    )
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64, device=dev)
    box = (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
    pair_excl = PairExcludeMask(2, [(0, 1), (1, 0)])
    expected = build_neighbor_graph(
        coord,
        atype,
        box,
        2.0,
        pair_excl=pair_excl,
    )
    actual = nv_builder.build_neighbor_graph_nv(
        coord,
        atype,
        box,
        2.0,
        pair_excl=pair_excl,
    )
    assert _edge_multiset(actual) == _edge_multiset(expected)
