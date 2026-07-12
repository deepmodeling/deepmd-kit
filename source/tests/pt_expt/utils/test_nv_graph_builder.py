# SPDX-License-Identifier: LGPL-3.0-or-later
"""nvalchemiops carry-all NeighborGraph builder: neighbor SET must equal the
in-tree ``dense`` carry-all reference. CUDA + nvalchemi-toolkit-ops only.
"""

import numpy as np
import pytest
import torch

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


def _sets(ng, nloc):
    """Per-center set of (src_local, rounded edge_vec) over real edges."""
    ei = np.asarray(ng.edge_index.cpu())
    ev = np.asarray(ng.edge_vec.detach().cpu())
    em = np.asarray(ng.edge_mask.cpu())
    out = {c: set() for c in range(nloc)}
    for e in range(ei.shape[1]):
        if em[e]:
            out[int(ei[1, e])].add((int(ei[0, e]), tuple(np.round(ev[e], 6))))
    return out


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
    assert _sets(ng, 4) == _sets(ng_ref, 4)


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
    # per-frame node offset: frame f centers occupy nodes [f*5, (f+1)*5)
    for f in range(3):
        s_ref = {
            (
                int(ng_ref.edge_index[0, e]),
                tuple(np.round(np.asarray(ng_ref.edge_vec[e].detach().cpu()), 6)),
            )
            for e in range(ng_ref.edge_index.shape[1])
            if bool(ng_ref.edge_mask[e])
            and f * 5 <= int(ng_ref.edge_index[1, e]) < (f + 1) * 5
        }
        s = {
            (
                int(ng.edge_index[0, e]),
                tuple(np.round(np.asarray(ng.edge_vec[e].detach().cpu()), 6)),
            )
            for e in range(ng.edge_index.shape[1])
            if bool(ng.edge_mask[e]) and f * 5 <= int(ng.edge_index[1, e]) < (f + 1) * 5
        }
        assert s == s_ref, f"frame {f} neighbor set mismatch"


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
    assert _sets(ng, 4) == _sets(ng_ref, 4)
    ei = np.asarray(ng.edge_index.cpu())[:, np.asarray(ng.edge_mask.cpu())]
    at = atype.reshape(-1).cpu().numpy()
    assert np.all(at[ei[0]] >= 0) and np.all(at[ei[1]] >= 0)
