# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    apply_pair_exclusion,
    build_neighbor_graph,
)

vesin_builder = pytest.importorskip("deepmd.pt_expt.utils.vesin_graph_builder")
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)

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


def test_vesin_excludes_virtual_atoms_like_dense():
    """Virtual atoms (atype < 0) excluded as center AND neighbor (dense contract)."""
    coord, _, box = _system(periodic=True)
    atype = torch.tensor([[0, -1, 0, 1]], dtype=torch.int64)  # atom 1 virtual
    rcut = 2.0
    ng_ref = build_neighbor_graph(
        coord.reshape(1, 4, 3), atype, box.reshape(1, 3, 3), rcut
    )
    ng = vesin_builder.build_neighbor_graph_vesin(
        coord.reshape(1, 4, 3), atype, box.reshape(1, 3, 3), rcut
    )
    assert _sets(ng, 4) == _sets(ng_ref, 4)
    ei = np.asarray(ng.edge_index)[:, np.asarray(ng.edge_mask)]
    at = atype.reshape(-1).numpy()
    assert np.all(at[ei[0]] >= 0) and np.all(at[ei[1]] >= 0)


def _valid_edge_set(ng):
    """Return the set of (src, dst, rounded edge_vec) for all real edges."""
    ei = np.asarray(ng.edge_index)
    ev = np.asarray(ng.edge_vec)
    em = np.asarray(ng.edge_mask)
    return {
        (int(ei[0, k]), int(ei[1, k]), tuple(np.round(ev[k], 6)))
        for k in range(ei.shape[1])
        if em[k]
    }


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC and PBC
def test_vesin_pair_excl_none_identity(periodic):
    """pair_excl=None: vesin builder output is unchanged (identity)."""
    coord, atype, box = _system(periodic)
    coord = coord.reshape(1, 4, 3)
    box_3d = None if box is None else box.reshape(1, 3, 3)
    ng_ref = vesin_builder.build_neighbor_graph_vesin(coord, atype, box_3d, 2.0)
    ng_excl = vesin_builder.build_neighbor_graph_vesin(
        coord, atype, box_3d, 2.0, pair_excl=None
    )
    assert _valid_edge_set(ng_ref) == _valid_edge_set(ng_excl)


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC and PBC
def test_vesin_pair_excl_oracle_set_equality(periodic):
    """Vesin builder(pair_excl=X) == dense ref + apply_pair_exclusion(X)."""
    coord, atype, box = _system(periodic)
    coord = coord.reshape(1, 4, 3)
    box_3d = None if box is None else box.reshape(1, 3, 3)
    rcut = 2.0
    pe = PairExcludeMask(2, [(0, 1), (1, 0)])
    # dense reference + separate post-process
    ng_dense = build_neighbor_graph(coord, atype, box_3d, rcut)
    atype_flat = atype.reshape(-1)
    ng_ref = apply_pair_exclusion(ng_dense, atype_flat, pe)
    # vesin builder with fused post-process
    ng_vesin = vesin_builder.build_neighbor_graph_vesin(
        coord, atype, box_3d, rcut, pair_excl=pe
    )
    assert _valid_edge_set(ng_ref) == _valid_edge_set(ng_vesin)
    # exclusion actually removed edges
    ng_plain = vesin_builder.build_neighbor_graph_vesin(coord, atype, box_3d, rcut)
    assert int(np.asarray(ng_vesin.edge_mask).sum()) < int(
        np.asarray(ng_plain.edge_mask).sum()
    )


def test_vesin_nlist_edges_pair_excl_raises():
    """VesinNeighborList.build with return_mode='edges' and pair_excl raises NotImplementedError."""
    from deepmd.pt_expt.utils.vesin_neighbor_list import (
        VesinNeighborList,
    )

    coord = torch.zeros((1, 4, 3), dtype=torch.float64)
    atype = torch.zeros((1, 4), dtype=torch.int64)
    pe = PairExcludeMask(2, [(0, 1)])
    nl = VesinNeighborList()
    with pytest.raises(NotImplementedError, match="return_mode='edges'"):
        nl.build(coord, atype, None, 2.0, [4], return_mode="edges", pair_excl=pe)
