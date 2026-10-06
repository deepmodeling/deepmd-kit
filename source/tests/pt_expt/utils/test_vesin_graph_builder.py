# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import (
    replace,
)

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
from deepmd.pt_expt.utils.canonical_graph import (
    canonical_graph_from_neighbor_graph,
    validate_canonical_graph_shapes,
)

vesin_builder = pytest.importorskip("deepmd.pt_expt.utils.vesin_graph_builder")
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)
from deepmd.pt_expt.utils.vesin_search import (
    VesinNeighborSearch,
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
    ei = ng.edge_index.detach().cpu().numpy()
    ev = ng.edge_vec.detach().cpu().numpy()
    em = ng.edge_mask.detach().cpu().numpy()
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_graph_tracks_geometry_and_atom_changes(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    coord, atype, box = _system(True)
    coord = coord.reshape(1, 4, 3).to(device)
    atype = atype.to(device)
    box = box.reshape(1, 3, 3).to(device)
    search = VesinNeighborSearch(skin=1.0, sorted=True)
    for step in range(9):
        points = coord.clone()
        types = atype.clone()
        cell = box.clone()
        if step == 1:
            points[0, 1, 0] += 0.2
        elif step == 2:
            points[0, 1, 0] += 0.8
        elif step == 3:
            cell *= 1.15
        elif step == 4:
            cell = None
        elif step == 5:
            types[0, 1] = -1
        elif step == 6:
            points = torch.cat((points, points[:, :1] + 0.4), dim=1)
            types = torch.cat((types, types[:, :1]), dim=1)
        elif step == 7:
            cell[0, 0, 1] += 0.3
        elif step == 8:
            points[0, 1] += cell[0, 0]
        actual = vesin_builder.build_neighbor_graph_vesin(
            points, types, cell, 2.0, search=search, canonicalize=True
        )
        expected = build_neighbor_graph(points, types, cell, 2.0, canonicalize=True)
        assert _valid_edge_set(actual) == _valid_edge_set(expected)
        torch.testing.assert_close(actual.n_node, expected.n_node)
        assert actual.destination_sorted


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_search_preserves_earlier_autograd_inputs(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    coord, atype, box = _system(True)
    coord = coord.reshape(1, 4, 3).to(device).requires_grad_()
    atype = atype.to(device)
    box = box.reshape(1, 3, 3).to(device)
    search = VesinNeighborSearch(skin=1.0, sorted=True)
    graph = vesin_builder.build_neighbor_graph_vesin(
        coord, atype, box, 2.0, search=search
    )
    moved = coord.detach().clone()
    moved[0, 1, 0] += 0.3
    vesin_builder.build_neighbor_graph_vesin(moved, atype, box, 2.0, search=search)
    actual = torch.autograd.grad(graph.edge_vec.square().sum(), coord)[0]
    reference = build_neighbor_graph(coord, atype, box, 2.0)
    expected = torch.autograd.grad(reference.edge_vec.square().sum(), coord)[0]
    torch.testing.assert_close(actual, expected)


def test_cached_search_keeps_frames_independent() -> None:
    coord, atype, box = _system(True)
    coord = coord.unsqueeze(0).repeat(2, 1, 1)
    atype = atype.repeat(2, 1)
    box = box.unsqueeze(0).repeat(2, 1, 1)
    box[1] *= 1.5
    search = VesinNeighborSearch(skin=1.0, sorted=True)
    for _ in range(3):
        actual = vesin_builder.build_neighbor_graph_vesin(
            coord, atype, box, 2.0, search=search, canonicalize=True
        )
        expected = build_neighbor_graph(coord, atype, box, 2.0, canonicalize=True)
        assert _valid_edge_set(actual) == _valid_edge_set(expected)
        coord[1, 1, 0] += 0.2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("nloc", [0, 1, 4])
@pytest.mark.parametrize("nframes", [1, 2])
def test_canonical_vesin_matches_dense(
    device: str, periodic: bool, nloc: int, nframes: int
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    coord, atype, box = _system(periodic)
    coord = coord[:nloc].unsqueeze(0).repeat(nframes, 1, 1).to(device)
    atype = atype[:, :nloc].repeat(nframes, 1).to(device)
    box = box.unsqueeze(0).repeat(nframes, 1, 1).to(device) if periodic else None
    if nloc == 4:
        atype[0, 1] = -1
    search = VesinNeighborSearch(skin=1.0)
    for step in range(3):
        points = coord.clone()
        if nloc:
            points[-1, 0, 0] += step * 0.3
        actual = vesin_builder.build_canonical_graph_vesin(
            points, box, 2.0, search, node_mask=atype >= 0 if nloc == 4 else None
        )
        reference = build_neighbor_graph(points, atype, box, 2.0, canonicalize=True)
        expected = canonical_graph_from_neighbor_graph(
            replace(reference, n_local=reference.n_node)
        )
        validate_canonical_graph_shapes(actual, nframes * nloc)
        torch.testing.assert_close(actual.n_node, expected.n_node)
        torch.testing.assert_close(
            actual.destination_row_ptr, expected.destination_row_ptr
        )
        torch.testing.assert_close(actual.source_row_ptr, expected.source_row_ptr)
        payloads = []
        for graph in (actual, expected):
            edge_count = int(graph.destination_row_ptr[-1])
            destination = torch.repeat_interleave(
                torch.arange(nframes * nloc, device=device),
                graph.destination_row_ptr.diff(),
            )
            values = (
                torch.cat(
                    (
                        destination[:, None],
                        graph.source[:edge_count].to(torch.int64)[:, None],
                        graph.edge_vec[:edge_count],
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            order = np.lexsort(values[:, ::-1].T)
            payloads.append(values[order])
        np.testing.assert_allclose(*payloads, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("return_vectors", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_search_handles_empty_nonempty_transition(
    device: str, return_vectors: bool
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    search = VesinNeighborSearch(skin=1.0)
    for distance in (8.0, 0.8, 8.0):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            dtype=torch.float64,
            device=device,
        )
        ii, jj, geometry = search.compute(
            positions, None, 2.0, return_vectors=return_vectors
        )
        assert ii.numel() == (2 if distance < 2.0 else 0)
        expected = positions[jj] - positions[ii] if return_vectors else geometry * 0
        torch.testing.assert_close(geometry, expected)
