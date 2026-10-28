# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the native CPU cell-list NeighborGraph builder."""

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
from deepmd.pt_expt.utils import (
    cell_graph_builder,
)

pytestmark = pytest.mark.skipif(
    not cell_graph_builder.is_cell_search_available(),
    reason="the native CPU cell-list search is unavailable",
)


def _system(periodic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return one four-atom system in the dense builder's batched layout."""
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]]],
        dtype=torch.float64,
    )
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64)
    box = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 3.0
    return coord, atype, box if periodic else None


def _valid_edge_set(graph) -> set[tuple[int, int, tuple[float, ...]]]:
    """Return the endpoint and displacement of every unmasked edge."""
    edge_index = np.asarray(graph.edge_index)
    edge_vec = np.asarray(graph.edge_vec)
    edge_mask = np.asarray(graph.edge_mask)
    return {
        (
            int(edge_index[0, edge]),
            int(edge_index[1, edge]),
            tuple(np.round(edge_vec[edge], 6)),
        )
        for edge in range(edge_index.shape[1])
        if edge_mask[edge]
    }


@pytest.mark.parametrize("periodic", [False, True])
def test_cell_matches_dense_with_csr(periodic: bool) -> None:
    """The native search changes the algorithm, not the graph contract."""
    coord, atype, box = _system(periodic)
    expected = build_neighbor_graph(coord, atype, box, 2.0)
    actual = cell_graph_builder.build_neighbor_graph_cell(
        coord, atype, box, 2.0, with_csr=True, canonicalize=True
    )

    assert _valid_edge_set(actual) == _valid_edge_set(expected)
    assert actual.destination_sorted
    assert actual.destination_row_ptr is not None
    assert actual.source_row_ptr is not None
    valid_edges = int(np.asarray(actual.edge_mask).sum())
    assert int(actual.destination_row_ptr[-1]) == valid_edges
    assert int(actual.source_row_ptr[-1]) == valid_edges


@pytest.mark.parametrize("periodic", [False, True])
def test_cell_pair_exclusion_matches_dense(periodic: bool) -> None:
    """Type exclusions preserve dense semantics before CSR canonicalization."""
    coord, atype, box = _system(periodic)
    pair_excl = PairExcludeMask(2, [(0, 1), (1, 0)])
    dense = build_neighbor_graph(coord, atype, box, 2.0)
    expected = apply_pair_exclusion(dense, atype.reshape(-1), pair_excl)
    actual = cell_graph_builder.build_neighbor_graph_cell(
        coord,
        atype,
        box,
        2.0,
        with_csr=True,
        canonicalize=True,
        pair_excl=pair_excl,
    )

    assert _valid_edge_set(actual) == _valid_edge_set(expected)


def test_cell_excludes_virtual_atoms_like_dense() -> None:
    """Virtual atoms are absent as both centers and neighbors."""
    coord, _, box = _system(periodic=True)
    atype = torch.tensor([[0, -1, 0, 1]], dtype=torch.int64)
    expected = build_neighbor_graph(coord, atype, box, 2.0)
    actual = cell_graph_builder.build_neighbor_graph_cell(coord, atype, box, 2.0)

    assert _valid_edge_set(actual) == _valid_edge_set(expected)
    edge_index = np.asarray(actual.edge_index)[:, np.asarray(actual.edge_mask)]
    flat_type = np.asarray(atype).reshape(-1)
    assert np.all(flat_type[edge_index[0]] >= 0)
    assert np.all(flat_type[edge_index[1]] >= 0)


def test_cell_edge_vectors_remain_differentiable() -> None:
    """Only the search is detached; displacements retain coordinate gradients."""
    coord, atype, box = _system(periodic=True)
    coord.requires_grad_(True)
    graph = cell_graph_builder.build_neighbor_graph_cell(coord, atype, box, 2.0)

    (graph.edge_vec.square().sum()).backward()
    assert coord.grad is not None
    assert torch.any(coord.grad != 0)
