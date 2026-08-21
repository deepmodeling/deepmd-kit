# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)
from deepmd.pt_expt.utils.canonical_graph import (
    UINT32_MAX,
    CanonicalGraph,
    canonical_graph_from_neighbor_graph,
    validate_canonical_graph_shapes,
)


def _generic_graph(physical_edge_count: int) -> NeighborGraph:
    storage_count = physical_edge_count + 2
    edge_index = torch.zeros((2, storage_count), dtype=torch.int64)
    edge_vec = torch.zeros((storage_count, 3), dtype=torch.float64)
    edge_mask = torch.zeros(storage_count, dtype=torch.bool)
    if physical_edge_count:
        edge_mask[:physical_edge_count] = True
        edge_index[0, :physical_edge_count] = 0
        edge_index[1, :physical_edge_count] = 0
        edge_vec[:physical_edge_count, 0] = 1.5
    row_ptr = torch.tensor([0, physical_edge_count], dtype=torch.int64)
    return NeighborGraph(
        n_node=torch.tensor([1], dtype=torch.int64),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
        n_local=torch.tensor([1], dtype=torch.int64),
        destination_order=torch.arange(storage_count, dtype=torch.int64),
        destination_row_ptr=row_ptr,
        source_order=torch.arange(storage_count, dtype=torch.int64),
        source_row_ptr=row_ptr.clone(),
        destination_sorted=True,
    )


@pytest.mark.parametrize("physical_edge_count", [0, 1])
def test_storage_guards_remain_outside_csr(physical_edge_count: int) -> None:
    compact = canonical_graph_from_neighbor_graph(_generic_graph(physical_edge_count))
    assert compact.source.shape == (2,)
    assert compact.edge_vec.shape == (2, 3)
    assert compact.source_order.shape == (2,)
    assert compact.source.dtype == torch.uint32
    assert compact.source_order.dtype == torch.uint32
    assert int(compact.destination_row_ptr[-1]) == physical_edge_count
    assert int(compact.source_row_ptr[-1]) == physical_edge_count
    validate_canonical_graph_shapes(compact, 1)


def test_storage_exceeding_uint32_range_is_rejected() -> None:
    storage_count = UINT32_MAX + 1
    graph = CanonicalGraph(
        n_node=torch.empty(1, dtype=torch.int64, device="meta"),
        n_local=torch.empty(1, dtype=torch.int64, device="meta"),
        source=torch.empty(storage_count, dtype=torch.uint32, device="meta"),
        edge_vec=torch.empty(
            storage_count,
            3,
            dtype=torch.float32,
            device="meta",
        ),
        destination_row_ptr=torch.empty(2, dtype=torch.int64, device="meta"),
        source_row_ptr=torch.empty(2, dtype=torch.int64, device="meta"),
        source_order=torch.empty(
            storage_count,
            dtype=torch.uint32,
            device="meta",
        ),
    )
    with pytest.raises(ValueError, match="exceeds the uint32 range"):
        validate_canonical_graph_shapes(graph, 1)
