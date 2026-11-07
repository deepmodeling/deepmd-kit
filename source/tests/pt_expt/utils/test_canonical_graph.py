# SPDX-License-Identifier: LGPL-3.0-or-later
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)
from deepmd.pt_expt.utils.canonical_graph import (
    INT32_MAX,
    CanonicalGraph,
    canonical_graph_from_neighbor_graph,
    canonical_graph_from_pairs,
    canonical_index_bytes,
    canonical_index_words,
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
    assert compact.source_order.dtype == torch.int64
    assert int(compact.destination_row_ptr[-1]) == physical_edge_count
    assert int(compact.source_row_ptr[-1]) == physical_edge_count
    validate_canonical_graph_shapes(compact, 1)


@pytest.mark.parametrize("index_dtype", [torch.uint32, torch.int64])
def test_edge_storage_range_is_independent_of_node_width(
    index_dtype: torch.dtype,
) -> None:
    storage_count = 1 << 32
    graph = CanonicalGraph(
        n_node=torch.empty(1, dtype=torch.int64, device="meta"),
        n_local=torch.empty(1, dtype=torch.int64, device="meta"),
        source=torch.empty(storage_count, dtype=index_dtype, device="meta"),
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
            dtype=torch.int64,
            device="meta",
        ),
    )
    validate_canonical_graph_shapes(graph, 1)


@pytest.mark.parametrize(
    "node_count, expected_dtype",
    [(INT32_MAX + 1, torch.uint32), (INT32_MAX + 2, torch.int64)],
)
def test_pairs_choose_width_from_node_extent(
    node_count: int, expected_dtype: torch.dtype
) -> None:
    graph = canonical_graph_from_pairs(
        torch.empty(2, dtype=torch.int64, device="meta"),
        torch.empty(2, dtype=torch.int64, device="meta"),
        torch.empty(2, 3, dtype=torch.float32, device="meta"),
        torch.empty(1, dtype=torch.int64, device="meta"),
        node_count,
    )
    assert graph.source.dtype == expected_dtype
    assert graph.source_order.dtype == torch.int64
    validate_canonical_graph_shapes(graph, node_count)


@pytest.mark.parametrize("dtype", [torch.uint32, torch.int32, torch.int64])
def test_index_byte_carrier_preserves_words_without_copy(dtype: torch.dtype) -> None:
    values = [0, 7, (1 << 40) + 123] if dtype == torch.int64 else [0, 7, INT32_MAX]
    index = torch.tensor(values, dtype=dtype)
    storage = canonical_index_bytes(index)
    restored = canonical_index_words(storage)
    assert storage.shape == (3, index.element_size())
    assert storage.data_ptr() == restored.data_ptr() == index.data_ptr()
    torch.testing.assert_close(
        restored.to(torch.int64), index.to(torch.int64), atol=0, rtol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edge_count", [0, 1, 5])
def test_pairs_preserve_geometry_and_both_csr_views(
    device: str, edge_count: int
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    center = torch.tensor([3, 1, 0, 3, 1], device=device)[:edge_count]
    source = torch.tensor([2, 0, 1, 2, 1], device=device)[:edge_count]
    vectors = torch.arange(15, dtype=torch.float64, device=device).reshape(5, 3)
    graph = canonical_graph_from_pairs(
        center,
        source,
        vectors[:edge_count],
        torch.tensor([2, 99, 2, 99], device=device)[::2],
        4,
    )
    validate_canonical_graph_shapes(graph, 4)
    torch.testing.assert_close(
        graph.destination_row_ptr.diff(), torch.bincount(center, minlength=4)
    )
    torch.testing.assert_close(
        graph.source_row_ptr.diff(), torch.bincount(source, minlength=4)
    )
    destination = torch.repeat_interleave(
        torch.arange(4, device=device), graph.destination_row_ptr.diff()
    )
    payload = (
        torch.cat(
            (
                destination[:, None],
                graph.source[:edge_count].to(torch.int64)[:, None],
                graph.edge_vec[:edge_count],
            ),
            dim=1,
        )
        .cpu()
        .tolist()
    )
    expected = (
        torch.cat((center[:, None], source[:, None], vectors[:edge_count]), dim=1)
        .cpu()
        .tolist()
    )
    assert sorted(payload) == sorted(expected)
    by_source = graph.source.to(torch.int64).index_select(
        0, graph.source_order[:edge_count].to(torch.int64)
    )
    torch.testing.assert_close(
        by_source,
        torch.repeat_interleave(
            torch.arange(4, device=device), graph.source_row_ptr.diff()
        ),
    )
