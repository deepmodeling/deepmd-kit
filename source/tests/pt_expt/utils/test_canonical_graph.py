# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import (
    replace,
)
from unittest import (
    mock,
)

import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    attach_edge_csr,
    build_neighbor_graph,
)
from deepmd.pt_expt.utils.canonical_graph import (
    INT32_MAX,
    BorderPlan,
    CanonicalGraph,
    canonical_graph_from_neighbor_graph,
    canonical_graph_from_pairs,
    canonical_index_bytes,
    canonical_index_values,
    canonical_index_words,
    neighbor_graph_from_canonical,
    validate_canonical_graph_shapes,
    validate_reciprocal_canonical_graph,
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


def test_reciprocity_checks_periodic_images_and_duplicate_counts() -> None:
    center = torch.tensor([0, 0, 1, 1, 0, 0])
    source = torch.tensor([1, 1, 0, 0, 0, 0])
    vector = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
        ]
    )
    for count in (0, 4, 6):
        graph = canonical_graph_from_pairs(
            center[:count], source[:count], vector[:count], torch.tensor([2]), 2
        )
        validate_reciprocal_canonical_graph(graph)
    for count in (1, 3, 5):
        graph = canonical_graph_from_pairs(
            center[:count], source[:count], vector[:count], torch.tensor([2]), 2
        )
        with pytest.raises(ValueError, match="bidirectional"):
            validate_reciprocal_canonical_graph(graph)
    vector[2] = vector[3]
    graph = canonical_graph_from_pairs(center, source, vector, torch.tensor([2]), 2)
    with pytest.raises(ValueError, match="bidirectional"):
        validate_reciprocal_canonical_graph(graph)


def test_reciprocity_matching_accepts_periodic_roundoff() -> None:
    graph = canonical_graph_from_pairs(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 1, 0, 0]),
        torch.tensor(
            [[1e-7, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0], [-1e-7, -2.0, 0.0]]
        ),
        torch.tensor([2]),
        2,
    )
    validate_reciprocal_canonical_graph(graph)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_periodic_self_images_are_matched_without_host_transfers(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    coord = torch.zeros(4, 1, 3, dtype=torch.float64, device=device)
    types = torch.zeros(4, 1, dtype=torch.int64, device=device)
    box = 1.5 * torch.eye(3, dtype=torch.float64, device=device).repeat(4, 1, 1)
    neighbor = build_neighbor_graph(coord, types, box, 2.0, canonicalize=True)
    graph = canonical_graph_from_neighbor_graph(
        replace(neighbor, n_local=neighbor.n_node)
    )
    with mock.patch.object(
        torch.Tensor, "cpu", side_effect=AssertionError("host transfer")
    ):
        validate_reciprocal_canonical_graph(graph)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_only_ambiguous_image_groups_reach_host_matching(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    center = torch.tensor([0, 0, 1, 1, 2, 3, 4, 4], device=device)
    source = torch.tensor([1, 1, 0, 0, 3, 2, 4, 4], device=device)
    vector = torch.tensor(
        [
            [1e-7, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1e-7, -2.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        device=device,
    )
    graph = canonical_graph_from_pairs(
        center, source, vector, torch.tensor([5], device=device), 5
    )
    cpu = torch.Tensor.cpu
    transfers = []

    def record_transfer(tensor: torch.Tensor) -> torch.Tensor:
        transfers.append(tuple(tensor.shape))
        return cpu(tensor)

    with mock.patch.object(torch.Tensor, "cpu", record_transfer):
        validate_reciprocal_canonical_graph(graph)
    assert [shape for shape in transfers if len(shape) == 2] == [(4, 3), (4, 3)]


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
    for carrier in (storage, index):
        torch.testing.assert_close(
            canonical_index_values(carrier), index.to(torch.int64), atol=0, rtol=0
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edge_count", [0, 1, 7])
def test_neighbor_graph_view_spreads_guards_over_the_nodes(
    device: str, edge_count: int
) -> None:
    """The view of a compact graph keeps every edge and its CSR views.

    The physical edges keep their endpoints and geometry. The guard slots,
    here the storage that a Kokkos build reserves past the physical edges,
    join the nodes to themselves in turn and are masked. Both CSR views are
    those that the graph builder derives from the edges and their mask.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator(device="cpu").manual_seed(3)
    center = torch.randint(0, 4, (edge_count,), generator=generator).to(device)
    source = torch.randint(0, 4, (edge_count,), generator=generator).to(device)
    vectors = torch.randn(edge_count, 3, generator=generator, dtype=torch.float64)
    compact = canonical_graph_from_pairs(
        center,
        source,
        vectors.to(device),
        torch.tensor([4], dtype=torch.int64, device=device),
        4,
    )
    reserved = compact.source.shape[0]
    storage = reserved + 5
    compact = replace(
        compact,
        source=torch.cat((compact.source, compact.source.new_zeros(5))),
        edge_vec=torch.cat((compact.edge_vec, compact.edge_vec.new_zeros(5, 3))),
        source_order=torch.cat(
            (compact.source_order, torch.arange(reserved, storage, device=device))
        ),
    )
    view = neighbor_graph_from_canonical(compact, 4)
    assert view.edge_index.shape == (2, storage)
    assert view.destination_sorted
    physical = int(compact.destination_row_ptr[-1])
    slot = torch.arange(storage, device=device)
    torch.testing.assert_close(view.edge_mask, slot < physical)
    torch.testing.assert_close(
        view.edge_index[0, :physical], compact.source[:physical].to(torch.int64)
    )
    torch.testing.assert_close(
        view.edge_index[1, :physical],
        torch.repeat_interleave(
            torch.arange(4, device=device), compact.destination_row_ptr.diff()
        ),
    )
    torch.testing.assert_close(
        view.edge_index[:, physical:], (slot[physical:] % 4).expand(2, -1)
    )
    rebuilt = attach_edge_csr(view, 4, destination_sorted=True)
    for name in (
        "destination_order",
        "destination_row_ptr",
        "source_order",
        "source_row_ptr",
    ):
        torch.testing.assert_close(getattr(view, name), getattr(rebuilt, name))


def _border_exchange_available() -> bool:
    """Whether the loaded operator library carries the border exchange."""
    import deepmd.pt.cxx_op  # noqa: F401

    return hasattr(torch.ops.deepmd, "border_exchange")


def _border_plan(device: str) -> tuple[BorderPlan, torch.Tensor]:
    """A plan of three self swaps whose last forwards a row the first received.

    Returns the plan of a node set of four owned rows and four halo rows, and
    the row every row of that set holds after the exchange.
    """
    plan = BorderPlan(
        send_nodes=torch.tensor([1, 3, 0, 4], dtype=torch.int32, device=device),
        swaps=torch.tensor(
            [[0, 0, 2, 2, 4], [0, 0, 1, 1, 6], [0, 0, 1, 1, 7]], dtype=torch.int32
        ),
        communicator=torch.zeros(1, dtype=torch.int64),
    )
    return plan, torch.tensor([0, 1, 2, 3, 1, 3, 0, 1], device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_border_exchange_refreshes_halo_rows_and_returns_their_gradient(
    device: str, dtype: torch.dtype
) -> None:
    """The exchange copies owners into halo rows; its gradient folds back.

    The halo rows take the rows of the atoms they image, an image of an image
    included, and the gradient of every halo row is added onto that atom,
    while the halo rows, whose input values the exchange overwrites, receive
    none: the exchange matches the gather by the owner of every row.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not _border_exchange_available():
        pytest.skip("the operator library lacks the border exchange")
    plan, owner = _border_plan(device)
    generator = torch.Generator(device="cpu").manual_seed(5)
    rows = torch.randn(8, 6, generator=generator, dtype=dtype).to(device)
    cotangent = torch.randn(8, 6, generator=generator, dtype=dtype).to(device)
    rows.requires_grad_(True)
    refreshed = plan.exchange(rows)
    (gradient,) = torch.autograd.grad(refreshed, rows, cotangent)
    reference_rows = rows.detach().clone().requires_grad_(True)
    reference = reference_rows[owner]
    (reference_gradient,) = torch.autograd.grad(reference, reference_rows, cotangent)
    torch.testing.assert_close(refreshed, reference, rtol=0, atol=0)
    torch.testing.assert_close(gradient, reference_gradient)
    assert bool((gradient[4:] == 0).all())
