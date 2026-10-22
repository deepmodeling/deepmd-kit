# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact canonical graph contract for compressed CUDA deployment."""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import torch

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )

INT32_MAX = (1 << 31) - 1


def canonical_index_dtype(node_count: int) -> torch.dtype:
    """Choose node-index width; edge counts and CSR positions always use int64.

    Four-byte words retain a signed representation at native host boundaries.
    """
    return torch.uint32 if node_count <= INT32_MAX + 1 else torch.int64


def canonical_index_bytes(index: torch.Tensor) -> torch.Tensor:
    """Expose contiguous integer words as bytes with a final word-size axis."""
    return index.view(torch.uint8).reshape(*index.shape, index.element_size())


def canonical_index_words(storage: torch.Tensor) -> torch.Tensor:
    """Restore a canonical byte carrier to integer words without copying."""
    if storage.dtype != torch.uint8:
        return storage
    if storage.ndim < 2 or not storage.is_contiguous():
        raise ValueError(
            "canonical index bytes require contiguous storage and a word axis"
        )
    width = storage.shape[-1]
    if width not in (4, 8):
        raise ValueError(
            f"canonical index word width must be 4 or 8 bytes, got {width}"
        )
    dtype = torch.int64 if width == 8 else torch.uint32
    return storage.view(dtype).squeeze(-1)


@dataclass(frozen=True)
class CanonicalGraph:
    """Store a cutoff-compacted destination-major graph without redundant fields.

    The physical edge count is the final value of the destination row-pointer
    tensor. The storage edge axis has length at least ``max(E, 2)``; entries
    past the physical edges are guards excluded from every CSR row.

    The source CSR serves the consumers that group edges by source: the DPA1
    force assembly and the source-attributed per-atom virial of DPA4C. A DPA4C
    graph whose caller declines the per-atom virial may carry two-entry
    placeholders in its place.

    Parameters
    ----------
    n_node
        Per-frame total node counts with shape ``(nf,)``, int64.
    n_local
        Per-frame owned node counts with shape ``(nf,)``, int64.
    source
        Source-node index for each edge storage slot with shape ``(S,)``,
        uint32 or int64, selected from the node extent.
        Exported programs carry these words as uint8 with shape ``(S, W)``,
        where the dynamic word width ``W`` is four or eight bytes.
    edge_vec
        Neighbor-minus-center vectors with shape ``(S, 3)``, float32.
    destination_row_ptr
        Destination CSR offsets with shape ``(N + 1,)``, int64.
    source_row_ptr
        Source CSR offsets with shape ``(N + 1,)``, int64, or a placeholder.
    source_order
        Edge storage positions grouped by source with shape ``(S,)``, int64,
        or a placeholder. This width is independent of the node indices.
    """

    n_node: torch.Tensor
    n_local: torch.Tensor
    source: torch.Tensor
    edge_vec: torch.Tensor
    destination_row_ptr: torch.Tensor
    source_row_ptr: torch.Tensor
    source_order: torch.Tensor


def validate_canonical_graph_shapes(
    graph: CanonicalGraph,
    node_count: int,
) -> None:
    """Validate shape, dtype, and device invariants without reading tensor data.

    The extents of the source CSR are left to the operators that read it,
    because a placeholder is a valid source CSR for a DPA4C graph and the
    deployment trace keeps its extents independent of the node and edge axes.
    """
    index_dtype = graph.source.dtype
    byte_storage = index_dtype == torch.uint8
    if index_dtype not in (torch.uint8, torch.uint32, torch.int64):
        raise ValueError(
            "canonical graph source must be uint32, int64 or exported bytes"
        )
    if graph.source_order.dtype != torch.int64:
        raise ValueError("canonical graph source_order must be int64")
    if graph.edge_vec.dtype != torch.float32:
        raise ValueError("canonical graph edge_vec must be float32")
    if graph.n_node.dtype != torch.int64 or graph.n_local.dtype != torch.int64:
        raise ValueError("canonical graph node counts must be int64")
    if (
        graph.destination_row_ptr.dtype != torch.int64
        or graph.source_row_ptr.dtype != torch.int64
    ):
        raise ValueError("canonical graph row pointers must be int64")
    if graph.source.ndim != (2 if byte_storage else 1) or graph.source_order.ndim != 1:
        raise ValueError("canonical graph source arrays have invalid rank")
    # A traced word width is a symbol of range [4, 8]: a bound adds no guard,
    # and the operators that read the words accept only four or eight bytes.
    if byte_storage and not 4 <= graph.source.shape[-1] <= 8:
        raise ValueError("canonical index word width must be four or eight bytes")
    if index_dtype == torch.uint32 and node_count > INT32_MAX + 1:
        raise ValueError("canonical graph nodes require int64 indices")
    if graph.edge_vec.shape != (graph.source.shape[0], 3):
        raise ValueError("canonical edge_vec must have shape (S, 3) matching source")
    if graph.source.shape[0] < 2:
        raise ValueError("canonical graph edge storage must contain at least two slots")
    if graph.destination_row_ptr.shape != (node_count + 1,):
        raise ValueError("destination_row_ptr must have shape (N + 1,)")
    if graph.source_row_ptr.ndim != 1:
        raise ValueError("source_row_ptr must be one-dimensional")
    if graph.n_node.shape != graph.n_local.shape or graph.n_node.ndim != 1:
        raise ValueError("n_node and n_local must share shape (nf,)")
    tensors = (
        graph.n_node,
        graph.n_local,
        graph.source,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("canonical graph tensors must reside on one device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("canonical graph tensors must be contiguous")


def canonical_graph_from_neighbor_graph(
    graph: NeighborGraph,
) -> CanonicalGraph:
    """Convert a compact generic graph into the source-only deployment contract.

    Parameters
    ----------
    graph
        Destination-major graph whose physical edges form a valid prefix.

    Returns
    -------
    CanonicalGraph
        Source-only graph with exactly two storage slots when ``E < 2``.

    Raises
    ------
    ValueError
        If the graph is not compact, canonical, or dual-CSR.
    """
    if not graph.destination_sorted:
        raise ValueError("canonical deployment requires destination-major payload")
    if (
        graph.destination_row_ptr is None
        or graph.source_order is None
        or graph.source_row_ptr is None
    ):
        raise ValueError("canonical deployment requires destination/source CSR")
    if graph.n_local is None:
        raise ValueError("canonical deployment requires explicit owned-node counts")

    physical_edge_count = int(graph.destination_row_ptr[-1].item())
    if int(graph.source_row_ptr[-1].item()) != physical_edge_count:
        raise ValueError("destination/source CSR physical edge counts differ")
    if physical_edge_count > graph.edge_index.shape[1]:
        raise ValueError("CSR physical edge count exceeds edge storage")
    if not bool(torch.all(graph.edge_mask[:physical_edge_count]).item()):
        raise ValueError("canonical deployment requires all physical edges active")
    if bool(torch.any(graph.edge_mask[physical_edge_count:]).item()):
        raise ValueError("canonical deployment guards must form an inactive suffix")

    node_count = graph.destination_row_ptr.shape[0] - 1

    storage_edge_count = max(physical_edge_count, 2)
    source = torch.zeros(
        storage_edge_count,
        dtype=canonical_index_dtype(node_count),
        device=graph.edge_index.device,
    )
    edge_vec = torch.zeros(
        storage_edge_count,
        3,
        dtype=torch.float32,
        device=graph.edge_vec.device,
    )
    source_order = torch.arange(
        storage_edge_count,
        dtype=torch.int64,
        device=graph.edge_index.device,
    )
    if physical_edge_count:
        source[:physical_edge_count] = graph.edge_index[0, :physical_edge_count].to(
            source.dtype
        )
        edge_vec[:physical_edge_count] = graph.edge_vec[:physical_edge_count].to(
            torch.float32
        )
        source_order[:physical_edge_count] = graph.source_order[
            :physical_edge_count
        ].to(torch.int64)

    result = CanonicalGraph(
        n_node=graph.n_node.contiguous(),
        n_local=graph.n_local.contiguous(),
        source=source,
        edge_vec=edge_vec,
        destination_row_ptr=graph.destination_row_ptr.contiguous(),
        source_row_ptr=graph.source_row_ptr.contiguous(),
        source_order=source_order,
    )
    validate_canonical_graph_shapes(result, node_count)
    return result


def canonical_graph_from_pairs(
    center: torch.Tensor,
    source: torch.Tensor,
    edge_vec: torch.Tensor,
    n_node: torch.Tensor,
    node_count: int,
) -> CanonicalGraph:
    """Build a local canonical graph from active, unpadded neighbor pairs.

    Parameters
    ----------
    center, source : torch.Tensor
        Valid global node indices with shape (E,), int64, on one device.
    edge_vec : torch.Tensor
        Neighbor-minus-center vectors in angstrom with shape (E, 3), on the
        index device. Geometry is converted to float32 for deployment.
    n_node : torch.Tensor
        Per-frame local node counts with shape (nf,), int64, on the same device.
    node_count : int
        Total number of local nodes across frames, equal to ``n_node.sum()``.

    Returns
    -------
    CanonicalGraph
        Destination-major graph with dual CSR and at least two storage slots.
        All nodes are local; no coordinate differentiation is required.
    """
    edge_count = source.shape[0]
    destination_order = torch.argsort(center, stable=True)
    center = center.index_select(0, destination_order)
    source = source.index_select(0, destination_order)
    edge_vec = edge_vec.index_select(0, destination_order).to(torch.float32)
    boundaries = torch.arange(node_count + 1, dtype=torch.int64, device=source.device)
    destination_row_ptr = torch.searchsorted(center, boundaries)
    source_order = torch.argsort(source, stable=True)
    source_row_ptr = torch.searchsorted(
        source.index_select(0, source_order), boundaries
    )
    # Guard slots keep the exported edge axis dynamic even for zero or one
    # physical edge. CSR offsets exclude these slots from every node row.
    if edge_count < 2:
        padding = 2 - edge_count
        source = torch.cat((source, source.new_zeros(padding)))
        edge_vec = torch.cat((edge_vec, edge_vec.new_zeros((padding, 3))))
        source_order = torch.cat(
            (
                source_order,
                torch.arange(edge_count, 2, dtype=torch.int64, device=source.device),
            )
        )
    n_node = n_node.contiguous()
    return CanonicalGraph(
        n_node=n_node,
        n_local=n_node,
        source=source.to(canonical_index_dtype(node_count)),
        edge_vec=edge_vec,
        destination_row_ptr=destination_row_ptr,
        source_row_ptr=source_row_ptr,
        source_order=source_order,
    )


__all__ = [
    "INT32_MAX",
    "CanonicalGraph",
    "canonical_graph_from_neighbor_graph",
    "canonical_graph_from_pairs",
    "canonical_index_bytes",
    "canonical_index_dtype",
    "canonical_index_words",
    "validate_canonical_graph_shapes",
]
