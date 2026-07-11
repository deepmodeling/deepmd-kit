# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact canonical graph contract for compressed DPA1 deployment."""

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


@dataclass(frozen=True)
class DPA1CanonicalGraph:
    """Store a cutoff-compacted destination-major graph without redundant fields.

    The physical edge count is the final value of either CSR row-pointer tensor.
    The storage edge axis has length ``max(E, 2)``; suffix entries are guards
    excluded from every CSR row.

    Parameters
    ----------
    n_node
        Per-frame total node counts with shape ``(nf,)``, int64.
    n_local
        Per-frame owned node counts with shape ``(nf,)``, int64.
    source
        Source-node index for each edge storage slot with shape ``(S,)``,
        int64.
    edge_vec
        Neighbor-minus-center vectors with shape ``(S, 3)``, float32.
    destination_row_ptr
        Destination CSR offsets with shape ``(N + 1,)``, int64.
    source_row_ptr
        Source CSR offsets with shape ``(N + 1,)``, int64.
    source_order
        Edge storage positions grouped by source with shape ``(S,)`` and the
        same dtype as ``source``.
    """

    n_node: torch.Tensor
    n_local: torch.Tensor
    source: torch.Tensor
    edge_vec: torch.Tensor
    destination_row_ptr: torch.Tensor
    source_row_ptr: torch.Tensor
    source_order: torch.Tensor


def validate_canonical_graph_shapes(
    graph: DPA1CanonicalGraph,
    node_count: int,
) -> None:
    """Validate shape, dtype, and device invariants without reading tensor data."""
    index_dtype = graph.source.dtype
    if index_dtype != torch.int64:
        raise ValueError("canonical graph source must be int64")
    if graph.source_order.dtype != torch.int64:
        raise ValueError("canonical graph source and source_order dtypes must match")
    if graph.edge_vec.dtype != torch.float32:
        raise ValueError("canonical graph edge_vec must be float32")
    if graph.n_node.dtype != torch.int64 or graph.n_local.dtype != torch.int64:
        raise ValueError("canonical graph node counts must be int64")
    if (
        graph.destination_row_ptr.dtype != torch.int64
        or graph.source_row_ptr.dtype != torch.int64
    ):
        raise ValueError("canonical graph row pointers must be int64")
    if graph.source.ndim != 1 or graph.source_order.shape != graph.source.shape:
        raise ValueError("canonical graph source arrays must share shape (S,)")
    if graph.edge_vec.shape != (graph.source.shape[0], 3):
        raise ValueError("canonical graph edge_vec must have shape (S, 3)")
    if graph.source.shape[0] < 2:
        raise ValueError("canonical graph edge storage must contain at least two slots")
    if graph.destination_row_ptr.shape != (node_count + 1,):
        raise ValueError("destination_row_ptr must have shape (N + 1,)")
    if graph.source_row_ptr.shape != (node_count + 1,):
        raise ValueError("source_row_ptr must have shape (N + 1,)")
    if graph.n_node.shape != graph.n_local.shape or graph.n_node.ndim != 1:
        raise ValueError("n_node and n_local must share shape (nf,)")
    devices = {
        graph.n_node.device,
        graph.n_local.device,
        graph.source.device,
        graph.edge_vec.device,
        graph.destination_row_ptr.device,
        graph.source_row_ptr.device,
        graph.source_order.device,
    }
    if len(devices) != 1:
        raise ValueError("canonical graph tensors must reside on one device")
    tensors = (
        graph.n_node,
        graph.n_local,
        graph.source,
        graph.edge_vec,
        graph.destination_row_ptr,
        graph.source_row_ptr,
        graph.source_order,
    )
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("canonical graph tensors must be contiguous")


def canonical_graph_from_neighbor_graph(
    graph: NeighborGraph,
) -> DPA1CanonicalGraph:
    """Convert a compact generic graph into the source-only deployment contract.

    Parameters
    ----------
    graph
        Destination-major graph whose physical edges form a valid prefix.

    Returns
    -------
    DPA1CanonicalGraph
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
        or graph.source_row_ptr is None
        or graph.source_order is None
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
        dtype=torch.int64,
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
            torch.int64
        )
        edge_vec[:physical_edge_count] = graph.edge_vec[:physical_edge_count].to(
            torch.float32
        )
        source_order[:physical_edge_count] = graph.source_order[
            :physical_edge_count
        ].to(torch.int64)

    result = DPA1CanonicalGraph(
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


__all__ = [
    "DPA1CanonicalGraph",
    "canonical_graph_from_neighbor_graph",
    "validate_canonical_graph_shapes",
]
