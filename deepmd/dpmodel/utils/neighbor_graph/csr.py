# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-agnostic CSR topology helpers for :class:`NeighborGraph`."""

from __future__ import (
    annotations,
)

from dataclasses import (
    replace,
)
from typing import (
    TYPE_CHECKING,
)

import array_api_compat

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )

    from .graph import (
        NeighborGraph,
    )


def build_edge_csr(
    edge_index: Array,
    edge_vec: Array,
    edge_mask: Array,
    n_nodes: int,
    canonicalize: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Build destination/source CSR views of an edge payload.

    By default, both views store permutations into the original edge stream.
    With ``canonicalize=True``, a stable destination permutation is applied to
    every edge field, masked entries move to the suffix, and
    ``destination_order`` becomes the identity. Stable ordering preserves the
    incoming order within each destination segment.

    Parameters
    ----------
    edge_index : Array
        Edge endpoints with shape ``(2, E)`` in ``[source, destination]`` order.
    edge_vec : Array
        Edge vectors with shape ``(E, 3)``.
    edge_mask : Array
        Real-edge mask with shape ``(E,)``.
    n_nodes : int
        Number of nodes in the flat graph.
    canonicalize : bool, default: False
        Whether to reorder the payload into destination-major form.

    Returns
    -------
    edge_index : Array
        Edge endpoints with shape ``(2, E)``.
    edge_vec : Array
        Edge vectors with shape ``(E, 3)``.
    edge_mask : Array
        Real-edge mask.
    destination_order : Array
        Edge indices grouped by destination with shape ``(E,)`` and the same
        dtype as ``edge_index``.
    destination_row_ptr : Array
        Destination CSR offsets with shape ``(N + 1,)`` and dtype int64.
    source_order : Array
        Edge indices grouped by source with shape ``(E,)`` and the same dtype
        as ``edge_index``.
    source_row_ptr : Array
        Source CSR offsets with shape ``(N + 1,)`` and dtype int64.

    Raises
    ------
    OverflowError
        If an int32 edge payload has more than ``2**31 - 1`` entries.
    """
    xp = array_api_compat.array_namespace(edge_index)
    device = array_api_compat.device(edge_index)
    edge_count = edge_index.shape[1]
    if (
        edge_index.dtype == xp.int32
        and isinstance(edge_count, int)
        and edge_count > 2**31 - 1
    ):
        raise OverflowError(
            "int32 edge payload cannot represent a permutation of more than "
            "2**31 - 1 edges"
        )
    padding_node = xp.asarray(n_nodes, dtype=edge_index.dtype, device=device)

    destination_key = xp.where(edge_mask, edge_index[1], padding_node)
    destination_order = xp.argsort(destination_key, stable=True)
    ordered_destination = xp.take(destination_key, destination_order, axis=0)
    if canonicalize:
        edge_index = xp.take(edge_index, destination_order, axis=1)
        edge_vec = xp.take(edge_vec, destination_order, axis=0)
        edge_mask = xp.take(edge_mask, destination_order, axis=0)
        destination_order = xp.arange(
            edge_index.shape[1], dtype=edge_index.dtype, device=device
        )
    else:
        destination_order = xp.astype(destination_order, edge_index.dtype)
    node_boundaries = xp.arange(
        n_nodes + 1,
        dtype=edge_index.dtype,
        device=device,
    )
    destination_row_ptr = xp.astype(
        xp.searchsorted(ordered_destination, node_boundaries, side="left"),
        xp.int64,
    )

    source_key = xp.where(edge_mask, edge_index[0], padding_node)
    source_order = xp.argsort(source_key, stable=True)
    ordered_source = xp.take(source_key, source_order, axis=0)
    source_order = xp.astype(source_order, edge_index.dtype)
    source_row_ptr = xp.astype(
        xp.searchsorted(ordered_source, node_boundaries, side="left"),
        xp.int64,
    )
    return (
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
    )


def attach_edge_csr(
    graph: NeighborGraph,
    n_nodes: int,
    canonicalize: bool = False,
) -> NeighborGraph:
    """Attach destination/source CSR views to an edge graph.

    Parameters
    ----------
    graph : NeighborGraph
        The graph whose current edge payload and mask define the CSR views.
    n_nodes : int
        Number of nodes on the flat graph axis.
    canonicalize : bool, default: False
        Whether to reorder the payload into destination-major form.

    Returns
    -------
    NeighborGraph
        A copy carrying CSR views consistent with its edge payload and mask.

    Raises
    ------
    ValueError
        If canonicalization is requested for a graph with angle indices.
    """
    if canonicalize and graph.angle_index is not None:
        raise ValueError("cannot canonicalize a graph with angle indices")
    (
        edge_index,
        edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
    ) = build_edge_csr(
        graph.edge_index,
        graph.edge_vec,
        graph.edge_mask,
        n_nodes,
        canonicalize=canonicalize,
    )
    return replace(
        graph,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
        destination_order=destination_order,
        destination_row_ptr=destination_row_ptr,
        source_order=source_order,
        source_row_ptr=source_row_ptr,
        destination_sorted=canonicalize,
    )


def canonicalize_neighbor_graph(
    graph: NeighborGraph,
    n_nodes: int,
) -> NeighborGraph:
    """Return a destination-major edge graph.

    Generic graph builders preserve the incoming edge order. Deployment
    adapters call this function when their ABI guarantees a destination-major
    payload and identity destination permutation.

    Parameters
    ----------
    graph : NeighborGraph
        The graph to canonicalize.
    n_nodes : int
        Number of nodes on the flat graph axis.

    Returns
    -------
    NeighborGraph
        A graph with every edge field reordered consistently and
        ``destination_sorted=True``.

    Raises
    ------
    ValueError
        If the graph contains angle indices, which would require a matching
        edge-index remapping.
    """
    return attach_edge_csr(graph, n_nodes, canonicalize=True)
