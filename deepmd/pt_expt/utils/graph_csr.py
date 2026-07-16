# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch validation for graph-lower CSR export inputs."""

from __future__ import (
    annotations,
)

import torch


def _validate_row_ptr(
    name: str,
    row_ptr: torch.Tensor,
    node_count: int,
    edge_count: int,
    device: torch.device,
) -> None:
    """Validate the bounds and layout of one CSR row-pointer tensor."""
    if row_ptr.dtype != torch.int64 or row_ptr.shape != (node_count + 1,):
        raise ValueError(
            f"graph export requires {name} with shape (N + 1,) and dtype int64"
        )
    if (
        row_ptr.device != device
        or not bool(torch.all(row_ptr[1:] >= row_ptr[:-1]))
        or int(row_ptr[0]) != 0
        or int(row_ptr[-1]) > edge_count
    ):
        raise ValueError(f"graph export received invalid {name}")


def _validate_permutation_csr(
    name: str,
    order: torch.Tensor,
    row_ptr: torch.Tensor,
    endpoint: torch.Tensor,
    edge_mask: torch.Tensor,
    node_count: int,
    edge_count: int,
    device: torch.device,
) -> None:
    """Validate that a CSR order covers every active edge in its endpoint row."""
    if order.dtype not in (torch.int32, torch.int64) or order.shape != (edge_count,):
        raise ValueError(
            f"graph export requires {name} with shape (E,) and integer dtype"
        )
    if order.device != device:
        raise ValueError(f"graph export requires {name} on the edge device")
    expected = torch.arange(edge_count, dtype=order.dtype, device=device)
    if not torch.equal(torch.sort(order).values, expected):
        raise ValueError(f"graph export requires {name} to be an edge permutation")

    active_edge = torch.arange(
        edge_count,
        dtype=torch.int64,
        device=device,
    )[edge_mask]
    active_endpoint = endpoint[active_edge]
    if active_endpoint.numel() and (
        int(active_endpoint.min()) < 0 or int(active_endpoint.max()) >= node_count
    ):
        raise ValueError("graph export received an active edge outside node bounds")

    inverse = torch.empty(edge_count, dtype=torch.int64, device=device)
    inverse.scatter_(
        0,
        order.to(torch.int64),
        torch.arange(edge_count, dtype=torch.int64, device=device),
    )
    active_position = inverse[active_edge]
    row_begin = row_ptr[active_endpoint]
    row_end = row_ptr[active_endpoint + 1]
    if not bool(
        torch.all((row_begin <= active_position) & (active_position < row_end))
    ):
        raise ValueError(f"graph export received {name} entries outside their CSR rows")


def validate_graph_csr_for_export(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    destination_order: torch.Tensor,
    destination_row_ptr: torch.Tensor,
    source_order: torch.Tensor,
    source_row_ptr: torch.Tensor,
    node_count: int,
    *,
    destination_sorted: bool,
) -> None:
    """Validate concrete graph CSR inputs before graph-form export tracing.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge endpoints with shape ``(2, E)`` and integer dtype.
    edge_mask : torch.Tensor
        Valid-edge mask with shape ``(E,)`` and dtype bool.
    destination_order, source_order : torch.Tensor
        Edge permutations with shape ``(E,)``.
    destination_row_ptr, source_row_ptr : torch.Tensor
        CSR offsets with shape ``(N + 1,)`` and dtype int64.
    node_count : int
        Number of nodes on the flat graph axis.
    destination_sorted : bool
        Whether the exported descriptor addresses destination rows directly.

    Raises
    ------
    ValueError
        If an order is not a permutation, a CSR row omits an active edge, or a
        canonical export does not use the identity destination order.

    Notes
    -----
    This function validates concrete trace inputs only. It must execute before
    ``make_fx`` or ``torch.export`` so it does not add inference operations.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("graph export requires edge_index with shape (2, E)")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("graph export requires integer edge_index")
    edge_count = edge_index.shape[1]
    if (
        edge_mask.shape != (edge_count,)
        or edge_mask.dtype != torch.bool
        or edge_mask.device != edge_index.device
    ):
        raise ValueError("graph export requires a bool edge_mask with shape (E,)")

    _validate_row_ptr(
        "destination_row_ptr",
        destination_row_ptr,
        node_count,
        edge_count,
        edge_index.device,
    )
    _validate_row_ptr(
        "source_row_ptr",
        source_row_ptr,
        node_count,
        edge_count,
        edge_index.device,
    )
    _validate_permutation_csr(
        "destination_order",
        destination_order,
        destination_row_ptr,
        edge_index[1],
        edge_mask,
        node_count,
        edge_count,
        edge_index.device,
    )
    _validate_permutation_csr(
        "source_order",
        source_order,
        source_row_ptr,
        edge_index[0],
        edge_mask,
        node_count,
        edge_count,
        edge_index.device,
    )
    if destination_sorted:
        expected = torch.arange(
            edge_count,
            dtype=destination_order.dtype,
            device=destination_order.device,
        )
        if not torch.equal(destination_order, expected):
            raise ValueError(
                "destination_sorted=True requires an identity destination_order"
            )
