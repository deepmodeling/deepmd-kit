# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build CuTe SO2 metadata when the input does not carry graph-owned CSR."""

from __future__ import (
    annotations,
)

import torch


def _destination_row_ptr_impl(
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Build destination CSR where real storage offsets remain visible."""
    boundaries = torch.arange(
        n_nodes + 1,
        device=dst.device,
        dtype=dst.dtype,
    )
    return torch.searchsorted(
        dst.contiguous(),
        boundaries,
        out_int32=True,
    ).contiguous()


_destination_row_ptr_op = torch.library.custom_op(
    "sezm_cute::destination_row_ptr",
    mutates_args=(),
)(_destination_row_ptr_impl)


@_destination_row_ptr_op.register_fake
def _(
    dst: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    return torch.empty(
        (n_nodes + 1,),
        device=dst.device,
        dtype=torch.int32,
    )


def build_sorted_edge_index_metadata(
    src: torch.Tensor,
    dst: torch.Tensor,
    n_nodes: int,
    *,
    validate_sorted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build destination and source CSR metadata for one sorted edge list.

    Parameters
    ----------
    src : torch.Tensor
        Source indices with shape ``(E,)``.
    dst : torch.Tensor
        Nondecreasing destination indices with shape ``(E,)``.
    n_nodes : int
        Number of nodes addressed by the edge list.
    validate_sorted : bool, default: False
        Whether to assert the destination-order contract on the device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Destination row pointers, the source-major edge permutation, and source
        row pointers. All three tensors use int32 storage.

    Raises
    ------
    ValueError
        If the edge arrays have incompatible shapes or devices, or if their
        edge count exceeds int32 indexing.
    TypeError
        If either edge-index tensor is not int32 or int64.
    """
    if src.dim() != 1 or dst.dim() != 1:
        raise ValueError("src and dst must be one-dimensional")
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    if src.device != dst.device:
        raise ValueError("src and dst must be on the same device")
    if src.dtype not in (torch.int32, torch.int64):
        raise TypeError("src must have dtype int32 or int64")
    if dst.dtype not in (torch.int32, torch.int64):
        raise TypeError("dst must have dtype int32 or int64")
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    if src.numel() > 2**31 - 1:
        raise ValueError("sorted edge metadata requires E <= 2**31 - 1")

    src = src.contiguous()
    dst = dst.contiguous()
    if validate_sorted and dst.numel() > 1:
        torch._assert_async(
            torch.all(dst[1:] >= dst[:-1]),
            "Neo SO2 destinations_sorted=True requires monotonically "
            "nondecreasing destination indices",
        )
    destination_row_ptr = _destination_row_ptr_op(dst, n_nodes)

    source_order_i64 = torch.argsort(src, stable=True)
    sorted_src = src.index_select(0, source_order_i64)
    source_boundaries = torch.arange(
        n_nodes + 1,
        device=src.device,
        dtype=src.dtype,
    )
    source_row_ptr = torch.searchsorted(
        sorted_src,
        source_boundaries,
        out_int32=True,
    ).contiguous()
    source_order = source_order_i64.to(dtype=torch.int32).contiguous()
    return destination_row_ptr, source_order, source_row_ptr


__all__ = ["build_sorted_edge_index_metadata"]
