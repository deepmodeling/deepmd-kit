# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for PyTorch graph-lower CSR export validation."""

import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    build_edge_csr,
)
from deepmd.pt_expt.utils.graph_csr import (
    validate_graph_csr_for_export,
)


def _csr_inputs(canonicalize: bool) -> tuple[torch.Tensor, ...]:
    edge_index = torch.tensor(
        [[1, 2, 0, 0], [0, 0, 1, 0]],
        dtype=torch.int64,
    )
    edge_vec = torch.zeros(4, 3, dtype=torch.float64)
    edge_mask = torch.tensor([True, True, True, False])
    (
        edge_index,
        _edge_vec,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
    ) = build_edge_csr(
        edge_index,
        edge_vec,
        edge_mask,
        n_nodes=3,
        canonicalize=canonicalize,
    )
    return (
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
    )


def test_accepts_canonical_csr() -> None:
    inputs = _csr_inputs(canonicalize=True)

    validate_graph_csr_for_export(
        *inputs,
        node_count=3,
        destination_sorted=True,
    )


def test_accepts_permutation_csr() -> None:
    inputs = _csr_inputs(canonicalize=False)

    validate_graph_csr_for_export(
        *inputs,
        node_count=3,
        destination_sorted=False,
    )


def test_rejects_nonidentity_canonical_order() -> None:
    (
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
    ) = _csr_inputs(canonicalize=True)
    destination_order = destination_order.clone()
    destination_order[:2] = torch.tensor([1, 0], dtype=torch.int64)

    with pytest.raises(ValueError, match="identity destination_order"):
        validate_graph_csr_for_export(
            edge_index,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            node_count=3,
            destination_sorted=True,
        )


def test_rejects_active_edge_outside_its_csr_row() -> None:
    (
        edge_index,
        edge_mask,
        destination_order,
        destination_row_ptr,
        source_order,
        source_row_ptr,
    ) = _csr_inputs(canonicalize=True)
    destination_row_ptr = destination_row_ptr.clone()
    destination_row_ptr[1] = 1

    with pytest.raises(ValueError, match="destination_order entries outside"):
        validate_graph_csr_for_export(
            edge_index,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_order,
            source_row_ptr,
            node_count=3,
            destination_sorted=True,
        )
