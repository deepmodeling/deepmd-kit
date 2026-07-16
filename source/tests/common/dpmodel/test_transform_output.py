# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-preservation tests for fitting-output reductions."""

import pytest

from deepmd.dpmodel.model.edge_transform_output import (
    fit_output_to_model_output_graph,
)
from deepmd.dpmodel.model.transform_output import (
    fit_output_to_model_output,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)

torch = pytest.importorskip("torch")


def _output_def(*, intensive: bool) -> FittingOutputDef:
    """Build the smallest reducible fitting definition used by these tests."""
    return FittingOutputDef(
        [
            OutputVariableDef(
                name="energy",
                shape=[1],
                reducible=True,
                r_differentiable=False,
                c_differentiable=False,
                intensive=intensive,
            )
        ]
    )


def test_dense_torch_extensive_reduction_stays_on_backend() -> None:
    """Dense reduction must not rely on NumPy-style ``Tensor.astype``."""
    atomic = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
    coord = torch.zeros((1, 3, 3), dtype=torch.float32)

    result = fit_output_to_model_output(
        {"energy": atomic}, _output_def(intensive=False), coord
    )["energy_redu"]

    assert isinstance(result, torch.Tensor)
    assert result.dtype is torch.float64
    torch.testing.assert_close(result, torch.tensor([[6.0]], dtype=torch.float64))


def test_dense_torch_mask_count_uses_energy_dtype() -> None:
    """The intensive divisor must be reduced by Torch in energy precision."""
    atomic = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
    coord = torch.zeros((1, 3, 3), dtype=torch.float32)
    mask = torch.tensor([[True, True, False]])

    result = fit_output_to_model_output(
        {"energy": atomic}, _output_def(intensive=True), coord, mask=mask
    )["energy_redu"]

    assert isinstance(result, torch.Tensor)
    assert result.dtype is torch.float64
    torch.testing.assert_close(result, torch.tensor([[3.0]], dtype=torch.float64))


def test_dense_torch_no_mask_intensive_reduction_stays_on_backend() -> None:
    """Exercise the intensive reduction branch without a real-atom mask."""
    atomic = torch.tensor([[[1.0], [2.0], [4.0]]], dtype=torch.float32)
    coord = torch.zeros((1, 3, 3), dtype=torch.float32)

    result = fit_output_to_model_output(
        {"energy": atomic}, _output_def(intensive=True), coord
    )["energy_redu"]

    assert isinstance(result, torch.Tensor)
    assert result.dtype is torch.float64
    torch.testing.assert_close(result, torch.tensor([[7.0 / 3.0]], dtype=torch.float64))


def test_graph_torch_mask_count_uses_backend_dtype() -> None:
    """Graph reductions must translate the NumPy precision to ``torch.dtype``."""
    graph = NeighborGraph(
        n_node=torch.tensor([2, 1], dtype=torch.int64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        edge_vec=torch.empty((0, 3), dtype=torch.float32),
        edge_mask=torch.empty((0,), dtype=torch.bool),
    )
    atomic = torch.tensor([[1.0], [2.0], [6.0]], dtype=torch.float32)
    mask = torch.tensor([True, True, True])

    result = fit_output_to_model_output_graph(
        {"energy": atomic}, _output_def(intensive=True), graph, mask=mask
    )["energy_redu"]

    assert isinstance(result, torch.Tensor)
    assert result.dtype is torch.float64
    torch.testing.assert_close(
        result, torch.tensor([[1.5], [6.0]], dtype=torch.float64)
    )


def test_graph_torch_no_mask_intensive_reduction_stays_on_backend() -> None:
    """Exercise the graph intensive fallback using per-frame node counts."""
    graph = NeighborGraph(
        n_node=torch.tensor([2, 1], dtype=torch.int64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        edge_vec=torch.empty((0, 3), dtype=torch.float32),
        edge_mask=torch.empty((0,), dtype=torch.bool),
    )
    atomic = torch.tensor([[1.0], [3.0], [6.0]], dtype=torch.float32)

    result = fit_output_to_model_output_graph(
        {"energy": atomic}, _output_def(intensive=True), graph
    )["energy_redu"]

    assert isinstance(result, torch.Tensor)
    assert result.dtype is torch.float64
    torch.testing.assert_close(
        result, torch.tensor([[2.0], [6.0]], dtype=torch.float64)
    )


def test_graph_torch_extensive_reduction_stays_on_backend() -> None:
    """Exercise the graph extensive branch independently of intensive counts."""
    graph = NeighborGraph(
        n_node=torch.tensor([2, 1], dtype=torch.int64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        edge_vec=torch.empty((0, 3), dtype=torch.float32),
        edge_mask=torch.empty((0,), dtype=torch.bool),
    )
    atomic = torch.tensor([[1.0], [3.0], [6.0]], dtype=torch.float32)

    result = fit_output_to_model_output_graph(
        {"energy": atomic}, _output_def(intensive=False), graph
    )["energy_redu"]

    assert isinstance(result, torch.Tensor)
    assert result.dtype is torch.float64
    torch.testing.assert_close(
        result, torch.tensor([[4.0], [6.0]], dtype=torch.float64)
    )
