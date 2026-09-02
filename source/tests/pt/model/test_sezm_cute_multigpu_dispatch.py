# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU/mock coverage for operand-device-aware Neo CuTe dispatch."""

from __future__ import (
    annotations,
)

import importlib
import os
from types import (
    SimpleNamespace,
)
from unittest import (
    mock,
)

import pytest
import torch

from deepmd.pt.model.network import (
    mlp,
)
from deepmd.pt_expt.kernels.cute.sezm import (
    k1,
    runtime_policy,
)


def _cuda_operand(index: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cuda", index),
        is_cuda=True,
    )


def test_k1_custom_ops_are_direct_and_thin_path_uses_operand_device() -> None:
    operand = _cuda_operand()
    sentinel = object()

    with mock.patch.object(k1, "_cute_k1_impl", return_value=sentinel) as direct:
        actual = k1.cute_k1(
            1,
            operand,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    assert actual is sentinel
    direct.assert_called_once()

    context = object()
    with mock.patch.object(
        k1,
        "_k1_packed_direct_registered_backward_impl",
        return_value=sentinel,
    ) as backward:
        actual = k1._k1_packed_direct_backward(context, operand, None)

    assert actual is sentinel
    backward.assert_called_once_with(context, operand)

    with (
        mock.patch.object(
            k1,
            "_cuda_compute_capability",
            return_value=(9, 0),
        ) as capability,
        mock.patch.object(
            runtime_policy,
            "is_k1_thin_wrapper_enabled",
            return_value=False,
        ) as thin_selector,
        mock.patch.object(
            k1,
            "_maybe_run_cute_k1_fallback",
            return_value=sentinel,
        ) as fallback,
    ):
        actual = k1.maybe_run_cute_k1(object(), operand, object(), object())

    assert actual is sentinel
    capability.assert_called_once_with(1)
    thin_selector.assert_called_once_with((9, 0))
    fallback.assert_called_once()


def test_k1_prepare_uses_requested_device_instead_of_current_device() -> None:
    device = torch.device("cuda", 1)
    with (
        mock.patch.object(
            k1,
            "_cuda_compute_capability",
            return_value=(8, 6),
        ) as capability,
        mock.patch.object(
            k1,
            "is_supported_k1_compute_capability",
            return_value=False,
        ) as supported,
        mock.patch.object(
            torch.cuda,
            "current_device",
            side_effect=AssertionError("must not read the current CUDA device"),
        ),
    ):
        assert not k1.prepare_cute_k1_blocks(
            [object()],
            training=False,
            device=device,
            dtype=torch.float32,
        )

    capability.assert_called_once_with(1)
    supported.assert_called_once_with((8, 6))


def test_k4_custom_ops_are_called_directly() -> None:
    pytest.importorskip("cutlass.cute")
    k4_wignerd = importlib.import_module("deepmd.pt_expt.kernels.cute.sezm.k4_wignerd")
    operand = _cuda_operand()
    sentinel = object()

    with mock.patch.object(
        k4_wignerd,
        "_run_cute_wignerd_impl",
        return_value=sentinel,
    ) as forward:
        actual = k4_wignerd.run_cute_wignerd(operand, object())

    assert actual is sentinel
    forward.assert_called_once_with(operand, mock.ANY, packed_wigner=False)

    context = SimpleNamespace(saved_tensors=(operand,))
    grad_panel = object()
    with mock.patch.object(
        k4_wignerd,
        "_wignerd_panel_registered_backward_impl",
        return_value=sentinel,
    ) as backward:
        actual = k4_wignerd._wignerd_panel_backward(context, grad_panel)

    assert actual is sentinel
    backward.assert_called_once_with(context, grad_panel)


def test_mlp_selector_and_forward_use_input_device() -> None:
    cuda1 = torch.device("cuda", 1)

    def capability(device: torch.device | None = None) -> tuple[int, int]:
        return (8, 0) if device == cuda1 else (9, 0)

    with (
        mock.patch.dict(
            os.environ,
            {"DP_CUTE_INFER": "1"},
            clear=True,
        ),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=capability,
        ) as get_capability,
    ):
        assert mlp._use_k1_compile_visible_linear(cuda1)
        assert not mlp._use_k1_compile_visible_linear(torch.device("cuda", 0))
        assert not mlp._use_k1_compile_visible_linear(torch.device("cpu"))

    assert get_capability.call_args_list == [
        mock.call(cuda1),
        mock.call(torch.device("cuda", 0)),
    ]

    layer = mlp.MLPLayer(
        4,
        4,
        activation_function="none",
        precision="float32",
    ).to("cpu")
    layer.eval()
    mlp.enable_neo_cute_compile_visible_linears(layer)
    value = torch.randn(3, 4, device="cpu")
    with mock.patch.object(
        mlp,
        "_use_k1_compile_visible_linear",
        return_value=False,
    ) as selector:
        layer(value)
    selector.assert_called_once_with(value.device)


def test_nv_neighbor_list_selector_and_build_use_coordinate_device() -> None:
    sezm_model = importlib.import_module("deepmd.pt.model.model.sezm_model")
    cuda1 = torch.device("cuda", 1)

    def capability(device: torch.device | None = None) -> tuple[int, int]:
        return (8, 0) if device == cuda1 else (9, 0)

    with (
        mock.patch.dict(
            os.environ,
            {"DP_CUTE_INFER": "1"},
            clear=True,
        ),
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=capability,
        ) as get_capability,
    ):
        assert sezm_model._neo_cute_nlist_eager_island_enabled(cuda1)
        assert not sezm_model._neo_cute_nlist_eager_island_enabled(
            torch.device("cuda", 0)
        )
        assert not sezm_model._neo_cute_nlist_eager_island_enabled(torch.device("cpu"))

    assert get_capability.call_args_list == [
        mock.call(cuda1),
        mock.call(torch.device("cuda", 0)),
    ]

    builder = sezm_model.NvNeighborList()
    sentinel = object()
    with (
        mock.patch.object(
            sezm_model,
            "_neo_cute_nlist_eager_island_enabled",
            return_value=False,
        ) as selector,
        mock.patch.object(builder, "build", return_value=sentinel) as build,
    ):
        actual = sezm_model._build_neo_neighbor_list(
            builder,
            mock.Mock(device=cuda1),
            object(),
            None,
            4.0,
            [32],
            return_mode="edges",
        )

    assert actual is sentinel
    selector.assert_called_once_with(cuda1)
    build.assert_called_once()

    with (
        mock.patch.object(
            sezm_model,
            "_neo_cute_nlist_eager_island_enabled",
            return_value=True,
        ),
        mock.patch.object(
            sezm_model,
            "_build_neo_neighbor_list_eager_island",
            return_value=sentinel,
        ) as eager,
    ):
        actual = sezm_model._build_neo_neighbor_list(
            builder,
            mock.Mock(device=cuda1),
            object(),
            None,
            4.0,
            [32],
            return_mode="edges",
        )

    assert actual is sentinel
    eager.assert_called_once()
