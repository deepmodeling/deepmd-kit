# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differential tests for the opt-in CuTe geometric initial embedding."""

from __future__ import (
    annotations,
)

import unittest
from types import (
    SimpleNamespace,
)

import torch

from deepmd.kernels.cute.neo import gie as gie_module


def _load_gie_module():
    return gie_module


def _degree_slots(lmax: int, *, device: torch.device) -> torch.Tensor:
    degrees = torch.arange(1, lmax + 1, device=device, dtype=torch.long)
    return torch.repeat_interleave(degrees - 1, 2 * degrees + 1)


def _zonal_indices(
    lmax: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(1, (lmax + 1) ** 2, device=device, dtype=torch.long)
    degrees = torch.arange(1, lmax + 1, device=device, dtype=torch.long)
    degree_for_row = torch.repeat_interleave(degrees, 2 * degrees + 1)
    return rows, degree_for_row * (degree_for_row + 1)


def _materialized_reference(
    radial: torch.Tensor,
    zonal: torch.Tensor,
    inv_sqrt_deg: torch.Tensor,
    dst: torch.Tensor,
    gate: torch.Tensor,
    *,
    n_nodes: int,
    lmax: int,
) -> torch.Tensor:
    slots = _degree_slots(lmax, device=radial.device)
    message = zonal.unsqueeze(-1) * radial.index_select(1, slots)
    if gate.numel() != 0:
        message = message * gate.reshape(-1, 1, 1)
    non_scalar = radial.new_zeros(n_nodes, zonal.shape[1], radial.shape[2])
    non_scalar.index_add_(0, dst, message)
    out = radial.new_zeros(n_nodes, zonal.shape[1] + 1, radial.shape[2])
    out[:, 1:, :] = non_scalar
    return out * inv_sqrt_deg


def _inputs(
    *,
    n_nodes: int,
    dst_values: tuple[int, ...],
    lmax: int,
    channels: int,
    device: torch.device,
    with_gate: bool,
) -> tuple[torch.Tensor, ...]:
    edge_count = len(dst_values)
    generator = torch.Generator(device=device).manual_seed(20260703 + edge_count)
    radial = torch.randn(
        edge_count,
        lmax,
        channels,
        generator=generator,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    dense_dt = torch.randn(
        edge_count,
        (lmax + 1) ** 2,
        (lmax + 1) ** 2,
        generator=generator,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    rows, cols = _zonal_indices(lmax, device=device)
    zonal = dense_dt[:, rows, cols]
    inv_sqrt_deg = (
        torch.rand(
            n_nodes,
            1,
            1,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        + 0.25
    ).requires_grad_(True)
    dst = torch.tensor(dst_values, device=device, dtype=torch.long)
    if with_gate:
        gate = torch.rand(
            edge_count,
            1,
            generator=generator,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
    else:
        gate = torch.empty(0, device=device, dtype=torch.float32)
    return radial, dense_dt, zonal, inv_sqrt_deg, dst, gate


class TestSeZMCuTeGIEContract(unittest.TestCase):
    def test_backward_compile_key_separates_destination_dtypes(self):
        gie = _load_gie_module()
        common = ((0, 8, 0), 3, 32, True, (128, 32, 1))

        int32_key = gie._backward_compile_key(*common, torch.int32)
        int64_key = gie._backward_compile_key(*common, torch.int64)

        self.assertNotEqual(int32_key, int64_key)

    def test_contract_requires_sorted_dynamic_strict_fp32_inputs(self):
        gie = _load_gie_module()
        radial, _dense_dt, zonal, inv_sqrt_deg, dst, gate = _inputs(
            n_nodes=5,
            dst_values=(0, 0, 1, 3, 3, 4),
            lmax=3,
            channels=4,
            device=torch.device("cpu"),
            with_gate=True,
        )
        module = SimpleNamespace(
            lmax=3,
            channels=4,
            training=False,
            non_scalar_row_index=torch.arange(1, 16, device=torch.device("cpu")),
            radial_slot_index_for_row=_degree_slots(3, device=torch.device("cpu")),
        )
        cache = SimpleNamespace(
            dst=dst,
            inv_sqrt_deg=inv_sqrt_deg,
            edge_src_gate=gate,
            destinations_sorted=True,
        )
        self.assertTrue(gie.validate_gie_contract(module, 5, cache, radial, zonal))

        cache.destinations_sorted = False
        self.assertFalse(gie.validate_gie_contract(module, 5, cache, radial, zonal))
        cache.destinations_sorted = True
        self.assertFalse(
            gie.validate_gie_contract(module, 5, cache, radial.double(), zonal)
        )
        module.non_scalar_row_index = torch.arange(15, device=torch.device("cpu"))
        self.assertFalse(gie.validate_gie_contract(module, 5, cache, radial, zonal))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_forward_and_wigner_radial_degree_gate_gradients(self):
        gie = _load_gie_module()
        if not gie.SEZM_CUTE_GIE_AVAILABLE:
            self.skipTest("CuTe DSL is not available")
        for n_nodes, dst_values, with_gate in (
            (4, (0, 0, 2, 2, 2, 3), False),
            (7, (0, 1, 1, 1, 4, 6, 6, 6, 6), True),
        ):
            with self.subTest(n_nodes=n_nodes, edges=len(dst_values), gate=with_gate):
                expected_inputs = _inputs(
                    n_nodes=n_nodes,
                    dst_values=dst_values,
                    lmax=3,
                    channels=32,
                    device=torch.device("cuda"),
                    with_gate=with_gate,
                )
                radial, dense_dt, zonal, inv_sqrt_deg, dst, gate = expected_inputs
                weight = torch.randn_like(
                    radial.new_empty(n_nodes, 16, radial.shape[2])
                )
                expected = _materialized_reference(
                    radial,
                    zonal,
                    inv_sqrt_deg,
                    dst,
                    gate,
                    n_nodes=n_nodes,
                    lmax=3,
                )
                expected_grads = torch.autograd.grad(
                    (expected * weight).sum(),
                    (radial, dense_dt, inv_sqrt_deg, *([gate] if with_gate else [])),
                )

                actual_inputs = _inputs(
                    n_nodes=n_nodes,
                    dst_values=dst_values,
                    lmax=3,
                    channels=32,
                    device=torch.device("cuda"),
                    with_gate=with_gate,
                )
                radial, dense_dt, zonal, inv_sqrt_deg, dst, gate = actual_inputs
                actual = gie.gie_fused_cuda(
                    radial,
                    zonal,
                    inv_sqrt_deg,
                    dst,
                    gate,
                    n_nodes=n_nodes,
                    lmax=3,
                )
                actual_grads = torch.autograd.grad(
                    (actual * weight).sum(),
                    (radial, dense_dt, inv_sqrt_deg, *([gate] if with_gate else [])),
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                for actual_grad, expected_grad in zip(
                    actual_grads, expected_grads, strict=True
                ):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=2e-5, atol=2e-5
                    )


if __name__ == "__main__":
    unittest.main()
