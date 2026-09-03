# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contracts for the in-place CuTe SO2 residual/addmm helper."""

from __future__ import (
    annotations,
)

import unittest

import torch
from torch.utils._python_dispatch import (
    TorchDispatchMode,
)

from deepmd.pt_expt.kernels.cute.sezm.so2 import linear as so2_linear


def _load_candidate():
    return so2_linear


def _out_of_place_reference(grad_out, residual, w0_t, wpair_t):
    edge_count = grad_out.shape[0]
    grad_flat = grad_out.view(edge_count, 2, 10 * 32)
    residual_flat = residual.view(edge_count, 2, 10 * 32)
    out = torch.empty_like(residual)
    out_flat = out.view(edge_count, 2, 10 * 32)
    for focus in range(2):
        torch.addmm(
            residual_flat[:, focus, : 4 * 32],
            grad_flat[:, focus, : 4 * 32],
            w0_t[focus],
            out=out_flat[:, focus, : 4 * 32],
        )
        torch.addmm(
            residual_flat[:, focus, 4 * 32 :],
            grad_flat[:, focus, 4 * 32 :],
            wpair_t[focus],
            out=out_flat[:, focus, 4 * 32 :],
        )
    return out


class _DispatchRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.calls.append((func, args, kwargs))
        return func(*args, **kwargs)


class TestSO2InplaceResidual(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260630)
        self.edge_count = 7
        self.grad_out = torch.randn(
            self.edge_count,
            2,
            10,
            32,
            dtype=torch.float32,
            device="cpu",
        )
        self.residual = torch.randn_like(self.grad_out)
        self.w0_t = torch.randn(
            2,
            4 * 32,
            4 * 32,
            dtype=torch.float32,
            device="cpu",
        )
        self.wpair_t = torch.randn(
            2,
            6 * 32,
            6 * 32,
            dtype=torch.float32,
            device="cpu",
        )

    def test_inplace_result_matches_out_of_place_equations(self):
        candidate = _load_candidate()
        expected = _out_of_place_reference(
            self.grad_out,
            self.residual,
            self.w0_t,
            self.wpair_t,
        )
        residual = self.residual.clone()
        storage_ptr = residual.untyped_storage().data_ptr()

        actual = candidate.neo_so2_linear_backward_residual_inplace(
            residual,
            self.grad_out,
            self.w0_t,
            self.wpair_t,
        )

        self.assertIs(actual, residual)
        self.assertEqual(actual.untyped_storage().data_ptr(), storage_ptr)
        self.assertEqual(actual.stride(), (2 * 10 * 32, 10 * 32, 32, 1))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_only_four_inplace_addmm_ops_cross_the_dispatch_boundary(self):
        candidate = _load_candidate()
        residual = self.residual.clone()
        recorder = _DispatchRecorder()

        with recorder:
            candidate.neo_so2_linear_backward_residual_inplace(
                residual,
                self.grad_out,
                self.w0_t,
                self.wpair_t,
            )

        addmm_calls = [
            args
            for func, args, _kwargs in recorder.calls
            if func is torch.ops.aten.addmm_.default
        ]
        self.assertEqual(len(addmm_calls), 4)
        forbidden = {
            torch.ops.aten.clone.default,
            torch.ops.aten.contiguous.default,
            torch.ops.aten.copy_.default,
        }
        self.assertFalse(
            any(func in forbidden for func, _args, _kwargs in recorder.calls)
        )

        for residual_block, grad_block, weight in addmm_calls:
            width = residual_block.shape[1]
            self.assertIn(width, (4 * 32, 6 * 32))
            self.assertEqual(residual_block.stride(), (2 * 10 * 32, 1))
            self.assertEqual(grad_block.stride(), (2 * 10 * 32, 1))
            self.assertEqual(weight.stride(), (width, 1))
            self.assertTrue(candidate._has_direct_cublas_layout(residual_block))
            self.assertTrue(candidate._has_direct_cublas_layout(grad_block))
            self.assertTrue(candidate._has_direct_cublas_layout(weight))

    def test_layout_predicate_and_fp32_scope_are_explicit(self):
        candidate = _load_candidate()

        self.assertFalse(hasattr(candidate, "is_cublas_compatible_matrix"))
        self.assertIn("PyTorch 2.10", candidate._has_direct_cublas_layout.__doc__)
        self.assertIn("dtype/device", candidate._has_direct_cublas_layout.__doc__)
        self.assertIn("highest", candidate.__doc__)
        self.assertIn("TF32", candidate.__doc__)

    def test_meta_execution_preserves_outer_shape_and_stride_contract(self):
        candidate = _load_candidate()
        residual = torch.empty(11, 2, 10, 32, dtype=torch.float32, device="meta")
        grad_out = torch.empty_like(residual)
        w0_t = torch.empty(2, 4 * 32, 4 * 32, dtype=torch.float32, device="meta")
        wpair_t = torch.empty(
            2,
            6 * 32,
            6 * 32,
            dtype=torch.float32,
            device="meta",
        )

        result = candidate.neo_so2_linear_backward_residual_inplace(
            residual,
            grad_out,
            w0_t,
            wpair_t,
        )

        self.assertIs(result, residual)
        self.assertEqual(result.shape, (11, 2, 10, 32))
        self.assertEqual(result.stride(), (2 * 10 * 32, 10 * 32, 32, 1))

    def test_aliasing_grad_out_is_rejected_for_final_layer_safety(self):
        candidate = _load_candidate()
        shared = self.residual.clone()

        with self.assertRaisesRegex(ValueError, "must not alias"):
            candidate.neo_so2_linear_backward_residual_inplace(
                shared,
                shared,
                self.w0_t,
                self.wpair_t,
            )

    def test_exact_cross_storage_grad_out_alias_is_rejected(self):
        candidate = _load_candidate()
        byte_count = self.residual.numel() * self.residual.element_size()
        backing = bytearray(byte_count)
        residual = torch.frombuffer(
            backing,
            dtype=torch.float32,
            count=self.residual.numel(),
        ).view_as(self.residual)
        grad_out = torch.frombuffer(
            backing,
            dtype=torch.float32,
            count=self.grad_out.numel(),
        ).view_as(self.grad_out)
        self.assertNotEqual(
            residual.untyped_storage()._cdata,
            grad_out.untyped_storage()._cdata,
        )
        self.assertEqual(residual.data_ptr(), grad_out.data_ptr())
        self.assertFalse(torch._C._overlaps(residual, grad_out))

        with self.assertRaisesRegex(
            ValueError,
            "residual and grad_out must not alias",
        ):
            candidate.neo_so2_linear_backward_residual_inplace(
                residual,
                grad_out,
                self.w0_t,
                self.wpair_t,
            )

    def test_residual_overlapping_w0_is_rejected(self):
        candidate = _load_candidate()
        storage = torch.randn(
            2 * 4 * 32 * 4 * 32,
            dtype=torch.float32,
            device="cpu",
        )
        residual = storage[: self.residual.numel()].view_as(self.residual)
        w0_t = storage.view(2, 4 * 32, 4 * 32)
        self.assertTrue(residual.is_contiguous())
        self.assertTrue(w0_t.is_contiguous())
        self.assertTrue(torch._C._overlaps(residual, w0_t))

        with self.assertRaisesRegex(ValueError, "residual and w0_t must not alias"):
            candidate.neo_so2_linear_backward_residual_inplace(
                residual,
                self.grad_out,
                w0_t,
                self.wpair_t,
            )

    def test_partial_cross_storage_w0_alias_is_rejected(self):
        candidate = _load_candidate()
        overlap_elements = 17
        weight_elements = self.w0_t.numel()
        weight_offset = self.residual.numel() - overlap_elements
        backing = bytearray(
            (weight_offset + weight_elements) * self.residual.element_size()
        )
        residual = torch.frombuffer(
            backing,
            dtype=torch.float32,
            count=self.residual.numel(),
        ).view_as(self.residual)
        w0_t = torch.frombuffer(
            backing,
            dtype=torch.float32,
            count=weight_elements,
            offset=weight_offset * self.residual.element_size(),
        ).view_as(self.w0_t)
        self.assertNotEqual(
            residual.untyped_storage()._cdata,
            w0_t.untyped_storage()._cdata,
        )
        self.assertNotEqual(residual.data_ptr(), w0_t.data_ptr())
        self.assertFalse(torch._C._overlaps(residual, w0_t))

        with self.assertRaisesRegex(ValueError, "residual and w0_t must not alias"):
            candidate.neo_so2_linear_backward_residual_inplace(
                residual,
                self.grad_out,
                w0_t,
                self.wpair_t,
            )

    def test_residual_overlapping_wpair_is_rejected(self):
        candidate = _load_candidate()
        storage = torch.randn(
            2 * 6 * 32 * 6 * 32,
            dtype=torch.float32,
            device="cpu",
        )
        residual = storage[: self.residual.numel()].view_as(self.residual)
        wpair_t = storage.view(2, 6 * 32, 6 * 32)
        self.assertTrue(residual.is_contiguous())
        self.assertTrue(wpair_t.is_contiguous())
        self.assertTrue(torch._C._overlaps(residual, wpair_t))

        with self.assertRaisesRegex(ValueError, "residual and wpair_t must not alias"):
            candidate.neo_so2_linear_backward_residual_inplace(
                residual,
                self.grad_out,
                self.w0_t,
                wpair_t,
            )

    def test_partial_cross_storage_wpair_alias_is_rejected(self):
        candidate = _load_candidate()
        overlap_elements = 17
        weight_elements = self.wpair_t.numel()
        weight_offset = self.residual.numel() - overlap_elements
        backing = bytearray(
            (weight_offset + weight_elements) * self.residual.element_size()
        )
        residual = torch.frombuffer(
            backing,
            dtype=torch.float32,
            count=self.residual.numel(),
        ).view_as(self.residual)
        wpair_t = torch.frombuffer(
            backing,
            dtype=torch.float32,
            count=weight_elements,
            offset=weight_offset * self.residual.element_size(),
        ).view_as(self.wpair_t)
        self.assertNotEqual(
            residual.untyped_storage()._cdata,
            wpair_t.untyped_storage()._cdata,
        )
        self.assertNotEqual(residual.data_ptr(), wpair_t.data_ptr())
        self.assertFalse(torch._C._overlaps(residual, wpair_t))

        with self.assertRaisesRegex(
            ValueError,
            "residual and wpair_t must not alias",
        ):
            candidate.neo_so2_linear_backward_residual_inplace(
                residual,
                self.grad_out,
                self.w0_t,
                wpair_t,
            )

    def test_non_fp32_and_noncanonical_layouts_are_rejected(self):
        candidate = _load_candidate()
        with self.assertRaisesRegex(TypeError, "float32"):
            candidate.neo_so2_linear_backward_residual_inplace(
                self.residual.double(),
                self.grad_out.double(),
                self.w0_t.double(),
                self.wpair_t.double(),
            )

        residual = torch.empty(
            2,
            self.edge_count,
            10,
            32,
            dtype=torch.float32,
            device="cpu",
        ).transpose(0, 1)
        self.assertEqual(residual.shape, self.residual.shape)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            candidate.neo_so2_linear_backward_residual_inplace(
                residual,
                self.grad_out,
                self.w0_t,
                self.wpair_t,
            )


if __name__ == "__main__":
    unittest.main()
