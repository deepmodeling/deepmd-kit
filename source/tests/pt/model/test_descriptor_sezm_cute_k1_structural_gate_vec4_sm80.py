# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contracts for the shared structural-gate vec4 path."""

from __future__ import (
    annotations,
)

import ast
import importlib
import math
import random
import struct
import unittest
from pathlib import (
    Path,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CUTE_ROOT = REPO_ROOT / "deepmd/kernels/cute/neo"
KERNEL_PATH = CUTE_ROOT / "k1_kernels/cute_neo_gate_split_structural_vec4_sm80.py"
DYNAMIC_EDGE_COUNTS = (0, 1, 7, 8, 9, 31, 32, 33)


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _sigmoid_f32(value: float) -> float:
    value = _f32(value)
    return _f32(_f32(1.0) / _f32(_f32(1.0) + _f32(math.exp(_f32(-value)))))


def _mul_add_f32(lhs: float, rhs: float, addend: float) -> float:
    return _f32(_f32(lhs * rhs) + addend)


def _reference_forward(
    residual: list[float],
    y: list[float],
    logits: list[float],
    edge_count: int,
) -> list[float]:
    out = residual.copy()
    for edge in range(edge_count):
        for focus in range(2):
            row = edge * 2 + focus
            row_base = row * 10 * 32
            logits_base = (focus * edge_count + edge) * 3 * 32
            gates = [
                [
                    _sigmoid_f32(logits[logits_base + gate * 32 + channel])
                    for channel in range(32)
                ]
                for gate in range(3)
            ]
            for channel in range(32):
                index = row_base + channel
                y0 = y[index]
                out[index] = _mul_add_f32(y0, _sigmoid_f32(y0), out[index])
            for degree in range(1, 10):
                gate = gates[(degree - 1) % 3]
                for channel in range(32):
                    index = row_base + degree * 32 + channel
                    out[index] = _mul_add_f32(
                        y[index],
                        gate[channel],
                        out[index],
                    )
    return out


def _vec4_forward_inplace_model(
    out: list[float],
    y: list[float],
    logits: list[float],
    edge_count: int,
) -> None:
    rows = edge_count * 2
    for block_row in range((rows + 15) // 16):
        for row_slot in range(16):
            row = block_row * 16 + row_slot
            if row >= rows:
                continue
            edge, focus = divmod(row, 2)
            row_base = row * 10 * 32
            logits_base = (focus * edge_count + edge) * 3 * 32
            for channel_group in range(8):
                channel_base = channel_group * 4
                for lane in range(4):
                    index = row_base + channel_base + lane
                    y0 = y[index]
                    out[index] = _mul_add_f32(y0, _sigmoid_f32(y0), out[index])
                for gate_index in range(3):
                    gates = [
                        _sigmoid_f32(
                            logits[logits_base + gate_index * 32 + channel_base + lane]
                        )
                        for lane in range(4)
                    ]
                    for repeat in range(3):
                        degree = 1 + gate_index + repeat * 3
                        for lane in range(4):
                            index = row_base + degree * 32 + channel_base + lane
                            out[index] = _mul_add_f32(
                                y[index],
                                gates[lane],
                                out[index],
                            )


def _backward_reference(
    grad_out: list[float],
    y: list[float],
    logits: list[float],
    edge_count: int,
) -> tuple[list[float], list[float]]:
    grad_y = [0.0] * len(y)
    grad_logits = [0.0] * len(logits)
    for edge in range(edge_count):
        for focus in range(2):
            row = edge * 2 + focus
            row_base = row * 10 * 32
            logits_base = (focus * edge_count + edge) * 3 * 32
            for channel in range(32):
                y0 = _f32(y[row_base + channel])
                sig0 = _sigmoid_f32(y0)
                grad0 = _f32(grad_out[row_base + channel])
                inner = _f32(_f32(1.0) + _f32(y0 * _f32(_f32(1.0) - sig0)))
                grad_y[row_base + channel] = _f32(_f32(grad0 * sig0) * inner)
                for gate_index in range(3):
                    gate_offset = logits_base + gate_index * 32 + channel
                    gate = _sigmoid_f32(logits[gate_offset])
                    grad_logit = _f32(0.0)
                    for repeat in range(3):
                        degree = 1 + gate_index + repeat * 3
                        index = row_base + degree * 32 + channel
                        gout = _f32(grad_out[index])
                        grad_y[index] = _f32(gout * gate)
                        term = _f32(gout * _f32(y[index]))
                        term = _f32(term * gate)
                        term = _f32(term * _f32(_f32(1.0) - gate))
                        grad_logit = _f32(grad_logit + term)
                    grad_logits[gate_offset] = grad_logit
    return grad_y, grad_logits


def _vec4_backward_model(
    grad_out: list[float],
    y: list[float],
    logits: list[float],
    edge_count: int,
) -> tuple[list[float], list[float]]:
    grad_y = [0.0] * len(y)
    grad_logits = [0.0] * len(logits)
    rows = edge_count * 2
    for block_row in range((rows + 15) // 16):
        for row_slot in range(16):
            row = block_row * 16 + row_slot
            if row >= rows:
                continue
            edge, focus = divmod(row, 2)
            row_base = row * 10 * 32
            logits_base = (focus * edge_count + edge) * 3 * 32
            for channel_group in range(8):
                channel_base = channel_group * 4
                for lane in range(4):
                    channel = channel_base + lane
                    y0 = _f32(y[row_base + channel])
                    sig0 = _sigmoid_f32(y0)
                    grad0 = _f32(grad_out[row_base + channel])
                    inner = _f32(_f32(1.0) + _f32(y0 * _f32(_f32(1.0) - sig0)))
                    grad_y[row_base + channel] = _f32(_f32(grad0 * sig0) * inner)
                for gate_index in range(3):
                    for lane in range(4):
                        channel = channel_base + lane
                        gate_offset = logits_base + gate_index * 32 + channel
                        gate = _sigmoid_f32(logits[gate_offset])
                        grad_logit = _f32(0.0)
                        for repeat in range(3):
                            degree = 1 + gate_index + repeat * 3
                            index = row_base + degree * 32 + channel
                            gout = _f32(grad_out[index])
                            grad_y[index] = _f32(gout * gate)
                            term = _f32(gout * _f32(y[index]))
                            term = _f32(term * gate)
                            term = _f32(term * _f32(_f32(1.0) - gate))
                            grad_logit = _f32(grad_logit + term)
                        grad_logits[gate_offset] = grad_logit
    return grad_y, grad_logits


def _module_constants(tree: ast.Module) -> dict[str, object]:
    def constant_value(node: ast.expr, namespace: dict[str, object]) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return namespace[node.id]
        if isinstance(node, ast.BinOp):
            lhs = constant_value(node.left, namespace)
            rhs = constant_value(node.right, namespace)
            if isinstance(node.op, ast.FloorDiv):
                return lhs // rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
        raise ValueError("not a supported module constant")

    constants: dict[str, object] = {}
    namespace: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = constant_value(node.value, namespace)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        constants[target.id] = value
        namespace[target.id] = value
    return constants


def _load_wrapper_module():
    return importlib.import_module("deepmd.kernels.cute.neo.k1_gate_structural")


class TestSM80StructuralGateVec4Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel_source = KERNEL_PATH.read_text()
        cls.constants = _module_constants(ast.parse(cls.kernel_source))

    def test_vector_layout_and_launch_are_fixed(self):
        self.assertEqual(self.constants["FOCUS_COUNT"], 2)
        self.assertEqual(self.constants["REDUCED_COUNT"], 10)
        self.assertEqual(self.constants["CHANNELS"], 32)
        self.assertEqual(self.constants["VECTOR_WIDTH"], 4)
        self.assertEqual(self.constants["CHANNEL_GROUPS"], 8)
        self.assertEqual(self.constants["ROWS_PER_BLOCK"], 16)
        self.assertEqual(self.constants["THREADS"], 128)
        self.assertIn("residual.element_type.width * VECTOR_WIDTH", self.kernel_source)
        self.assertIn("cute.make_tiled_copy_tv", self.kernel_source)
        self.assertIn("(1, CHANNEL_GROUPS)", self.kernel_source)
        self.assertIn("(1, VECTOR_WIDTH)", self.kernel_source)

    def test_kernel_is_strict_fp32_edge_major_only(self):
        for forbidden in (
            "Float16",
            "BFloat16",
            "TensorFloat32",
        ):
            self.assertNotIn(forbidden, self.kernel_source)
        self.assertIn("cutlass.Float32", self.kernel_source)
        self.assertIn('"assumed_align": 16', self.kernel_source)
        self.assertIn("_guard_vec4_dispatch", self.kernel_source)
        self.assertIn("runtime_policy.SUPPORTED_K1_CAPABILITIES", self.kernel_source)


class TestSM80StructuralGateVec4Arithmetic(unittest.TestCase):
    def test_forward_dynamic_edges_and_block_tails(self):
        for edge_count in DYNAMIC_EDGE_COUNTS:
            with self.subTest(edge_count=edge_count):
                rng = random.Random(20260721 + edge_count)
                values = edge_count * 2 * 10 * 32
                residual = [_f32(rng.uniform(-2.0, 2.0)) for _ in range(values)]
                y = [_f32(rng.uniform(-2.0, 2.0)) for _ in range(values)]
                logits = [
                    _f32(rng.uniform(-4.0, 4.0)) for _ in range(2 * edge_count * 3 * 32)
                ]
                expected = _reference_forward(residual, y, logits, edge_count)
                actual = residual.copy()
                _vec4_forward_inplace_model(actual, y, logits, edge_count)
                self.assertEqual(actual, expected)

    def test_backward_dynamic_edges_and_block_tails(self):
        for edge_count in DYNAMIC_EDGE_COUNTS:
            with self.subTest(edge_count=edge_count):
                rng = random.Random(20260722 + edge_count)
                values = edge_count * 2 * 10 * 32
                grad_out = [_f32(rng.uniform(-2.0, 2.0)) for _ in range(values)]
                y = [_f32(rng.uniform(-2.0, 2.0)) for _ in range(values)]
                logits = [
                    _f32(rng.uniform(-4.0, 4.0)) for _ in range(2 * edge_count * 3 * 32)
                ]
                expected_grad_y, expected_grad_logits = _backward_reference(
                    grad_out,
                    y,
                    logits,
                    edge_count,
                )
                actual_grad_y, actual_grad_logits = _vec4_backward_model(
                    grad_out,
                    y,
                    logits,
                    edge_count,
                )
                self.assertEqual(actual_grad_y, expected_grad_y)
                self.assertEqual(actual_grad_logits, expected_grad_logits)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class TestSM80StructuralGateVec4Dispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dispatch = staticmethod(
            _load_wrapper_module()._dispatch_aligned_vec4_kernel
        )

    def test_aligned_fp32_tensor_dispatches(self):
        calls = []
        tensor = torch.empty(16, dtype=torch.float32, device="cpu")

        def kernel(value):
            calls.append(value)
            return "dispatched"

        result = self.dispatch(kernel, ("value",), tensor)
        self.assertEqual(result, "dispatched")
        self.assertEqual(calls, [tensor])

    def test_misaligned_storage_offset_is_rejected_before_dispatch(self):
        calls = []
        tensor = torch.empty(17, dtype=torch.float32, device="cpu")[1:]
        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.storage_offset(), 1)

        with self.assertRaisesRegex(ValueError, "storage offsets divisible by 4"):
            self.dispatch(lambda value: calls.append(value), ("value",), tensor)
        self.assertEqual(calls, [])

    def test_misaligned_pointer_is_rejected_before_dispatch(self):
        class MisalignedTensor:
            dtype = torch.float32
            shape = (4,)

            @staticmethod
            def numel():
                return 4

            @staticmethod
            def is_contiguous():
                return True

            @staticmethod
            def stride():
                return (1,)

            @staticmethod
            def storage_offset():
                return 0

            @staticmethod
            def data_ptr():
                return 4

        calls = []
        with self.assertRaisesRegex(ValueError, "16-byte-aligned"):
            self.dispatch(
                lambda value: calls.append(value),
                ("value",),
                MisalignedTensor(),
            )
        self.assertEqual(calls, [])

    def test_noncompact_and_non_fp32_tensors_are_rejected(self):
        noncompact = torch.empty(4, 8, dtype=torch.float32, device="cpu").T
        with self.assertRaisesRegex(ValueError, "compact tensors"):
            self.dispatch(lambda value: value, ("value",), noncompact)

        with self.assertRaisesRegex(TypeError, "requires float32"):
            self.dispatch(
                lambda value: value,
                ("value",),
                torch.empty(4, dtype=torch.float64, device="cpu"),
            )

    def test_empty_tensor_does_not_launch(self):
        calls = []
        result = self.dispatch(
            lambda value: calls.append(value),
            ("value",),
            torch.empty(0, dtype=torch.float32, device="cpu"),
        )
        self.assertIsNone(result)
        self.assertEqual(calls, [])


@unittest.skipUnless(
    torch is not None
    and torch.cuda.is_available()
    and tuple(torch.cuda.get_device_capability())
    in {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)},
    "A supported Neo K1 CUDA runtime is required",
)
class TestSM80StructuralGateVec4CudaDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from deepmd.kernels.cute.neo.k1_kernels.cute_neo_gate_split_structural_vec4_sm80 import (
            compile_neo_gate_split_structural_vec4_sm80_backward,
            compile_neo_gate_split_structural_vec4_sm80_forward,
        )

        capability = tuple(torch.cuda.get_device_capability())
        compile_identity = (torch.cuda.current_device(), *capability)
        cls.vec4_forward = staticmethod(
            compile_neo_gate_split_structural_vec4_sm80_forward(compile_identity)
        )
        cls.vec4_backward = staticmethod(
            compile_neo_gate_split_structural_vec4_sm80_backward(compile_identity)
        )

    def test_forward_dynamic_edges_tails_and_empty(self):
        for edge_count in DYNAMIC_EDGE_COUNTS:
            with self.subTest(edge_count=edge_count):
                torch.manual_seed(20260721 + edge_count)
                residual = torch.randn(
                    edge_count,
                    2,
                    10,
                    32,
                    device="cuda",
                    dtype=torch.float32,
                )
                y = torch.randn_like(residual)
                logits = torch.randn(
                    2,
                    edge_count,
                    3 * 32,
                    device="cuda",
                    dtype=torch.float32,
                )
                gates = torch.sigmoid(
                    logits.permute(1, 0, 2).reshape(edge_count, 2, 3, 32)
                )
                expected = residual.clone()
                expected[:, :, 0, :].add_(torch.nn.functional.silu(y[:, :, 0, :]))
                for degree in range(1, 10):
                    expected[:, :, degree, :].add_(
                        y[:, :, degree, :] * gates[:, :, (degree - 1) % 3, :]
                    )

                vec4 = residual.clone()
                self.vec4_forward(
                    vec4.view(edge_count * 2, 10 * 32),
                    y.view(edge_count * 2, 10 * 32),
                    logits,
                    vec4.view(edge_count * 2, 10 * 32),
                )
                torch.testing.assert_close(vec4, expected, atol=5e-5, rtol=5e-5)

    def test_backward_dynamic_edges_tails_and_empty(self):
        gate_indices = torch.tensor(
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            device="cuda",
        )
        for edge_count in DYNAMIC_EDGE_COUNTS:
            with self.subTest(edge_count=edge_count):
                torch.manual_seed(20260722 + edge_count)
                grad_out = torch.randn(
                    edge_count,
                    2,
                    10,
                    32,
                    device="cuda",
                    dtype=torch.float32,
                )
                y = torch.randn_like(grad_out)
                logits = torch.randn(
                    2,
                    edge_count,
                    3 * 32,
                    device="cuda",
                    dtype=torch.float32,
                )
                vec4_grad_y = torch.empty_like(y)
                vec4_grad_logits = torch.empty_like(logits)
                self.vec4_backward(
                    grad_out.view(edge_count * 2, 10 * 32),
                    y.view(edge_count * 2, 10 * 32),
                    logits,
                    vec4_grad_y.view(edge_count * 2, 10 * 32),
                    vec4_grad_logits,
                )

                y_reference = y.detach().clone().requires_grad_(True)
                logits_reference = logits.detach().clone().requires_grad_(True)
                gates = torch.sigmoid(
                    logits_reference.permute(1, 0, 2).reshape(
                        edge_count,
                        2,
                        3,
                        32,
                    )
                )
                output = torch.cat(
                    (
                        torch.nn.functional.silu(y_reference[:, :, :1, :]),
                        y_reference[:, :, 1:, :] * gates.index_select(2, gate_indices),
                    ),
                    dim=2,
                )
                expected_grad_y, expected_grad_logits = torch.autograd.grad(
                    output,
                    (y_reference, logits_reference),
                    grad_out,
                )
                torch.testing.assert_close(
                    vec4_grad_y,
                    expected_grad_y,
                    atol=5e-5,
                    rtol=5e-5,
                )
                torch.testing.assert_close(
                    vec4_grad_logits,
                    expected_grad_logits,
                    atol=5e-5,
                    rtol=5e-5,
                )

    def test_misaligned_cuda_view_is_rejected(self):
        edge_count = 1
        storage = torch.empty(
            edge_count * 2 * 10 * 32 + 1,
            device="cuda",
            dtype=torch.float32,
        )
        residual = storage[1:].view(edge_count * 2, 10 * 32)
        y = torch.empty_like(residual)
        logits = torch.empty(
            2,
            edge_count,
            3 * 32,
            device="cuda",
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "storage offsets divisible by 4"):
            self.vec4_forward(residual, y, logits, residual)


if __name__ == "__main__":
    unittest.main()
