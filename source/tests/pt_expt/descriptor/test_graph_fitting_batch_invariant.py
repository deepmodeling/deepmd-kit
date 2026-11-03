# SPDX-License-Identifier: LGPL-3.0-or-later
"""GPU regressions for full-batch fitting energy and gradient invariance.

The optional library path supports testing an isolated C++ build without
replacing an installed Python environment.
"""

import importlib
import os
import unittest
from concurrent.futures import (
    ThreadPoolExecutor,
)

import torch

if os.environ.get("DEEPMD_TEST_OP_LIBRARY"):
    torch.ops.load_library(os.environ["DEEPMD_TEST_OP_LIBRARY"])
else:
    importlib.import_module("deepmd.pt.cxx_op")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGraphFittingBatchInvariant(unittest.TestCase):
    """Check the entire feature matrix, residual backward, and stream ownership."""

    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=self.device).manual_seed(20260917)
        options = {
            "generator": generator,
            "device": self.device,
            "dtype": torch.float32,
        }
        self.x = torch.randn(1087, 144, **options) * 0.2
        self.types = torch.arange(1087, device=self.device, dtype=torch.int64) % 6
        self.weights = [
            torch.randn(a, b, **options) * 0.04
            for a, b in [(144, 192), (192, 192), (192, 192)]
        ]
        self.biases = [torch.randn(192, **options) * 0.03 for _ in range(3)]
        self.head = torch.randn(192, **options) * 0.04
        self.head_bias = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.atomic_bias = torch.arange(6, dtype=torch.float64, device=self.device)

    def evaluate(self, x, types, activation=1):
        """Evaluate the separate per-layer forward and backward operators."""
        energy, saved = torch.ops.deepmd.graph_fitting(
            x,
            types,
            self.weights,
            self.biases,
            [1, 1, 1],
            self.head,
            self.head_bias,
            self.atomic_bias,
            activation,
        )
        gradient = torch.ops.deepmd.graph_fitting_backward(
            torch.ones_like(energy),
            saved,
            self.weights,
            self.biases,
            [1, 1, 1],
            self.head,
            activation,
        )
        return energy, gradient

    def evaluate_fused(self, x, types, activation=1):
        """Evaluate the in-place energy/gradient route used by canonical inference."""
        gradient = x.clone()
        energy = torch.ops.deepmd.graph_fitting_energy_gradient(
            gradient,
            types,
            self.weights,
            self.biases,
            [1, 1, 1],
            self.head,
            self.head_bias,
            self.atomic_bias,
            activation,
            torch.ones((x.shape[0], 1), device=x.device, dtype=torch.float64),
            0,
        )
        return energy, gradient

    def test_batch_rows_and_backward_are_identical(self):
        for activation in (0, 1):
            single = self.evaluate(self.x, self.types, activation)
            for evaluate in (self.evaluate, self.evaluate_fused):
                for repetitions in (1, 2, 4, 8, 16):
                    with self.subTest(
                        operator=evaluate.__name__,
                        activation=activation,
                        repetitions=repetitions,
                    ):
                        x = self.x.repeat(repetitions, 1)
                        types = self.types.repeat(repetitions)
                        full = evaluate(x, types, activation)
                        for expected, actual in zip(single, full, strict=True):
                            # Check every repeated block, including the final
                            # partial GEMM tile, not only the leading rows.
                            torch.testing.assert_close(
                                actual.reshape(repetitions, *expected.shape),
                                expected.unsqueeze(0).expand(
                                    repetitions, *expected.shape
                                ),
                                atol=0,
                                rtol=0,
                            )

    def test_independent_streams_and_host_threads(self):
        expected = self.evaluate(self.x, self.types)
        torch.cuda.synchronize(self.device)

        for evaluate in (self.evaluate, self.evaluate_fused):

            def worker(_):
                stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(stream):
                    outputs = evaluate(self.x, self.types)
                stream.synchronize()
                return outputs

            with ThreadPoolExecutor(max_workers=2) as pool:
                outputs = list(pool.map(worker, range(4)))
            for output in outputs:
                for reference, value in zip(expected, output, strict=True):
                    torch.testing.assert_close(reference, value, atol=0, rtol=0)

    def test_concurrent_streams_on_one_host_thread(self):
        # Distinct inputs expose accidental sharing of intermediate storage.
        # Warm both streams before the concurrent submissions so first-use
        # initialization cannot serialize the work being checked.
        cases = [
            (self.x.repeat(4, 1), self.types.repeat(4)),
            ((self.x * -0.5 + 0.03).repeat(3, 1), self.types.repeat(3)),
        ]
        expected = [self.evaluate(x, types) for x, types in cases]
        torch.cuda.synchronize(self.device)
        streams = [torch.cuda.Stream(device=self.device) for _ in cases]
        for evaluate in (self.evaluate, self.evaluate_fused):
            for stream, (x, types) in zip(streams, cases, strict=True):
                with torch.cuda.stream(stream):
                    evaluate(x, types)
            for stream in streams:
                stream.synchronize()

            outputs = []
            for _ in range(3):
                for stream, (x, types), reference in zip(
                    streams, cases, expected, strict=True
                ):
                    with torch.cuda.stream(stream):
                        outputs.append((reference, evaluate(x, types)))
            for stream in streams:
                stream.synchronize()
            for reference, output in outputs:
                for expected_value, value in zip(reference, output, strict=True):
                    torch.testing.assert_close(expected_value, value, atol=0, rtol=0)

    def test_minimum_aligned_feature_slice(self):
        # A contiguous view can start only four bytes into its allocation.
        # Pointer alignment must not change the result of the reduction.
        backing = torch.empty(
            self.x.numel() + 1, device=self.device, dtype=self.x.dtype
        )
        shifted = backing[1:].view_as(self.x)
        shifted.copy_(self.x)
        self.assertEqual(shifted.data_ptr() % 16, 4)
        for expected, actual in zip(
            self.evaluate(self.x, self.types),
            self.evaluate(shifted, self.types),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_independent_double_precision_reference(self):
        # Independent autograd reference: do not obtain it from the candidate's
        # own backward implementation or weaken tolerances to hide a mismatch.
        x = self.x[:19].double().requires_grad_(True)
        current = x
        for weight, bias in zip(self.weights, self.biases, strict=True):
            value = torch.nn.functional.silu(current @ weight.double() + bias.double())
            if current.shape[1] == value.shape[1]:
                value = value + current
            current = value
        energy = current @ self.head.double() + self.atomic_bias[self.types[:19]]
        (gradient,) = torch.autograd.grad(energy.sum(), x)
        for evaluate in (self.evaluate, self.evaluate_fused):
            actual_energy, actual_gradient = evaluate(self.x[:19], self.types[:19])
            torch.testing.assert_close(
                actual_energy.flatten(), energy, atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(
                actual_gradient.double(), gradient, atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
