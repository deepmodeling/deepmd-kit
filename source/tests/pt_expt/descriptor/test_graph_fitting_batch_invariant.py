# SPDX-License-Identifier: LGPL-3.0-or-later
"""GPU regression for a fixed full-matrix fitting GEMM configuration.

Run this focused file in a fresh process with
``DP_GRAPH_FITTING_GEMM_POLICY=shape_stable``. The policy is process-immutable.
The optional library path supports testing an isolated C++ build without
replacing an installed Python environment.
"""

import importlib
import os
import subprocess
import sys
import unittest
from concurrent.futures import (
    ThreadPoolExecutor,
)
from pathlib import (
    Path,
)

import torch

if os.environ.get("DEEPMD_TEST_OP_LIBRARY"):
    torch.ops.load_library(os.environ["DEEPMD_TEST_OP_LIBRARY"])
else:
    importlib.import_module("deepmd.pt.cxx_op")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestShapeStablePolicyProcess(unittest.TestCase):
    """Exercise the opt-in path in ordinary GPU CI without changing its policy.

    The operator reads its policy only once. Other tests may already have used
    the legacy policy, so changing os.environ in this test process is unsafe.
    A child process also preserves the caller's Slurm device visibility.
    """

    def test_shape_stable_in_fresh_process(self):
        environment = dict(os.environ, DP_GRAPH_FITTING_GEMM_POLICY="shape_stable")
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "TestShapeStableFitting",
                "-v",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Ran 4 tests", result.stderr)
        self.assertNotIn("skipped", result.stderr)


@unittest.skipUnless(
    torch.cuda.is_available()
    and os.environ.get("DP_GRAPH_FITTING_GEMM_POLICY") == "shape_stable",
    "requires CUDA and a fresh shape_stable fitting process",
)
class TestShapeStableFitting(unittest.TestCase):
    """Check the entire feature matrix, residual backward, and stream ownership."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        generator = torch.Generator(device="cuda").manual_seed(20260917)
        self.x = torch.randn(1087, 144, generator=generator, device="cuda") * 0.2
        self.types = torch.arange(1087, device="cuda") % 6
        self.weights = [
            torch.randn(a, b, generator=generator, device="cuda") * 0.04
            for a, b in [(144, 192), (192, 192), (192, 192)]
        ]
        self.biases = [
            torch.randn(192, generator=generator, device="cuda") * 0.03
            for _ in range(3)
        ]
        self.head = torch.randn(192, generator=generator, device="cuda") * 0.04
        self.head_bias = torch.zeros(1, device="cuda")
        self.atomic_bias = torch.arange(6, dtype=torch.float64, device="cuda")

    def evaluate(self, x, types, activation=1):
        """Invoke the same fitting forward and reverse paths as canonical MD."""
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

    def test_batch_rows_and_backward_are_identical(self):
        for activation in (0, 1):
            single = self.evaluate(self.x[:268], self.types[:268], activation)
            for repetitions in (1, 2, 4, 8, 16):
                x = self.x.repeat(repetitions, 1)
                types = self.types.repeat(repetitions)
                full = self.evaluate(x, types, activation)
                for expected, actual in zip(single, full, strict=True):
                    torch.testing.assert_close(actual[:268], expected, atol=0, rtol=0)

    def test_independent_streams_and_host_threads(self):
        expected = self.evaluate(self.x, self.types)
        torch.cuda.synchronize()

        def worker(_):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                outputs = self.evaluate(self.x, self.types)
            stream.synchronize()
            return outputs

        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(worker, range(4)))
        for output in outputs:
            for reference, value in zip(expected, output, strict=True):
                torch.testing.assert_close(reference, value, atol=0, rtol=0)

    def test_minimum_aligned_feature_slice(self):
        # A contiguous view can start only four bytes into its allocation.
        # Pointer alignment must not change the selected reduction schedule.
        backing = torch.empty(self.x.numel() + 1, device="cuda", dtype=self.x.dtype)
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
        actual_energy, actual_gradient = self.evaluate(self.x[:19], self.types[:19])
        torch.testing.assert_close(
            actual_energy.flatten(), energy, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            actual_gradient.double(), gradient, atol=1e-6, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
