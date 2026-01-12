# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for AdaMuonOptimizer."""

import unittest

# NOTE: avoid torch thread reconfiguration errors during import.
import torch

torch_set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
torch_set_num_threads = getattr(torch, "set_num_threads", None)
if torch_set_num_interop_threads is not None:
    torch.set_num_interop_threads = lambda *args, **kwargs: None  # type: ignore[assignment]
if torch_set_num_threads is not None:
    torch.set_num_threads = lambda *args, **kwargs: None  # type: ignore[assignment]

from deepmd.pt.optimizer.adamuon import (
    AdaMuonOptimizer,
    zeropower_via_newtonschulz5,
)
from deepmd.pt.utils import (
    env,
)


class TestNewtonSchulzOrthogonalization(unittest.TestCase):
    """Test Newton-Schulz orthogonalization algorithm for AdaMuon."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_square_matrix_approximate_orthogonality(self) -> None:
        """Test that output is approximately orthogonal for square matrices."""
        torch.manual_seed(42)
        G = torch.randn(4, 4, dtype=torch.float32, device=self.device)
        X = zeropower_via_newtonschulz5(G, steps=5)

        # X @ X.T should be approximately identity (diagonal dominant)
        # Note: NS returns bf16, so use relaxed tolerance
        XXT = X.float() @ X.float().T
        # Check diagonal elements are close to 1 (relaxed tolerance for bf16 + 5 iterations)
        diag = torch.diag(XXT)
        self.assertTrue(
            torch.allclose(
                diag, torch.ones(4, dtype=torch.float32, device=self.device), atol=0.5
            ),
            f"Diagonal not close to 1: {diag}",
        )
        # Check off-diagonal elements are relatively small
        off_diag_norm = (XXT - torch.diag(diag)).norm()
        self.assertLess(
            off_diag_norm, 1.5, f"Off-diagonal norm too large: {off_diag_norm}"
        )

    def test_output_shape_preserved(self) -> None:
        """Test that output shape matches input shape and dtype is bf16."""
        torch.manual_seed(42)
        for shape in [(4, 4), (6, 4), (4, 6), (3, 4, 4)]:
            G = torch.randn(*shape, dtype=torch.float32, device=self.device)
            X = zeropower_via_newtonschulz5(G, steps=5)
            self.assertEqual(
                X.shape, G.shape, f"Shape mismatch for input shape {shape}"
            )
            self.assertEqual(
                X.dtype, torch.bfloat16, f"Output should be bf16, got {X.dtype}"
            )


class TestAdaMuonOptimizerBasic(unittest.TestCase):
    """Test AdaMuonOptimizer class basic functionality."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_optimizer_step_smoke(self) -> None:
        """Smoke test: step runs and updates both >=2D and 1D params."""
        torch.manual_seed(42)
        # Model with 2D weights and 1D biases/LayerNorm params
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20, bias=True, device=self.device),
            torch.nn.LayerNorm(20, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5, bias=True, device=self.device),
        )

        optimizer = AdaMuonOptimizer(model.parameters(), lr=0.02)

        # Dummy forward-backward pass
        x = torch.randn(4, 10, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Store initial params
        initial_params = [p.clone() for p in model.parameters()]

        # Optimizer step
        optimizer.step()

        # Verify all parameters with gradients changed
        for i, (p, init_p) in enumerate(zip(model.parameters(), initial_params)):
            if p.grad is not None:
                self.assertFalse(
                    torch.allclose(p, init_p),
                    f"Parameter {i} did not change after optimizer step",
                )

    def test_adamuon_for_2d_adam_for_1d(self) -> None:
        """Test that AdaMuon is applied to 2D params and Adam to 1D params."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = AdaMuonOptimizer(model.parameters(), lr=0.02)

        # Forward-backward
        x = torch.randn(4, 10, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Check state - weight (2D) should have momentum_buffer and v_buffer
        weight_state = optimizer.state[model.weight]
        self.assertIn("momentum_buffer", weight_state)
        self.assertNotIn("exp_avg", weight_state)

        # Bias (1D) should have exp_avg and exp_avg_sq
        bias_state = optimizer.state[model.bias]
        self.assertIn("exp_avg", bias_state)
        self.assertIn("exp_avg_sq", bias_state)
        self.assertNotIn("momentum_buffer", bias_state)


class TestAdaMuonOptimizerState(unittest.TestCase):
    """Test AdaMuonOptimizer state creation and management."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_2d_state_creation(self) -> None:
        """State creation test: verify momentum_buffer and v_buffer for 2D path."""
        torch.manual_seed(42)
        model = torch.nn.Linear(8, 16, bias=False, device=self.device)
        optimizer = AdaMuonOptimizer(model.parameters(), lr=0.02)

        # Forward-backward
        x = torch.randn(4, 8, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Check state for 2D weight
        weight_state = optimizer.state[model.weight]
        self.assertIn("momentum_buffer", weight_state)
        self.assertIn("v_buffer", weight_state)

        # momentum_buffer shape should match grad shape
        self.assertEqual(
            weight_state["momentum_buffer"].shape,
            model.weight.shape,
            "momentum_buffer shape should match weight shape",
        )

        # v_buffer numel should equal reshaped matrix numel (m * n)
        m, n = model.weight.shape[0], model.weight.numel() // model.weight.shape[0]
        self.assertEqual(
            weight_state["v_buffer"].numel(),
            m * n,
            "v_buffer numel should equal reshaped matrix numel",
        )

    def test_1d_state_fp32(self) -> None:
        """Verify 1D Adam path uses FP32 state tensors."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = AdaMuonOptimizer(model.parameters(), lr=0.02)

        # Forward-backward
        x = torch.randn(4, 10, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Bias state should be FP32
        bias_state = optimizer.state[model.bias]
        self.assertEqual(
            bias_state["exp_avg"].dtype,
            torch.float32,
            "exp_avg should be FP32",
        )
        self.assertEqual(
            bias_state["exp_avg_sq"].dtype,
            torch.float32,
            "exp_avg_sq should be FP32",
        )


class TestAdaMuonOptimizerBucketing(unittest.TestCase):
    """Test bucketed batch Newton-Schulz processing."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_bucketed_path(self) -> None:
        """Test that two layers with same weight shape land in the same bucket."""
        torch.manual_seed(42)
        # Create two Linear layers with SAME weight shape (32, 16)
        # Use parallel structure to avoid dimension mismatch
        layer1 = torch.nn.Linear(16, 32, bias=False, device=self.device)
        layer2 = torch.nn.Linear(16, 32, bias=False, device=self.device)

        optimizer = AdaMuonOptimizer([layer1.weight, layer2.weight], lr=0.02)

        # Store initial weights
        weight1_before = layer1.weight.clone()
        weight2_before = layer2.weight.clone()

        # Forward-backward with same input for both layers (parallel)
        x = torch.randn(4, 16, device=self.device)
        y = layer1(x) + layer2(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Both weights should have changed
        self.assertFalse(
            torch.allclose(layer1.weight, weight1_before),
            "Layer1 weight should change after optimizer step",
        )
        self.assertFalse(
            torch.allclose(layer2.weight, weight2_before),
            "Layer2 weight should change after optimizer step",
        )

        # Both should have v_buffer in state
        self.assertIn("v_buffer", optimizer.state[layer1.weight])
        self.assertIn("v_buffer", optimizer.state[layer2.weight])


class TestAdaMuonOptimizerLrAdjust(unittest.TestCase):
    """Test lr_adjust behavior."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_lr_adjust_modes_differ(self) -> None:
        """Test that lr_adjust <= 0 (match-RMS) and > 0 (rectangular) produce different updates."""
        torch.manual_seed(42)

        # Create two identical models
        model1 = torch.nn.Linear(16, 32, bias=False, device=self.device)
        model2 = torch.nn.Linear(16, 32, bias=False, device=self.device)
        model2.load_state_dict(model1.state_dict())

        # Two optimizers with different lr_adjust
        opt1 = AdaMuonOptimizer(
            model1.parameters(), lr=0.02, lr_adjust=-1.0
        )  # match-RMS
        opt2 = AdaMuonOptimizer(
            model2.parameters(), lr=0.02, lr_adjust=10.0
        )  # rectangular

        # Same input for both
        torch.manual_seed(123)
        x = torch.randn(4, 16, device=self.device)

        # Forward-backward for model1
        y1 = model1(x)
        loss1 = y1.sum()
        loss1.backward()
        opt1.step()

        # Reset seed and run for model2
        torch.manual_seed(123)
        x = torch.randn(4, 16, device=self.device)
        y2 = model2(x)
        loss2 = y2.sum()
        loss2.backward()
        opt2.step()

        # Updates should be different (not equal) due to different scaling
        self.assertFalse(
            torch.allclose(model1.weight, model2.weight),
            "Different lr_adjust modes should produce different updates",
        )


class TestAdaMuonOptimizerWeightDecay(unittest.TestCase):
    """Test weight decay application."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_weight_decay_only(self) -> None:
        """Test decoupled weight decay scales weights when gradients are zero."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, bias=False, device=self.device)
        optimizer = AdaMuonOptimizer(model.parameters(), lr=0.02, weight_decay=0.1)

        w_before = model.weight.detach().clone()

        # === Step 1. Make zero gradients ===
        model.weight.grad = torch.zeros_like(model.weight)

        # === Step 2. Step once ===
        optimizer.step()

        # === Step 3. Expect pure multiplicative decay: w <- (1 - lr*wd) * w ===
        expected = w_before * (1.0 - 0.02 * 0.1)
        self.assertTrue(
            torch.allclose(model.weight, expected),
            "Weight should be scaled by (1 - lr * weight_decay)",
        )


class TestAdaMuonOptimizerClosure(unittest.TestCase):
    """Test optimizer with closure."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_closure(self) -> None:
        """Test optimizer with closure returns loss."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 5, device=self.device)
        optimizer = AdaMuonOptimizer(model.parameters(), lr=0.02)

        def closure():
            optimizer.zero_grad()
            x = torch.randn(4, 10, device=self.device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertIsNotNone(loss)


class TestAdaMuonOptimizerStateDict(unittest.TestCase):
    """Test optimizer state dict save/load."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_state_dict_save_load(self) -> None:
        """Test saving and loading optimizer state."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = AdaMuonOptimizer(model.parameters(), lr=0.02)

        # Run a few steps to populate state
        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(4, 10, device=self.device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer and load state
        optimizer2 = AdaMuonOptimizer(model.parameters(), lr=0.02)
        optimizer2.load_state_dict(state_dict)

        # Verify state matches using param_groups as anchor
        params1 = list(optimizer.param_groups[0]["params"])
        params2 = list(optimizer2.param_groups[0]["params"])

        for p1, p2 in zip(params1, params2):
            s1 = optimizer.state[p1]
            s2 = optimizer2.state[p2]
            self.assertEqual(set(s1.keys()), set(s2.keys()))
            for key in s1:
                if isinstance(s1[key], torch.Tensor):
                    self.assertTrue(
                        torch.allclose(s1[key], s2[key]),
                        f"State mismatch for key {key}",
                    )
                else:
                    self.assertEqual(s1[key], s2[key], f"State mismatch for key {key}")


if __name__ == "__main__":
    unittest.main()
