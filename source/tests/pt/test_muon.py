# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

# NOTE: avoid torch thread reconfiguration errors during import.
import torch

torch_set_num_interop_threads = getattr(torch, "set_num_interop_threads", None)
torch_set_num_threads = getattr(torch, "set_num_threads", None)
if torch_set_num_interop_threads is not None:
    torch.set_num_interop_threads = lambda *args, **kwargs: None  # type: ignore[assignment]
if torch_set_num_threads is not None:
    torch.set_num_threads = lambda *args, **kwargs: None  # type: ignore[assignment]

from deepmd.pt.optimizer.muon import (
    MuonOptimizer,
    zeropower_via_newtonschulz5,
)
from deepmd.pt.utils import (
    env,
)


class TestNewtonSchulzOrthogonalization(unittest.TestCase):
    """Test Newton-Schulz orthogonalization algorithm."""

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


class TestMuonOptimizer(unittest.TestCase):
    """Test MuonOptimizer class."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_optimizer_step(self) -> None:
        """Test basic optimizer step."""
        torch.manual_seed(42)
        # Simple model with 2D and 1D parameters
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5, device=self.device),
        )

        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        # Dummy forward-backward pass
        x = torch.randn(4, 10, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Store initial params
        initial_params = [p.clone() for p in model.parameters()]

        # Optimizer step
        optimizer.step()

        # Verify parameters changed
        for i, (p, init_p) in enumerate(zip(model.parameters(), initial_params)):
            self.assertFalse(
                torch.allclose(p, init_p),
                f"Parameter {i} did not change after optimizer step",
            )

    def test_weight_decay(self) -> None:
        """Test weight decay application."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02, weight_decay=0.1)

        initial_weight_norm = model.weight.norm().item()

        # Multiple steps with gradients
        for _ in range(10):
            optimizer.zero_grad()
            x = torch.randn(4, 10, device=self.device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        # Weight norm should decrease due to weight decay
        final_weight_norm = model.weight.norm().item()
        self.assertLess(
            final_weight_norm,
            initial_weight_norm,
            "Weight norm should decrease with weight decay",
        )

    def test_muon_for_2d_adam_for_1d(self) -> None:
        """Test that Muon is applied to 2D params and Adam to 1D params."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        # Forward-backward
        x = torch.randn(4, 10, device=self.device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Check state - weight (2D) should have momentum_buffer
        weight_state = optimizer.state[model.weight]
        self.assertIn("momentum_buffer", weight_state)
        self.assertNotIn("exp_avg", weight_state)

        # Bias (1D) should have exp_avg and exp_avg_sq
        bias_state = optimizer.state[model.bias]
        self.assertIn("exp_avg", bias_state)
        self.assertIn("exp_avg_sq", bias_state)
        self.assertNotIn("momentum_buffer", bias_state)

    def test_closure(self) -> None:
        """Test optimizer with closure."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 5, device=self.device)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        def closure():
            optimizer.zero_grad()
            x = torch.randn(4, 10, device=self.device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertIsNotNone(loss)


class TestMuonOptimizerStateDict(unittest.TestCase):
    """Test optimizer state dict save/load."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_state_dict_save_load(self) -> None:
        """Test saving and loading optimizer state."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

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
        optimizer2 = MuonOptimizer(model.parameters(), lr=0.02)
        optimizer2.load_state_dict(state_dict)

        # Verify state matches
        for (_, s1), (_, s2) in zip(optimizer.state.items(), optimizer2.state.items()):
            for key in s1:
                if isinstance(s1[key], torch.Tensor):
                    self.assertTrue(
                        torch.allclose(s1[key], s2[key]),
                        f"State mismatch for key {key}",
                    )
                else:
                    self.assertEqual(s1[key], s2[key])


if __name__ == "__main__":
    unittest.main()
