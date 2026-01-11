# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

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

    def test_orthogonalization(self) -> None:
        """Test that NS produces approximately orthogonal output."""
        torch.manual_seed(42)
        G = torch.randn(4, 4, dtype=torch.float32, device=self.device)
        X = zeropower_via_newtonschulz5(G)

        # X @ X.T should be approximately identity
        # Note: NS uses bf16 internally, 5 iterations gives ~0.1-0.3 error
        XXT = X.float() @ X.float().T
        diag = torch.diag(XXT)
        self.assertTrue(
            torch.allclose(
                diag, torch.ones(4, dtype=torch.float32, device=self.device), atol=0.5
            ),
            f"Diagonal not close to 1: {diag}",
        )
        off_diag_norm = (XXT - torch.diag(diag)).norm()
        self.assertLess(
            off_diag_norm, 1.5, f"Off-diagonal norm too large: {off_diag_norm}"
        )

    def test_shape_and_dtype(self) -> None:
        """Test that output preserves shape and returns bf16."""
        torch.manual_seed(42)
        for shape in [(4, 4), (6, 4), (3, 4, 4)]:
            G = torch.randn(*shape, dtype=torch.float32, device=self.device)
            X = zeropower_via_newtonschulz5(G)
            self.assertEqual(X.shape, G.shape)
            self.assertEqual(X.dtype, torch.bfloat16)

    def test_invalid_input(self) -> None:
        """Test that <2D input raises ValueError."""
        G_1d = torch.randn(10, dtype=torch.float32, device=self.device)
        with self.assertRaises(ValueError):
            zeropower_via_newtonschulz5(G_1d)


class TestMuonOptimizer(unittest.TestCase):
    """Test MuonOptimizer class."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_step(self) -> None:
        """Test basic optimizer step changes parameters."""
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5, device=self.device),
        )
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        x = torch.randn(4, 10, device=self.device)
        model(x).sum().backward()

        initial_params = [p.clone() for p in model.parameters()]
        optimizer.step()

        for i, (p, init_p) in enumerate(zip(model.parameters(), initial_params)):
            self.assertFalse(torch.allclose(p, init_p), f"Parameter {i} did not change")

    def test_weight_decay(self) -> None:
        """Test weight decay reduces parameter norm."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02, weight_decay=0.1)

        initial_norm = model.weight.norm().item()
        for _ in range(10):
            optimizer.zero_grad()
            x = torch.randn(4, 10, device=self.device)
            model(x).sum().backward()
            optimizer.step()

        self.assertLess(model.weight.norm().item(), initial_norm)

    def test_muon_adam_separation(self) -> None:
        """Test Muon for 2D params, Adam for 1D params."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        x = torch.randn(4, 10, device=self.device)
        model(x).sum().backward()
        optimizer.step()

        # 2D weight uses Muon (momentum_buffer)
        self.assertIn("momentum_buffer", optimizer.state[model.weight])
        self.assertNotIn("exp_avg", optimizer.state[model.weight])
        # 1D bias uses Adam (exp_avg, exp_avg_sq)
        self.assertIn("exp_avg", optimizer.state[model.bias])
        self.assertIn("exp_avg_sq", optimizer.state[model.bias])
        self.assertNotIn("momentum_buffer", optimizer.state[model.bias])

    def test_lr_adjust_modes(self) -> None:
        """Test lr_adjust modes: match-RMS (<=0) vs rectangular (>0)."""
        torch.manual_seed(42)

        model1 = torch.nn.Linear(10, 20, bias=False, device=self.device)
        model2 = torch.nn.Linear(10, 20, bias=False, device=self.device)
        model2.load_state_dict(model1.state_dict())

        opt1 = MuonOptimizer(model1.parameters(), lr=0.02, lr_adjust=0.0)
        opt2 = MuonOptimizer(model2.parameters(), lr=0.02, lr_adjust=10.0)

        x = torch.randn(4, 10, device=self.device)

        opt1.zero_grad()
        model1(x).sum().backward()
        opt1.step()

        opt2.zero_grad()
        model2(x).sum().backward()
        opt2.step()

        self.assertFalse(
            torch.allclose(model1.weight, model2.weight),
            "Different lr_adjust modes should produce different updates",
        )


class TestMuonOptimizerStateDict(unittest.TestCase):
    """Test optimizer state dict save/load."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_state_dict_save_load(self) -> None:
        """Test saving and loading optimizer state."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = MuonOptimizer(model.parameters(), lr=0.02)

        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(4, 10, device=self.device)
            model(x).sum().backward()
            optimizer.step()

        state_dict = optimizer.state_dict()

        optimizer2 = MuonOptimizer(model.parameters(), lr=0.02)
        optimizer2.load_state_dict(state_dict)

        # Verify state matches by param id, not iteration order
        for p in model.parameters():
            s1 = optimizer.state.get(p, {})
            s2 = optimizer2.state.get(p, {})
            self.assertEqual(len(s1), len(s2))
            for key in s1:
                if isinstance(s1[key], torch.Tensor):
                    self.assertTrue(torch.allclose(s1[key], s2[key]))
                else:
                    self.assertEqual(s1[key], s2[key])


if __name__ == "__main__":
    unittest.main()
