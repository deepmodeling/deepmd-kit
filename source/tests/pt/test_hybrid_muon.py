# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.optimizer.hybrid_muon import (
    TRITON_AVAILABLE,
    HybridMuonOptimizer,
    _newton_schulz_orth,
)
from deepmd.pt.utils import (
    env,
)


def _bf16_matmul_supported(device: torch.device) -> bool:
    """Check if bf16 matmul is reliably supported on the given device."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        # bf16 requires compute capability >= 8.0 (Ampere+) for native support
        # or >= 7.0 (Volta) with tensor cores, but may have precision issues
        if hasattr(torch.cuda, "is_bf16_supported"):
            return torch.cuda.is_bf16_supported()
        # Fallback: check compute capability directly
        cap = torch.cuda.get_device_capability(device)
        return cap[0] >= 8
    # CPU bf16 support: available on x86 with AVX-512 BF16 or ARM with BF16 extension
    # Since it's hard to detect reliably, try a small matmul and check for errors
    try:
        a = torch.randn(4, 4, dtype=torch.bfloat16, device=device)
        _ = torch.mm(a, a.T)
        return True
    except (RuntimeError, TypeError):
        return False


BF16_SUPPORTED = _bf16_matmul_supported(env.DEVICE)


@unittest.skipIf(not BF16_SUPPORTED, "bf16 matmul not supported on this device")
class TestNewtonSchulzOrthogonalization(unittest.TestCase):
    """Test Newton-Schulz orthogonalization algorithm."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_orthogonalization(self) -> None:
        """Test that NS produces approximately orthogonal output."""
        torch.manual_seed(42)
        G = torch.randn(4, 4, dtype=torch.float32, device=self.device)
        X = _newton_schulz_orth(G)

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
        for shape in [(4, 4), (6, 4)]:
            G = torch.randn(*shape, dtype=torch.float32, device=self.device)
            X = _newton_schulz_orth(G)
            self.assertEqual(X.shape, G.shape)
            self.assertEqual(X.dtype, torch.bfloat16)

    def test_invalid_input(self) -> None:
        """Test that 1D input raises error."""
        G_1d = torch.randn(10, dtype=torch.float32, device=self.device)
        with self.assertRaises((ValueError, RuntimeError, IndexError)):
            _newton_schulz_orth(G_1d)


@unittest.skipIf(not BF16_SUPPORTED, "bf16 matmul not supported on this device")
class TestHybridMuonOptimizer(unittest.TestCase):
    """Test HybridMuonOptimizer class."""

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
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02)

        x = torch.randn(4, 10, device=self.device)
        model(x).sum().backward()

        initial_params = [p.clone() for p in model.parameters()]
        optimizer.step()

        for i, (p, init_p) in enumerate(
            zip(model.parameters(), initial_params, strict=True)
        ):
            self.assertFalse(torch.allclose(p, init_p), f"Parameter {i} did not change")

    def test_weight_decay(self) -> None:
        """Test weight decay reduces parameter norm."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02, weight_decay=0.1)

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
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02)

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

    def test_muon_adam_fallback_small_2d(self) -> None:
        """Test Adam fallback for small 2D matrices when min_2d_dim is set."""
        torch.manual_seed(42)
        linear_small = torch.nn.Linear(10, 1, bias=False, device=self.device)
        linear_large = torch.nn.Linear(10, 10, bias=False, device=self.device)
        optimizer = HybridMuonOptimizer(
            list(linear_small.parameters()) + list(linear_large.parameters()),
            lr=0.02,
            min_2d_dim=2,
        )

        x = torch.randn(4, 10, device=self.device)
        loss = linear_small(x).sum() + linear_large(x).sum()
        loss.backward()
        optimizer.step()

        # Small 2D weight should use Adam fallback.
        self.assertIn("exp_avg", optimizer.state[linear_small.weight])
        self.assertNotIn("momentum_buffer", optimizer.state[linear_small.weight])

        # Large 2D weight should use Muon.
        self.assertIn("momentum_buffer", optimizer.state[linear_large.weight])
        self.assertNotIn("exp_avg", optimizer.state[linear_large.weight])

    def test_lr_adjust_modes(self) -> None:
        """Test lr_adjust modes: match-RMS (<=0) vs rectangular (>0)."""
        torch.manual_seed(42)

        model1 = torch.nn.Linear(10, 20, bias=False, device=self.device)
        model2 = torch.nn.Linear(10, 20, bias=False, device=self.device)
        model2.load_state_dict(model1.state_dict())

        opt1 = HybridMuonOptimizer(model1.parameters(), lr=0.02, lr_adjust=0.0)
        opt2 = HybridMuonOptimizer(model2.parameters(), lr=0.02, lr_adjust=10.0)

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


@unittest.skipIf(not BF16_SUPPORTED, "bf16 matmul not supported on this device")
class TestHybridMuonOptimizerStateDict(unittest.TestCase):
    """Test optimizer state dict save/load."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_state_dict_save_load(self) -> None:
        """Test saving and loading optimizer state."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 10, device=self.device)
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02)

        for _ in range(3):
            optimizer.zero_grad()
            x = torch.randn(4, 10, device=self.device)
            model(x).sum().backward()
            optimizer.step()

        state_dict = optimizer.state_dict()

        optimizer2 = HybridMuonOptimizer(model.parameters(), lr=0.02)
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


@unittest.skipIf(not BF16_SUPPORTED, "bf16 matmul not supported on this device")
class TestFlashMuon(unittest.TestCase):
    """Test flash_muon triton-accelerated Newton-Schulz path."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    def test_flash_muon_false_runs(self) -> None:
        """Test that flash_muon=False uses pure PyTorch path without error."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 20, device=self.device)
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02, flash_muon=False)
        x = torch.randn(4, 10, device=self.device)
        model(x).sum().backward()
        optimizer.step()
        # Should complete without error

    def test_flash_muon_true_runs(self) -> None:
        """Test that flash_muon=True runs (falls back on CPU, uses triton on CUDA)."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 20, device=self.device)
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02, flash_muon=True)
        x = torch.randn(4, 10, device=self.device)
        model(x).sum().backward()
        optimizer.step()

    def test_flash_vs_pytorch_consistency(self) -> None:
        """Test that flash and non-flash paths produce consistent results.

        On CPU (no triton), both paths are identical (PyTorch fallback).
        On CUDA with triton, results should be close (same math, bf16 rounding).
        """
        torch.manual_seed(42)
        model1 = torch.nn.Linear(32, 64, device=self.device)
        model2 = torch.nn.Linear(32, 64, device=self.device)
        model2.load_state_dict(model1.state_dict())

        opt1 = HybridMuonOptimizer(model1.parameters(), lr=0.02, flash_muon=False)
        opt2 = HybridMuonOptimizer(model2.parameters(), lr=0.02, flash_muon=True)

        x = torch.randn(4, 32, device=self.device)

        opt1.zero_grad()
        model1(x).sum().backward()
        opt1.step()

        opt2.zero_grad()
        model2(x).sum().backward()
        opt2.step()

        # Both paths should produce similar results
        self.assertTrue(
            torch.allclose(model1.weight, model2.weight, atol=1e-2),
            f"Flash and non-flash weight diff: {(model1.weight - model2.weight).abs().max().item():.6f}",
        )
        self.assertTrue(
            torch.allclose(model1.bias, model2.bias, atol=1e-2),
            f"Flash and non-flash bias diff: {(model1.bias - model2.bias).abs().max().item():.6f}",
        )

    @unittest.skipIf(
        not (TRITON_AVAILABLE and env.DEVICE.type == "cuda"),
        "Triton + CUDA required for flash path verification",
    )
    def test_flash_path_actually_used(self) -> None:
        """Verify that flash path is actually active when triton + CUDA available."""
        from deepmd.pt.optimizer.hybrid_muon import (
            FLASH_MIN_DIM,
        )

        torch.manual_seed(42)
        # Use matrix large enough to exceed FLASH_MIN_DIM threshold
        dim = max(FLASH_MIN_DIM, 128)
        model = torch.nn.Linear(dim, dim * 2, device=self.device)
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02, flash_muon=True)
        # _use_flash should be True when triton is available
        self.assertTrue(optimizer._use_flash)
        # _ns_buffers should be empty before first step
        self.assertEqual(len(optimizer._ns_buffers), 0)

        x = torch.randn(4, dim, device=self.device)
        model(x).sum().backward()
        optimizer.step()

        # After step, buffers should have been allocated for the weight matrix
        self.assertGreater(len(optimizer._ns_buffers), 0)


if __name__ == "__main__":
    unittest.main()
