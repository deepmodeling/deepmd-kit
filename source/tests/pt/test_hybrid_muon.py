# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.optimizer.hybrid_muon import (
    MAGMA_MIN_SCALE,
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

    def test_slice_mode_uses_muon_for_3d_weight(self) -> None:
        """Test muon_mode='slice' + name rules route params as expected."""
        torch.manual_seed(42)

        class ToySliceModule(torch.nn.Module):
            def __init__(self, device: torch.device) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(2, 6, 4, dtype=torch.float32, device=device)
                )
                self.adam_scale = torch.nn.Parameter(
                    torch.ones(2, 6, dtype=torch.float32, device=device)
                )
                self.adam_stack = torch.nn.ParameterList(
                    [
                        torch.nn.Parameter(
                            torch.ones(2, 6, dtype=torch.float32, device=device)
                        )
                    ]
                )
                self.adamw_layer_scale = torch.nn.Parameter(
                    torch.ones(2, 6, dtype=torch.float32, device=device)
                )
                # Contains "bias" (case-insensitive) but not prefix.
                self.gateBiAsScale = torch.nn.Parameter(
                    torch.ones(2, 6, dtype=torch.float32, device=device)
                )
                # Module name contains "bias", but parameter leaf is "weight".
                self.bias_proj = torch.nn.Linear(4, 6, bias=False, device=device)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.einsum("bi,foi->bfo", x, self.weight)
                y = y * self.adam_scale.unsqueeze(0)
                y = y * self.adam_stack[0].unsqueeze(0)
                y = y * self.adamw_layer_scale.unsqueeze(0)
                y = y * self.gateBiAsScale.unsqueeze(0)
                y = y + self.bias_proj(x).unsqueeze(1)
                return y.sum()

        model = ToySliceModule(self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            muon_mode="slice",
            named_parameters=tuple(model.named_parameters()),
        )

        x = torch.randn(4, 4, device=self.device)
        model(x).backward()
        optimizer.step()

        # 3D weight → Muon (slice mode)
        self.assertIn("momentum_buffer", optimizer.state[model.weight])
        self.assertNotIn("exp_avg", optimizer.state[model.weight])
        # adam_ prefix → Adam (no weight decay)
        self.assertIn("exp_avg", optimizer.state[model.adam_scale])
        self.assertNotIn("momentum_buffer", optimizer.state[model.adam_scale])
        self.assertIn("exp_avg", optimizer.state[model.adam_stack[0]])
        self.assertNotIn("momentum_buffer", optimizer.state[model.adam_stack[0]])
        # adamw_ prefix → AdamW (decoupled weight decay)
        self.assertIn("exp_avg", optimizer.state[model.adamw_layer_scale])
        self.assertNotIn("momentum_buffer", optimizer.state[model.adamw_layer_scale])
        # Contains "bias" (case-insensitive) → Adam
        self.assertIn("exp_avg", optimizer.state[model.gateBiAsScale])
        self.assertNotIn("momentum_buffer", optimizer.state[model.gateBiAsScale])
        # Module name "bias_proj" but leaf is "weight" → Muon
        self.assertIn("momentum_buffer", optimizer.state[model.bias_proj.weight])
        self.assertNotIn("exp_avg", optimizer.state[model.bias_proj.weight])

    def test_2d_mode_routes_3d_weight_to_adam(self) -> None:
        """Test muon_mode='2d' routes 3D matrix weights to Adam."""
        torch.manual_seed(42)

        class Toy2DModeModule(torch.nn.Module):
            def __init__(self, device: torch.device) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(2, 6, 4, dtype=torch.float32, device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.einsum("bi,foi->bfo", x, self.weight).sum()

        model = Toy2DModeModule(self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            muon_mode="2d",
            named_parameters=tuple(model.named_parameters()),
        )

        x = torch.randn(4, 4, device=self.device)
        model(x).backward()
        optimizer.step()

        self.assertIn("exp_avg", optimizer.state[model.weight])
        self.assertNotIn("momentum_buffer", optimizer.state[model.weight])

    def test_2d_mode_singleton_3d_routes_to_muon(self) -> None:
        """Test muon_mode='2d' treats singleton-expanded matrix as 2D."""
        torch.manual_seed(42)

        class ToySingleton2DModeModule(torch.nn.Module):
            def __init__(self, device: torch.device) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(1, 6, 4, dtype=torch.float32, device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.einsum("bi,foi->bfo", x, self.weight).sum()

        model = ToySingleton2DModeModule(self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            muon_mode="2d",
            named_parameters=tuple(model.named_parameters()),
        )

        x = torch.randn(4, 4, device=self.device)
        model(x).backward()
        optimizer.step()

        self.assertIn("momentum_buffer", optimizer.state[model.weight])
        self.assertNotIn("exp_avg", optimizer.state[model.weight])

    def test_magma_muon_slice_state_and_range(self) -> None:
        """Test magma_muon creates bounded per-slice scores on Muon path."""
        torch.manual_seed(42)

        class ToyMagmaSlice(torch.nn.Module):
            def __init__(self, device: torch.device) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(2, 6, 4, dtype=torch.float32, device=device)
                )
                self.adam_scale = torch.nn.Parameter(
                    torch.ones(2, 6, dtype=torch.float32, device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.einsum("bi,foi->bfo", x, self.weight)
                y = y * self.adam_scale.unsqueeze(0)
                return y.sum()

        model = ToyMagmaSlice(self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            muon_mode="slice",
            named_parameters=tuple(model.named_parameters()),
            magma_muon=True,
        )

        x = torch.randn(4, 4, device=self.device)
        optimizer.zero_grad()
        model(x).backward()
        optimizer.step()

        score = optimizer.state[model.weight]["magma_score"]
        self.assertEqual(score.shape, (2,))
        self.assertTrue(torch.all(score >= 0.0))
        self.assertTrue(torch.all(score <= 1.0))
        scale = MAGMA_MIN_SCALE + (1.0 - MAGMA_MIN_SCALE) * score
        self.assertTrue(torch.all(scale >= MAGMA_MIN_SCALE))
        self.assertTrue(torch.all(scale <= 1.0))
        self.assertNotIn("magma_score", optimizer.state[model.adam_scale])

    def test_magma_muon_only_affects_muon_path(self) -> None:
        """Test Magma damping changes Muon updates but keeps Adam path unchanged."""
        torch.manual_seed(42)

        class ToyMagmaMixed(torch.nn.Module):
            def __init__(self, device: torch.device) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(2, 6, 4, dtype=torch.float32, device=device)
                )
                self.adam_scale = torch.nn.Parameter(
                    torch.ones(2, 6, dtype=torch.float32, device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.einsum("bi,foi->bfo", x, self.weight)
                y = y * self.adam_scale.unsqueeze(0)
                return y.sum()

        model1 = ToyMagmaMixed(self.device)
        model2 = ToyMagmaMixed(self.device)
        model2.load_state_dict(model1.state_dict())

        opt_off = HybridMuonOptimizer(
            model1.parameters(),
            lr=0.02,
            muon_mode="slice",
            named_parameters=tuple(model1.named_parameters()),
            magma_muon=False,
        )
        opt_on = HybridMuonOptimizer(
            model2.parameters(),
            lr=0.02,
            muon_mode="slice",
            named_parameters=tuple(model2.named_parameters()),
            magma_muon=True,
        )

        x = torch.randn(4, 4, device=self.device)
        opt_off.zero_grad()
        model1(x).backward()
        opt_off.step()
        opt_on.zero_grad()
        model2(x).backward()
        opt_on.step()

        self.assertFalse(torch.allclose(model1.weight, model2.weight))
        self.assertTrue(
            torch.allclose(model1.adam_scale, model2.adam_scale, atol=1e-7, rtol=1e-7)
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
