# SPDX-License-Identifier: LGPL-3.0-or-later
import bisect
import math
import unittest

import torch

from deepmd.pt.optimizer.hybrid_muon import (
    FLASH_MIN_DIM,
    MAGMA_MIN_SCALE,
    TRITON_AVAILABLE,
    HybridMuonOptimizer,
    _batched_newton_schulz_orth,
    _GramNewtonSchulzOrthogonalizer,
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


def _fp16_matmul_supported(device: torch.device) -> bool:
    """Check if float16 matmul is reliably supported on the given device."""
    try:
        a = torch.randn(4, 4, dtype=torch.float16, device=device)
        _ = torch.mm(a, a.T)
        return True
    except (RuntimeError, TypeError):
        return False


BF16_SUPPORTED = _bf16_matmul_supported(env.DEVICE)
FP16_SUPPORTED = _fp16_matmul_supported(env.DEVICE)


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

    def test_enable_gram_defaults_to_true(self) -> None:
        """Test enable_gram is enabled by default."""
        model = torch.nn.Linear(10, 20, bias=False, device=self.device)
        optimizer = HybridMuonOptimizer(model.parameters(), lr=0.02)
        self.assertTrue(optimizer.param_groups[0]["enable_gram"])

    def test_enable_gram_square_falls_back_to_standard(self) -> None:
        """Test square matrices keep using the standard path when Gram is enabled."""
        torch.manual_seed(42)
        model_standard = torch.nn.Linear(10, 10, device=self.device)
        model_gram = torch.nn.Linear(10, 10, device=self.device)
        model_gram.load_state_dict(model_standard.state_dict())

        opt_standard = HybridMuonOptimizer(
            model_standard.parameters(),
            lr=0.02,
            enable_gram=False,
            flash_muon=False,
        )
        opt_gram = HybridMuonOptimizer(
            model_gram.parameters(),
            lr=0.02,
            enable_gram=True,
            flash_muon=False,
        )

        x = torch.randn(4, 10, device=self.device)
        opt_standard.zero_grad()
        model_standard(x).sum().backward()
        opt_standard.step()

        opt_gram.zero_grad()
        model_gram(x).sum().backward()
        opt_gram.step()

        self.assertTrue(
            torch.allclose(
                model_standard.weight, model_gram.weight, atol=1e-6, rtol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(model_standard.bias, model_gram.bias, atol=1e-6, rtol=1e-6)
        )

    @unittest.skipIf(
        not FP16_SUPPORTED,
        "float16 matmul not supported on this device",
    )
    def test_enable_gram_rectangular_step_runs(self) -> None:
        """Test a rectangular Muon step runs successfully with Gram enabled."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 20, bias=False, device=self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            enable_gram=True,
            flash_muon=False,
        )
        initial_weight = model.weight.detach().clone()

        x = torch.randn(4, 10, device=self.device)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

        self.assertFalse(torch.allclose(initial_weight, model.weight))
        self.assertIn("momentum_buffer", optimizer.state[model.weight])

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
                # pt_expt uses the exact leaf name "b" for affine biases.
                self.b = torch.nn.Parameter(
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
                y = y + self.b.unsqueeze(0)
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
        # Exact pt_expt leaf "b" → Adam even when matrix-shaped
        self.assertIn("exp_avg", optimizer.state[model.b])
        self.assertNotIn("momentum_buffer", optimizer.state[model.b])
        self.assertTrue(
            any(
                entry["param"] is model.b
                for entry in optimizer._routing[0]["adam_no_decay"]
            )
        )
        self.assertFalse(
            any(
                entry["param"] is model.b
                for entry in optimizer._routing[0]["adam_decay"]
            )
        )
        # Module name "bias_proj" but leaf is "weight" → Muon
        self.assertIn("momentum_buffer", optimizer.state[model.bias_proj.weight])
        self.assertNotIn("exp_avg", optimizer.state[model.bias_proj.weight])

    def test_adam_patterns_route_matrices_to_adamw(self) -> None:
        """Matrices whose name contains an ``adam_patterns`` entry leave Muon."""
        torch.manual_seed(42)

        class ToyModel(torch.nn.Module):
            def __init__(self, device: torch.device) -> None:
                super().__init__()
                self.descriptor = torch.nn.Linear(4, 6, bias=False, device=device)
                self.fitting_net = torch.nn.Linear(6, 3, device=device)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fitting_net(self.descriptor(x)).sum()

        model = ToyModel(self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            weight_decay=0.01,
            named_parameters=tuple(model.named_parameters()),
            adam_patterns=["Fitting_Net"],
        )
        model(torch.randn(4, 4, device=self.device)).backward()
        optimizer.step()

        # The matching matrix takes the AdamW path, its vector plain Adam.
        self.assertIn("exp_avg", optimizer.state[model.fitting_net.weight])
        self.assertNotIn("momentum_buffer", optimizer.state[model.fitting_net.weight])
        routes = optimizer._routing[0]
        self.assertTrue(
            any(e["param"] is model.fitting_net.weight for e in routes["adam_decay"])
        )
        self.assertTrue(
            any(e["param"] is model.fitting_net.bias for e in routes["adam_no_decay"])
        )
        # A matrix outside the patterns stays on Muon.
        self.assertIn("momentum_buffer", optimizer.state[model.descriptor.weight])
        self.assertNotIn("exp_avg", optimizer.state[model.descriptor.weight])

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

        opt1 = HybridMuonOptimizer(
            model1.parameters(),
            lr=0.02,
            enable_gram=False,
            flash_muon=False,
        )
        opt2 = HybridMuonOptimizer(
            model2.parameters(),
            lr=0.02,
            enable_gram=False,
            flash_muon=True,
        )

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
        torch.manual_seed(42)
        # Use matrix large enough to exceed FLASH_MIN_DIM threshold
        dim = max(FLASH_MIN_DIM, 128)
        model = torch.nn.Linear(dim, dim * 2, device=self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            enable_gram=False,
            flash_muon=True,
        )
        # _use_flash should be True when triton is available
        self.assertTrue(optimizer._use_flash)
        # _ns_buffers should be empty before first step
        self.assertEqual(len(optimizer._ns_buffers), 0)

        x = torch.randn(4, dim, device=self.device)
        model(x).sum().backward()
        optimizer.step()

        # After step, buffers should have been allocated for the weight matrix
        self.assertGreater(len(optimizer._ns_buffers), 0)


@unittest.skipIf(not BF16_SUPPORTED, "bf16 matmul not supported on this device")
class TestColumnPadMergeEquivalence(unittest.TestCase):
    """Verify column-pad merge produces numerically identical NS orth results.

    The column-pad merge optimization zero-pads the column (large) dimension
    of rectangular matrices so that matrices with the same min_dim can share
    a single Gram NS call.  These tests verify the mathematical invariant:

        R_pad = [X | 0] @ [X | 0]^T = X @ X^T = R

    which guarantees the NS iteration is identical, and truncating the output
    to the original column count exactly recovers the unpadded result.
    """

    def setUp(self) -> None:
        self.device = env.DEVICE

    @unittest.skipIf(
        not FP16_SUPPORTED,
        "float16 matmul not supported on this device",
    )
    def test_gram_ns_column_pad_exact_equivalence(self) -> None:
        """Gram NS orth on X vs [X|0] must agree in the first n columns."""
        torch.manual_seed(123)
        gram_orth = _GramNewtonSchulzOrthogonalizer()

        for m, n, pad in [(16, 32, 32), (64, 128, 256), (64, 192, 192)]:
            with self.subTest(m=m, n=n, pad=pad):
                X = torch.randn(1, m, n, dtype=torch.float32, device=self.device)
                X_padded = torch.nn.functional.pad(X, (0, pad))  # (1, m, n+pad)

                # Call the uncompiled implementation directly so the two
                # invocations share an identical numerical path.
                out_orig = gram_orth._orthogonalize_impl(X)
                out_padded = gram_orth._orthogonalize_impl(X_padded)

                # Truncate padded output to original column count
                out_padded_trunc = out_padded[:, :, :n]
                # Padded columns must be zero
                out_padded_tail = out_padded[:, :, n:]

                self.assertTrue(
                    torch.allclose(out_orig, out_padded_trunc, atol=1e-3, rtol=1e-3),
                    f"shape ({m},{n}) pad {pad}: max diff = {(out_orig - out_padded_trunc).abs().max().item():.6f}",
                )
                self.assertTrue(
                    torch.allclose(
                        out_padded_tail,
                        torch.zeros_like(out_padded_tail),
                        atol=1e-3,
                    ),
                    f"shape ({m},{n}) pad {pad}: padded tail not zero, max = {out_padded_tail.abs().max().item():.6f}",
                )

    def test_standard_ns_column_pad_exact_equivalence(self) -> None:
        """Standard (quintic) NS orth on X vs [X|0] must agree."""
        torch.manual_seed(456)

        for m, n, pad in [(16, 32, 32), (32, 64, 64)]:
            with self.subTest(m=m, n=n, pad=pad):
                X = torch.randn(1, m, n, dtype=torch.float32, device=self.device)
                X_padded = torch.nn.functional.pad(X, (0, pad))

                out_orig = _batched_newton_schulz_orth(X)
                out_padded = _batched_newton_schulz_orth(X_padded)
                out_padded_trunc = out_padded[:, :, :n]

                self.assertTrue(
                    torch.allclose(out_orig, out_padded_trunc, atol=1e-3, rtol=1e-3),
                    f"shape ({m},{n}) pad {pad}: max diff = {(out_orig - out_padded_trunc).abs().max().item():.6f}",
                )

    @unittest.skipIf(
        not FP16_SUPPORTED,
        "float16 matmul not supported on this device",
    )
    def test_gram_ns_batch_pad_equivalence(self) -> None:
        """Batched column-padded Gram NS must agree with per-matrix Gram NS."""
        torch.manual_seed(789)
        gram_orth = _GramNewtonSchulzOrthogonalizer()
        min_dim = 64

        # Simulate two different max_dims in the same super-bucket
        shapes = [(min_dim, 128), (min_dim, 192), (min_dim, 384)]
        padded_max = max(s[1] for s in shapes)

        per_matrix_results = []
        padded_batch = []
        for m, n in shapes:
            X = torch.randn(1, m, n, dtype=torch.float32, device=self.device)
            per_matrix_results.append(gram_orth._orthogonalize_impl(X))
            padded_batch.append(torch.nn.functional.pad(X, (0, padded_max - n)))

        # Run Gram NS on the batched padded tensor
        stacked = torch.cat(padded_batch, dim=0)  # (3, 64, 384)
        batched_out = gram_orth._orthogonalize_impl(stacked)

        for i, (m, n) in enumerate(shapes):
            expected = per_matrix_results[i]
            actual = batched_out[i : i + 1, :, :n]
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-3, rtol=1e-3),
                f"shape ({m},{n}): batched-pad max diff = {(expected - actual).abs().max().item():.6f}",
            )

    @unittest.skipIf(
        not FP16_SUPPORTED,
        "float16 matmul not supported on this device",
    )
    def test_optimizer_step_column_pad_merge_e2e(self) -> None:
        """End-to-end: optimizer with mixed rectangular shapes runs correctly.

        Constructs a model with multiple rectangular weight matrices of
        different shapes (same min_dim) that trigger the column-pad merge
        path, then verifies all parameters are updated and the model produces
        finite outputs.
        """
        torch.manual_seed(42)

        class MixedRectModel(torch.nn.Module):
            """Model with rectangular weights sharing min_dim=8 but different max_dims."""

            def __init__(self, device: torch.device) -> None:
                super().__init__()
                # 3 rectangular weights: min_dim=8, max_dim varies
                self.w1 = torch.nn.Parameter(torch.randn(8, 16, device=device))
                self.w2 = torch.nn.Parameter(torch.randn(8, 32, device=device))
                self.w3 = torch.nn.Parameter(torch.randn(24, 8, device=device))
                # A square weight for the standard NS path
                self.w_sq = torch.nn.Parameter(torch.randn(16, 16, device=device))
                # 1D bias for Adam path
                self.bias = torch.nn.Parameter(torch.zeros(16, device=device))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, 8), w1: (8, 16) -> (B, 16)
                h = x @ self.w1 + self.bias
                h = h @ self.w_sq  # (B, 16)
                h2 = x @ self.w2  # (B, 32)
                h3 = x @ self.w3.T  # (B, 24); w3: (24, 8)
                return h.sum() + h2.sum() + h3.sum()

        model = MixedRectModel(self.device)
        init_state = {k: v.detach().clone() for k, v in model.named_parameters()}

        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.01,
            enable_gram=True,
            flash_muon=False,
            magma_muon=True,
            muon_mode="slice",
            named_parameters=tuple(model.named_parameters()),
        )

        x = torch.randn(4, 8, device=self.device)
        for _ in range(3):
            optimizer.zero_grad()
            model(x).backward()
            optimizer.step()

        # All params must have been updated
        for name, param in model.named_parameters():
            self.assertFalse(
                torch.allclose(param, init_state[name]),
                f"Parameter {name} was not updated after 3 steps",
            )

        # Output must be finite
        with torch.no_grad():
            out = model(x)
        self.assertTrue(
            torch.isfinite(out).all(),
            f"Model output is not finite: {out}",
        )

        # Muon params should have momentum_buffer
        for name in ["w1", "w2", "w3", "w_sq"]:
            p = getattr(model, name)
            self.assertIn(
                "momentum_buffer",
                optimizer.state[p],
                f"{name} missing momentum_buffer",
            )
        # Magma should be active on Muon params
        for name in ["w1", "w2", "w3", "w_sq"]:
            p = getattr(model, name)
            self.assertIn(
                "magma_score",
                optimizer.state[p],
                f"{name} missing magma_score",
            )
        # Bias uses Adam
        self.assertIn("exp_avg", optimizer.state[model.bias])


class _MixedRouteModel(torch.nn.Module):
    """Small model covering every optimizer route.

    ``square`` exercises the square Newton-Schulz path (weight) and the Adam
    path (bias), ``rect`` the rectangular Gram path, ``adamw_gate`` the
    name-routed AdamW path, and ``scale`` the 1D Adam path.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.square = torch.nn.Linear(32, 32, bias=True, device=device)
        self.rect = torch.nn.Linear(32, 48, bias=False, device=device)
        self.adamw_gate = torch.nn.Parameter(torch.randn(48, 16, device=device) * 0.1)
        self.scale = torch.nn.Parameter(torch.ones(16, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.square(x))
        h = torch.tanh(self.rect(h))
        return (h @ self.adamw_gate) * self.scale


class _AlternatingHeadModel(torch.nn.Module):
    """Model whose task-specific heads produce two gradient signatures."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(32, 32, device=device)
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(32, 16, device=device) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, head: int) -> torch.Tensor:
        return self.heads[head](torch.tanh(self.shared(x)))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA graph capture needs CUDA")
class TestHybridMuonCudaGraph(unittest.TestCase):
    """The whole-step CUDA graph must be an exact execution detail.

    Every test runs a coupled trajectory -- the gradients of each step
    depend on every earlier update -- with a per-step learning-rate
    schedule, so the graph's device-resident scalars (learning rate,
    bias corrections) are exercised against the eager execution of
    the identical update.
    """

    N_STEPS = 8

    def setUp(self) -> None:
        self.device = torch.device("cuda")
        torch.manual_seed(1234)
        self.inputs = torch.randn(16, 32, device=self.device)
        self.targets = torch.randn(16, 16, device=self.device)

    def _make(self, graph_on: bool) -> tuple[torch.nn.Module, HybridMuonOptimizer]:
        torch.manual_seed(7)
        model = _MixedRouteModel(self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            weight_decay=0.01,
            named_parameters=list(model.named_parameters()),
        )
        optimizer._graph_enabled = graph_on
        return model, optimizer

    def _make_alternating(
        self, graph_on: bool
    ) -> tuple[_AlternatingHeadModel, HybridMuonOptimizer]:
        torch.manual_seed(17)
        model = _AlternatingHeadModel(self.device)
        optimizer = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            weight_decay=0.01,
            named_parameters=list(model.named_parameters()),
        )
        optimizer._graph_enabled = graph_on
        return model, optimizer

    def _run(
        self,
        model: torch.nn.Module,
        optimizer: HybridMuonOptimizer,
        n_steps: int,
        lr_decay: float = 0.8,
    ) -> list[torch.Tensor]:
        for step in range(n_steps):
            for group in optimizer.param_groups:
                group["lr"] = 0.02 * (lr_decay**step)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(self.inputs) - self.targets) ** 2).mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        return [p.detach().clone() for p in model.parameters()]

    def _run_alternating(
        self,
        model: _AlternatingHeadModel,
        optimizer: HybridMuonOptimizer,
        start_step: int,
        end_step: int,
    ) -> list[torch.Tensor]:
        for step in range(start_step, end_step):
            for group in optimizer.param_groups:
                group["lr"] = 0.02 * (0.8**step)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(self.inputs, step % 2) - self.targets) ** 2).mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        return [p.detach().clone() for p in model.parameters()]

    def test_graph_matches_eager_trajectory(self) -> None:
        """Graph and eager trajectories agree on a deterministic model."""
        eager = self._run(*self._make(graph_on=False), self.N_STEPS)
        graph_model, graph_opt = self._make(graph_on=True)
        graph = self._run(graph_model, graph_opt, self.N_STEPS)
        self.assertEqual(len(graph_opt._graphs), 1, "graph was never captured")
        # The replay re-executes the captured kernel sequence on the same
        # operands, so on a deterministic model the trajectories are
        # bitwise identical; any tolerance would hide a replay-frozen
        # scalar or state tensor.
        for pe, pg in zip(eager, graph, strict=True):
            torch.testing.assert_close(pe, pg, rtol=0.0, atol=0.0)

    def test_lr_schedule_reaches_replays(self) -> None:
        """The device learning rate follows the host schedule across replays.

        A frozen learning rate is the canonical capture bug; two schedules
        that share the capture-time value but diverge afterwards must
        produce different trajectories.
        """
        model_a, opt_a = self._make(graph_on=True)
        flat = self._run(model_a, opt_a, self.N_STEPS, lr_decay=1.0)
        model_b, opt_b = self._make(graph_on=True)
        decayed = self._run(model_b, opt_b, self.N_STEPS, lr_decay=0.5)
        max_diff = max(
            (a - b).abs().max().item() for a, b in zip(flat, decayed, strict=True)
        )
        self.assertGreater(max_diff, 1e-5)

    def test_uniform_bias_clock_advances_inside_graph(self) -> None:
        """A fixed owner set retains the group-clock fast path."""
        model, optimizer = self._make(graph_on=True)
        self._run(model, optimizer, self.N_STEPS)
        group = optimizer.param_groups[0]
        beta1, beta2 = group["adam_betas"]
        self.assertFalse(optimizer._per_parameter_adam_clock)
        self.assertAlmostEqual(
            group["beta1_pow_device"].item(), beta1**self.N_STEPS, places=6
        )
        self.assertAlmostEqual(
            group["beta2_pow_device"].item(), beta2**self.N_STEPS, places=6
        )
        for param in model.parameters():
            state = optimizer.state[param]
            if "exp_avg" in state:
                self.assertNotIn("bias_correction1", state)
                self.assertNotIn("bias_correction2", state)

    def test_legacy_bias_power_migration(self) -> None:
        """Per-parameter float powers become float32 device corrections."""
        model, optimizer = self._make(graph_on=True)
        self._run(model, optimizer, 3)
        state_dict = optimizer.state_dict()
        for group in state_dict["param_groups"]:
            group.pop("lr_device", None)
        for state in state_dict["state"].values():
            if "exp_avg" in state:
                state["beta1_pow"] = 0.9**3
                state["beta2_pow"] = 0.95**3

        _model2, optimizer2 = self._make(graph_on=True)
        optimizer2.load_state_dict(state_dict)
        optimizer2._build_param_routing()
        optimizer2._migrate_bias_corrections()
        self.assertTrue(optimizer2._per_parameter_adam_clock)
        for param, state in optimizer2.state.items():
            if "exp_avg" in state:
                self.assertEqual(state["bias_correction1"].dtype, torch.float32)
                self.assertEqual(state["bias_correction1"].device, param.device)
                self.assertAlmostEqual(
                    state["bias_correction1"].item(), 1.0 - 0.9**3, places=6
                )
                self.assertAlmostEqual(
                    state["bias_correction2"].item(), 1.0 - 0.95**3, places=6
                )

    def test_group_bias_power_migration(self) -> None:
        """Single-signature checkpoints retain the uniform fast path."""
        model, optimizer = self._make(graph_on=True)
        self._run(model, optimizer, 3)
        state_dict = optimizer.state_dict()
        for group in state_dict["param_groups"]:
            group["beta1_pow_device"] = torch.tensor(0.9**3, device=self.device)
            group["beta2_pow_device"] = torch.tensor(0.95**3, device=self.device)
        for state in state_dict["state"].values():
            state.pop("bias_correction1", None)
            state.pop("bias_correction2", None)

        model2, optimizer2 = self._make(graph_on=True)
        optimizer2.load_state_dict(state_dict)
        optimizer2._build_param_routing()
        optimizer2._migrate_bias_corrections()
        group = optimizer2.param_groups[0]
        self.assertFalse(optimizer2._per_parameter_adam_clock)
        self.assertAlmostEqual(group["beta1_pow_device"].item(), 0.9**3, places=6)
        self.assertAlmostEqual(group["beta2_pow_device"].item(), 0.95**3, places=6)
        for state in optimizer2.state.values():
            if "exp_avg" in state:
                self.assertNotIn("bias_correction1", state)
                self.assertNotIn("bias_correction2", state)

    def test_late_adam_owner_starts_at_its_first_update(self) -> None:
        """A newly active task head starts at Adam step one."""
        for use_foreach in (True, False):
            first = torch.nn.Parameter(torch.zeros(4, device=self.device))
            late = torch.nn.Parameter(torch.zeros(4, device=self.device))
            optimizer = HybridMuonOptimizer(
                [first, late],
                lr=0.1,
                weight_decay=0.0,
                use_foreach=use_foreach,
            )

            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                first.grad = torch.ones_like(first)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            late.grad = torch.ones_like(late)
            optimizer.step()

            with self.subTest(use_foreach=use_foreach):
                torch.testing.assert_close(
                    late, torch.full_like(late, -0.1), rtol=0.0, atol=1e-7
                )
                self.assertTrue(optimizer._per_parameter_adam_clock)
                self.assertAlmostEqual(
                    optimizer.state[first]["bias_correction1"].item(),
                    1.0 - 0.9**3,
                    places=6,
                )
                self.assertAlmostEqual(
                    optimizer.state[late]["bias_correction1"].item(),
                    0.1,
                    places=6,
                )

    def test_dynamic_signatures_match_eager_trajectory(self) -> None:
        """Alternating task heads use independent cached optimizer graphs."""
        n_steps = 10
        eager = self._run_alternating(
            *self._make_alternating(graph_on=False), 0, n_steps
        )
        graph_model, graph_opt = self._make_alternating(graph_on=True)
        graph = self._run_alternating(graph_model, graph_opt, 0, n_steps)

        self.assertEqual(len(graph_opt._graphs), 2)
        self.assertFalse(graph_opt._graph_warmups)
        self.assertTrue(graph_opt._per_parameter_adam_clock)
        self.assertEqual(
            sum(buffer is not None for buffer in graph_opt._static_grad_buffers),
            len(tuple(graph_model.parameters())),
        )
        for entry in graph_opt._graphs.values():
            self.assertEqual(entry.graph.pool(), graph_opt._graph_pool)
        for eager_param, graph_param in zip(eager, graph, strict=True):
            torch.testing.assert_close(eager_param, graph_param, rtol=0.0, atol=0.0)

        beta1 = graph_opt.param_groups[0]["adam_betas"][0]
        self.assertAlmostEqual(
            graph_opt.state[graph_model.shared.bias]["bias_correction1"].item(),
            1.0 - beta1**n_steps,
            places=6,
        )
        for head in graph_model.heads:
            self.assertAlmostEqual(
                graph_opt.state[head.bias]["bias_correction1"].item(),
                1.0 - beta1 ** (n_steps // 2),
                places=6,
            )

    def test_dynamic_state_dict_roundtrip_resumes_trajectory(self) -> None:
        """Alternating-head Adam clocks survive a checkpoint roundtrip."""
        n_steps = 10
        reference = self._run_alternating(
            *self._make_alternating(graph_on=True), 0, n_steps
        )

        model, optimizer = self._make_alternating(graph_on=True)
        self._run_alternating(model, optimizer, 0, 5)
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

        model2, optimizer2 = self._make_alternating(graph_on=True)
        model2.load_state_dict(model_state)
        optimizer2.load_state_dict(optimizer_state)
        self._run_alternating(model2, optimizer2, 5, n_steps)
        for reference_param, resumed_param in zip(
            reference, model2.parameters(), strict=True
        ):
            torch.testing.assert_close(
                reference_param, resumed_param.detach(), rtol=0.0, atol=0.0
            )

    def test_state_dict_roundtrip_resumes_trajectory(self) -> None:
        """Save/load mid-trajectory reproduces the uninterrupted run."""
        reference = self._run(*self._make(graph_on=True), self.N_STEPS)

        model, optimizer = self._make(graph_on=True)
        self._run(model, optimizer, 4)
        payload = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
        }

        model2, optimizer2 = self._make(graph_on=True)
        model2.load_state_dict(payload["model"])
        optimizer2.load_state_dict(payload["opt"])
        for step in range(4, self.N_STEPS):
            for group in optimizer2.param_groups:
                group["lr"] = 0.02 * (0.8**step)
            optimizer2.zero_grad(set_to_none=True)
            loss = ((model2(self.inputs) - self.targets) ** 2).mean()
            loss.backward()
            optimizer2.step()
        torch.cuda.synchronize()
        for pr, pg in zip(reference, model2.parameters(), strict=True):
            torch.testing.assert_close(pr, pg.detach(), rtol=0.0, atol=0.0)

    def test_eager_reference_path_stays_eager(self) -> None:
        """The eager reference execution never captures a graph."""
        model, optimizer = self._make(graph_on=False)
        self._run(model, optimizer, 4)
        self.assertFalse(optimizer._graphs)


@unittest.skipIf(not BF16_SUPPORTED, "bf16 matmul not supported on this device")
class TestHybridMuonPhysMuon(unittest.TestCase):
    """Tests for PhysMuon two-mode memory, gated readout, and spike guard."""

    def setUp(self) -> None:
        self.device = env.DEVICE

    @staticmethod
    def _reference_physmuon_momentum_step(
        grad: torch.Tensor,
        ref_state: dict[str, torch.Tensor | int],
        *,
        momentum: float,
        start_step: int,
        beta_fast: float,
        beta_slow: float,
        w_fast: float,
        use_gate: bool,
        w_max: float,
        gate_beta: float,
        shadow: bool,
        spike: bool,
    ) -> torch.Tensor:
        """Reference single-tensor PhysMuon two-mode momentum update."""
        if "step" not in ref_state:
            ref_state["step"] = 0
            ref_state["momentum"] = torch.zeros_like(grad)
            if shadow:
                ref_state["m_fast"] = torch.zeros_like(grad)
                ref_state["m_slow"] = torch.zeros_like(grad)
        t = int(ref_state["step"])
        ref_state["step"] = t + 1
        buf = ref_state["momentum"]
        assert isinstance(buf, torch.Tensor)
        if t < start_step and not shadow:
            buf.lerp_(grad, 1.0 - momentum)
            return torch.lerp(grad, buf, momentum)
        if "m_fast" not in ref_state:
            buf.lerp_(grad, 1.0 - momentum)
            u = torch.lerp(grad, buf, momentum)
            ref_state["m_fast"] = buf.clone()
            ref_state["m_slow"] = buf.clone()
            return u
        m_fast = ref_state["m_fast"]
        m_slow = ref_state["m_slow"]
        assert isinstance(m_fast, torch.Tensor) and isinstance(m_slow, torch.Tensor)
        if use_gate:
            gf = grad.float()
            g_norm = gf.norm().clamp_min(1e-20)
            mf = m_fast.float()
            osc = (-(gf * mf).sum() / (g_norm * mf.norm().clamp_min(1e-20))).clamp_min(
                0.0
            )
            drift = m_slow.float().norm() / g_norm
            if "osc" not in ref_state:
                ref_state["osc"] = osc.clone()
                ref_state["drift"] = drift.clone()
            else:
                ref_osc = ref_state["osc"]
                ref_drift = ref_state["drift"]
                assert isinstance(ref_osc, torch.Tensor)
                assert isinstance(ref_drift, torch.Tensor)
                ref_osc.lerp_(osc, 1.0 - gate_beta)
                ref_drift.lerp_(drift, 1.0 - gate_beta)
        m_fast.lerp_(grad, 1.0 - beta_fast)
        if not spike:
            m_slow.lerp_(grad, 1.0 - beta_slow)
        if t < start_step:
            buf.lerp_(grad, 1.0 - momentum)
            return torch.lerp(grad, buf, momentum)
        if use_gate:
            rho_f = (1.0 - beta_fast) / (1.0 + beta_fast)
            ref_osc = ref_state["osc"]
            ref_drift = ref_state["drift"]
            assert isinstance(ref_osc, torch.Tensor)
            assert isinstance(ref_drift, torch.Tensor)
            w: torch.Tensor | float = w_max * (
                1.0 - rho_f * ref_osc / ref_drift.clamp_min(1e-20)
            ).clamp(0.0, 1.0)
        else:
            w = w_fast
        m_eff = (w * m_fast + (1.0 - w) * m_slow).to(dtype=buf.dtype)
        buf.copy_(m_eff)
        return torch.lerp(grad, buf, momentum)

    def test_physmuon_disabled_is_bitwise_identical(self) -> None:
        """Default physmuon=False leaves updates and state_dict untouched."""
        torch.manual_seed(0)
        w1 = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
        b1 = torch.nn.Parameter(torch.randn(8, device=self.device))
        w2 = torch.nn.Parameter(w1.detach().clone())
        b2 = torch.nn.Parameter(b1.detach().clone())

        opt_default = HybridMuonOptimizer([w1, b1], lr=1e-2)
        opt_explicit = HybridMuonOptimizer([w2, b2], lr=1e-2, physmuon=False)

        for step in range(5):
            torch.manual_seed(100 + step)
            gw = torch.randn_like(w1)
            gb = torch.randn_like(b1)
            w1.grad, b1.grad = gw.clone(), gb.clone()
            w2.grad, b2.grad = gw.clone(), gb.clone()
            opt_default.step()
            opt_explicit.step()

        torch.testing.assert_close(w1, w2, rtol=0.0, atol=0.0)
        torch.testing.assert_close(b1, b2, rtol=0.0, atol=0.0)
        self.assertNotIn("pm_step", opt_default.param_groups[0])
        self.assertNotIn("pm_active_device", opt_default.param_groups[0])
        self.assertNotIn("pm_m_fast", opt_default.state[w1])

    def test_before_start_step_matches_baseline(self) -> None:
        """Steps before physmuon_start_step match stock HybridMuon bitwise."""
        for shadow in (True, False):
            with self.subTest(shadow=shadow):
                torch.manual_seed(11)
                p_base = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
                p_pm = torch.nn.Parameter(p_base.detach().clone())

                opt_base = HybridMuonOptimizer(
                    [p_base], lr=1e-2, magma_muon=True, enable_gram=False
                )
                opt_pm = HybridMuonOptimizer(
                    [p_pm],
                    lr=1e-2,
                    magma_muon=True,
                    enable_gram=False,
                    physmuon=True,
                    physmuon_start_step=5,
                    physmuon_shadow=shadow,
                )

                # With shadow=False, step == start_step also equals baseline
                # because both modes are seeded from live momentum on that step.
                n_equal_steps = 5 if shadow else 6
                for step in range(n_equal_steps):
                    torch.manual_seed(200 + step)
                    g = torch.randn_like(p_base)
                    p_base.grad = g.clone()
                    p_pm.grad = g.clone()
                    opt_base.step()
                    opt_pm.step()
                    torch.testing.assert_close(p_base, p_pm, rtol=0.0, atol=0.0)

                # One step further: two-mode memory takes effect and diverges.
                torch.manual_seed(300)
                g = torch.randn_like(p_base)
                p_base.grad = g.clone()
                p_pm.grad = g.clone()
                opt_base.step()
                opt_pm.step()
                self.assertGreater((p_base - p_pm).abs().max().item(), 1e-6)

    def test_exact_math_parity_with_reference(self) -> None:
        """PhysMuon modes, gate stats, and effective memory match reference."""
        configs = [
            {"use_gate": True, "w_fast": 0.3, "shadow": True, "start_step": 4},
            {"use_gate": False, "w_fast": 0.3, "shadow": True, "start_step": 4},
            {"use_gate": False, "w_fast": 0.0, "shadow": True, "start_step": 4},
            {"use_gate": True, "w_fast": 0.3, "shadow": False, "start_step": 4},
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                torch.manual_seed(42)
                p = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
                opt = HybridMuonOptimizer(
                    [p],
                    lr=1e-2,
                    momentum=0.95,
                    weight_decay=0.0,
                    enable_gram=False,
                    magma_muon=False,
                    physmuon=True,
                    physmuon_start_step=cfg["start_step"],
                    physmuon_beta_fast=0.90,
                    physmuon_beta_slow=0.995,
                    physmuon_w_fast=cfg["w_fast"],
                    physmuon_gate=cfg["use_gate"],
                    physmuon_w_max=0.3,
                    physmuon_gate_beta=0.9,
                    physmuon_shadow=cfg["shadow"],
                )
                ref_state: dict[str, torch.Tensor | int] = {}
                for step in range(12):
                    torch.manual_seed(500 + step)
                    g = torch.randn_like(p)
                    # Inject occasional sign reversals to exercise the osc gate.
                    if step % 3 == 1:
                        g = -g
                    p.grad = g.clone()
                    opt.step()
                    self._reference_physmuon_momentum_step(
                        g.clone(),
                        ref_state,
                        momentum=0.95,
                        start_step=cfg["start_step"],
                        beta_fast=0.90,
                        beta_slow=0.995,
                        w_fast=cfg["w_fast"],
                        use_gate=cfg["use_gate"],
                        w_max=0.3,
                        gate_beta=0.9,
                        shadow=cfg["shadow"],
                        spike=False,
                    )
                    st = opt.state[p]
                    torch.testing.assert_close(
                        st["momentum_buffer"],
                        ref_state["momentum"],
                        rtol=1e-6,
                        atol=1e-6,
                    )
                    if "m_fast" in ref_state:
                        torch.testing.assert_close(
                            st["pm_m_fast"], ref_state["m_fast"], rtol=1e-6, atol=1e-6
                        )
                        torch.testing.assert_close(
                            st["pm_m_slow"], ref_state["m_slow"], rtol=1e-6, atol=1e-6
                        )
                    if cfg["use_gate"] and "osc" in ref_state:
                        torch.testing.assert_close(
                            st["pm_osc"], ref_state["osc"], rtol=1e-6, atol=1e-6
                        )
                        torch.testing.assert_close(
                            st["pm_drift"], ref_state["drift"], rtol=1e-6, atol=1e-6
                        )

    def test_spike_guard_matches_bisect_reference_and_protects_slow_mode(self) -> None:
        """GPU-resident spike guard matches bisect median and shields m_slow."""
        torch.manual_seed(7)
        p = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
        window = 10
        spike_k = 5.0
        opt = HybridMuonOptimizer(
            [p],
            lr=1e-2,
            weight_decay=0.0,
            enable_gram=False,
            magma_muon=False,
            physmuon=True,
            physmuon_start_step=0,
            physmuon_beta_fast=0.90,
            physmuon_beta_slow=0.995,
            physmuon_gate=False,
            physmuon_w_fast=0.3,
            physmuon_shadow=True,
            physmuon_spike_guard=spike_k,
            physmuon_spike_window=window,
        )

        # Synthetic pre-clip norms: warmup (10 steps around 1.0), finite spike,
        # NaN spike, inf spike, then more normal steps cycling the FIFO window.
        norms = [1.0 + 0.05 * (i % 4) for i in range(12)]
        norms += [25.0, float("nan"), float("inf"), 1.1, 1.2, 50.0, 0.9, 1.0]

        ref_sorted: list[float] = []
        ref_fifo: list[float] = []
        ref_frozen = 0

        for step, norm_val in enumerate(norms):
            torch.manual_seed(700 + step)
            g = torch.randn_like(p)
            if math.isnan(norm_val):
                # Also verify NaN in grad does not poison m_slow when gated.
                g_in = torch.full_like(p, float("nan"))
            else:
                g_in = g

            # Reference bisect decision
            ref_spike = not math.isfinite(norm_val)
            if len(ref_sorted) >= min(100, window) and not ref_spike:
                med = ref_sorted[len(ref_sorted) // 2]
                ref_spike = med > 0.0 and norm_val > spike_k * med
            if math.isfinite(norm_val):
                bisect.insort(ref_sorted, norm_val)
                ref_fifo.append(norm_val)
                if len(ref_fifo) > window:
                    ref_sorted.pop(bisect.bisect_left(ref_sorted, ref_fifo.pop(0)))
            ref_frozen += int(ref_spike)

            prev_m_slow = (
                opt.state[p]["pm_m_slow"].clone()
                if "pm_m_slow" in opt.state[p]
                else None
            )
            prev_m_fast = (
                opt.state[p]["pm_m_fast"].clone()
                if "pm_m_fast" in opt.state[p]
                else None
            )

            p.grad = g_in
            # Exercise both set_pre_clip_grad_norm() and step(grad_norm=...).
            if step % 2 == 0:
                opt.set_pre_clip_grad_norm(
                    torch.tensor(norm_val, dtype=torch.float32, device=self.device)
                )
                opt.step()
            else:
                opt.step(grad_norm=norm_val)

            self.assertEqual(
                int(opt.param_groups[0]["pm_spike_frozen_steps"].item()), ref_frozen
            )
            if ref_spike and prev_m_slow is not None:
                # Slow mode must be strictly untouched (even on NaN gradient).
                torch.testing.assert_close(
                    opt.state[p]["pm_m_slow"], prev_m_slow, rtol=0.0, atol=0.0
                )
                self.assertTrue(torch.isfinite(opt.state[p]["pm_m_slow"]).all())
                if math.isfinite(norm_val):
                    # On finite spikes, fast mode still updates as normal.
                    assert prev_m_fast is not None
                    self.assertGreater(
                        (opt.state[p]["pm_m_fast"] - prev_m_fast).abs().max().item(),
                        1e-6,
                    )
                else:
                    # Restore finite weights/fast mode after testing NaN protection
                    # so subsequent steps stay finite.
                    with torch.no_grad():
                        p.zero_()
                        opt.state[p]["pm_m_fast"].zero_()
                        opt.state[p]["momentum_buffer"].zero_()

        self.assertEqual(ref_frozen, 4)

    def test_state_dict_roundtrip_across_switch_and_spike_guard(self) -> None:
        """Checkpoint save/load reproduces uninterrupted PhysMuon trajectory."""
        for shadow in (True, False):
            for split_step in (4, 9):
                with self.subTest(shadow=shadow, split_step=split_step):
                    torch.manual_seed(99)
                    init_w = torch.randn(8, 8, device=self.device)
                    init_b = torch.randn(8, device=self.device)

                    def make_opt(
                        w0: torch.Tensor, b0: torch.Tensor
                    ) -> tuple[list[torch.nn.Parameter], HybridMuonOptimizer]:
                        w = torch.nn.Parameter(w0.clone())
                        b = torch.nn.Parameter(b0.clone())
                        opt = HybridMuonOptimizer(
                            [w, b],
                            lr=1e-2,
                            weight_decay=1e-3,
                            enable_gram=False,
                            physmuon=True,
                            physmuon_start_step=6,
                            physmuon_gate=True,
                            physmuon_shadow=shadow,
                            physmuon_spike_guard=4.0,
                            physmuon_spike_window=8,
                        )
                        return [w, b], opt

                    ref_params, ref_opt = make_opt(init_w, init_b)
                    res_params, res_opt = make_opt(init_w, init_b)

                    grads = []
                    for step in range(14):
                        torch.manual_seed(900 + step)
                        scale = 12.0 if step == 11 else 1.0
                        grads.append(
                            (
                                torch.randn_like(init_w) * scale,
                                torch.randn_like(init_b) * scale,
                            )
                        )

                    for step, (gw, gb) in enumerate(grads):
                        ref_params[0].grad, ref_params[1].grad = gw.clone(), gb.clone()
                        ref_opt.step()

                        if step < split_step:
                            res_params[0].grad = gw.clone()
                            res_params[1].grad = gb.clone()
                            res_opt.step()

                    payload = {
                        "w": res_params[0].detach().clone(),
                        "b": res_params[1].detach().clone(),
                        "opt": res_opt.state_dict(),
                    }
                    res2_params, res2_opt = make_opt(payload["w"], payload["b"])
                    res2_opt.load_state_dict(payload["opt"])

                    for step in range(split_step, len(grads)):
                        gw, gb = grads[step]
                        res2_params[0].grad = gw.clone()
                        res2_params[1].grad = gb.clone()
                        res2_opt.step()

                    for pr, p2 in zip(ref_params, res2_params, strict=True):
                        torch.testing.assert_close(pr, p2, rtol=0.0, atol=0.0)

    def test_legacy_hybrid_muon_checkpoint_loads_into_physmuon(self) -> None:
        """Loading a stock HybridMuon checkpoint into physmuon=True works cleanly."""
        torch.manual_seed(123)
        w = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
        opt_old = HybridMuonOptimizer([w], lr=1e-2, enable_gram=False, physmuon=False)
        for step in range(4):
            torch.manual_seed(1000 + step)
            w.grad = torch.randn_like(w)
            opt_old.step()
        saved_state = opt_old.state_dict()
        saved_momentum = opt_old.state[w]["momentum_buffer"].clone()

        w_new = torch.nn.Parameter(w.detach().clone())
        opt_new = HybridMuonOptimizer(
            [w_new],
            lr=1e-2,
            enable_gram=False,
            physmuon=True,
            physmuon_start_step=4,
            physmuon_gate=True,
            physmuon_shadow=True,
        )
        opt_new.load_state_dict(saved_state)
        opt_new.param_groups[0]["pm_step"] = 4
        self.assertTrue(opt_new.param_groups[0]["physmuon"])
        self.assertEqual(opt_new.param_groups[0]["physmuon_start_step"], 4)

        w_new.grad = torch.randn_like(w_new)
        opt_new.step()
        self.assertIn("pm_m_fast", opt_new.state[w_new])
        self.assertIn("pm_m_slow", opt_new.state[w_new])
        self.assertIn("pm_osc", opt_new.state[w_new])
        self.assertIn("pm_drift", opt_new.state[w_new])
        # Both modes were seeded from saved_momentum (not zeroed).
        expected_fast = torch.lerp(saved_momentum, w_new.grad, 1.0 - 0.90)
        torch.testing.assert_close(
            opt_new.state[w_new]["pm_m_fast"], expected_fast, rtol=1e-6, atol=1e-6
        )

    def test_bucketed_foreach_matches_per_tensor_physmuon(self) -> None:
        """Bucketed multi-tensor PhysMuon fast path matches per-tensor path."""
        for use_gate in (True, False):
            for spike_guard in (0.0, 4.0):
                with self.subTest(use_gate=use_gate, spike_guard=spike_guard):
                    torch.manual_seed(55)
                    init = [
                        torch.randn(8, 8, device=self.device),
                        torch.randn(8, 8, device=self.device),
                        torch.randn(8, 8, device=self.device),
                        torch.randn(4, 16, device=self.device),  # same numel=64!
                        torch.randn(6, 6, device=self.device),  # singleton bucket
                    ]
                    p_vec = [torch.nn.Parameter(x.clone()) for x in init]
                    p_ref = [torch.nn.Parameter(x.clone()) for x in init]
                    kw = {
                        "lr": 1e-2,
                        "weight_decay": 1e-3,
                        "enable_gram": False,
                        "magma_muon": True,
                        "physmuon": True,
                        "physmuon_start_step": 4,
                        "physmuon_gate": use_gate,
                        "physmuon_shadow": True,
                        "physmuon_spike_guard": spike_guard,
                        "physmuon_spike_window": 6,
                    }
                    opt_vec = HybridMuonOptimizer(p_vec, use_foreach=True, **kw)
                    opt_ref = HybridMuonOptimizer(p_ref, use_foreach=False, **kw)
                    for step in range(12):
                        torch.manual_seed(1400 + step)
                        scale = 15.0 if step == 9 else 1.0
                        for pv, pr in zip(p_vec, p_ref, strict=True):
                            g = torch.randn_like(pv) * scale
                            if step % 3 == 1:
                                g = -g
                            pv.grad = g.clone()
                            pr.grad = g.clone()
                        opt_vec.step()
                        opt_ref.step()
                    for pv, pr in zip(p_vec, p_ref, strict=True):
                        torch.testing.assert_close(pv, pr, rtol=1e-6, atol=1e-6)
                        for key in ("momentum_buffer", "pm_m_fast", "pm_m_slow"):
                            torch.testing.assert_close(
                                opt_vec.state[pv][key],
                                opt_ref.state[pr][key],
                                rtol=1e-6,
                                atol=1e-6,
                            )

    def test_adam_routed_parameters_untouched_by_physmuon(self) -> None:
        """Adam-prefixed and adam_patterns 2D/3D params stay bitwise identical."""
        torch.manual_seed(31)
        muon_w1 = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
        adam_pref1 = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
        adamw_pref1 = torch.nn.Parameter(torch.randn(8, 8, device=self.device))
        pat_w1 = torch.nn.Parameter(torch.randn(4, 8, 8, device=self.device))

        muon_w2 = torch.nn.Parameter(muon_w1.detach().clone())
        adam_pref2 = torch.nn.Parameter(adam_pref1.detach().clone())
        adamw_pref2 = torch.nn.Parameter(adamw_pref1.detach().clone())
        pat_w2 = torch.nn.Parameter(pat_w1.detach().clone())

        named1 = [
            ("trunk.linear.weight", muon_w1),
            ("descriptor.adam_radial", adam_pref1),
            ("fitting.adamw_head", adamw_pref1),
            ("equiv_block.lmax_weight", pat_w1),
        ]
        named2 = [
            ("trunk.linear.weight", muon_w2),
            ("descriptor.adam_radial", adam_pref2),
            ("fitting.adamw_head", adamw_pref2),
            ("equiv_block.lmax_weight", pat_w2),
        ]

        opt_base = HybridMuonOptimizer(
            [p for _, p in named1],
            lr=1e-2,
            weight_decay=1e-2,
            enable_gram=False,
            named_parameters=named1,
            adam_patterns=("equiv_block",),
            physmuon=False,
        )
        opt_pm = HybridMuonOptimizer(
            [p for _, p in named2],
            lr=1e-2,
            weight_decay=1e-2,
            enable_gram=False,
            named_parameters=named2,
            adam_patterns=("equiv_block",),
            physmuon=True,
            physmuon_start_step=3,
            physmuon_gate=True,
            physmuon_shadow=True,
        )

        for step in range(8):
            torch.manual_seed(1200 + step)
            grads = [torch.randn_like(p) for _, p in named1]
            for (_, p1), (_, p2), g in zip(named1, named2, grads, strict=True):
                p1.grad = g.clone()
                p2.grad = g.clone()
            opt_base.step()
            opt_pm.step()

        # All Adam-routed 2D/3D params must match baseline bitwise and hold no pm state.
        for p1, p2 in (
            (adam_pref1, adam_pref2),
            (adamw_pref1, adamw_pref2),
            (pat_w1, pat_w2),
        ):
            torch.testing.assert_close(p1, p2, rtol=0.0, atol=0.0)
            self.assertNotIn("pm_m_fast", opt_pm.state[p2])
            self.assertNotIn("pm_m_slow", opt_pm.state[p2])

        # The Muon-routed matrix receives PhysMuon state and diverges after start_step.
        self.assertIn("pm_m_fast", opt_pm.state[muon_w2])
        self.assertIn("pm_m_slow", opt_pm.state[muon_w2])
        self.assertGreater((muon_w1 - muon_w2).abs().max().item(), 1e-6)

    def test_invalid_physmuon_hyperparams_raise(self) -> None:
        """Invalid PhysMuon parameters are rejected eagerly."""
        p = torch.nn.Parameter(torch.randn(4, 4, device=self.device))
        bad_kwargs = [
            {"physmuon_start_step": -1},
            {"physmuon_beta_fast": 1.0},
            {"physmuon_beta_slow": -0.1},
            {"physmuon_gate_beta": 1.5},
            {"physmuon_w_fast": 1.2},
            {"physmuon_w_max": -0.2},
            {"physmuon_spike_guard": -1.0},
            {"physmuon_spike_window": 0},
        ]
        for kw in bad_kwargs:
            with self.subTest(**kw):
                with self.assertRaises(ValueError):
                    HybridMuonOptimizer([p], physmuon=True, **kw)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA graph capture needs CUDA")
class TestHybridMuonPhysMuonCudaGraph(unittest.TestCase):
    """PhysMuon whole-step CUDA graph parity with eager execution."""

    def setUp(self) -> None:
        self.device = torch.device("cuda")
        torch.manual_seed(2026)
        self.inputs = torch.randn(16, 32, device=self.device)
        self.targets = torch.randn(16, 16, device=self.device)

    def _run_trajectory(
        self,
        *,
        graph_on: bool,
        shadow: bool,
        spike_guard: float,
    ) -> tuple[list[torch.Tensor], HybridMuonOptimizer]:
        torch.manual_seed(77)
        model = _MixedRouteModel(self.device)
        opt = HybridMuonOptimizer(
            model.parameters(),
            lr=0.02,
            weight_decay=0.01,
            named_parameters=list(model.named_parameters()),
            physmuon=True,
            physmuon_start_step=4,
            physmuon_gate=True,
            physmuon_shadow=shadow,
            physmuon_spike_guard=spike_guard,
            physmuon_spike_window=6,
        )
        opt._graph_enabled = graph_on
        for step in range(12):
            for group in opt.param_groups:
                group["lr"] = 0.02 * (0.9**step)
            opt.zero_grad(set_to_none=True)
            loss = ((model(self.inputs) - self.targets) ** 2).mean()
            loss.backward()
            if spike_guard > 0.0:
                norm_val = 100.0 if step == 9 else 1.0
                opt.set_pre_clip_grad_norm(
                    torch.tensor(norm_val, dtype=torch.float32, device=self.device)
                )
            opt.step()
        torch.cuda.synchronize()
        return [p.detach().clone() for p in model.parameters()], opt

    def test_physmuon_shadow_graph_matches_eager_without_recapture(self) -> None:
        """Shadow mode crosses start_step on one captured graph, bitwise identical."""
        eager_params, _ = self._run_trajectory(
            graph_on=False, shadow=True, spike_guard=5.0
        )
        graph_params, graph_opt = self._run_trajectory(
            graph_on=True, shadow=True, spike_guard=5.0
        )
        self.assertEqual(len(graph_opt._graphs), 1)
        self.assertEqual(
            int(graph_opt.param_groups[0]["pm_spike_frozen_steps"].item()), 1
        )
        for pe, pg in zip(eager_params, graph_params, strict=True):
            torch.testing.assert_close(pe, pg, rtol=0.0, atol=0.0)

    def test_physmuon_no_shadow_graph_matches_eager(self) -> None:
        """Non-shadow mode cleanly transitions at start_step and recaptures."""
        eager_params, _ = self._run_trajectory(
            graph_on=False, shadow=False, spike_guard=0.0
        )
        graph_params, graph_opt = self._run_trajectory(
            graph_on=True, shadow=False, spike_guard=0.0
        )
        self.assertEqual(len(graph_opt._graphs), 1)
        for pe, pg in zip(eager_params, graph_params, strict=True):
            torch.testing.assert_close(pe, pg, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
