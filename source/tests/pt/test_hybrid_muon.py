# SPDX-License-Identifier: LGPL-3.0-or-later
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


if __name__ == "__main__":
    unittest.main()
