# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.utils.network import (
    get_activation_fn,
)
from deepmd.pt_expt.utils.network import (
    _torch_activation,
)


class TestSilutActivation:
    """Tests for silut activation in _torch_activation."""

    def setup_method(self) -> None:
        # x values spanning both branches: below threshold and above
        self.x_np = np.array(
            [-5.0, -1.0, 0.0, 1.0, 2.5, 3.0, 5.0, 10.0, 15.0, 20.0],
            dtype=np.float64,
        )
        self.x_torch = torch.tensor(self.x_np, dtype=torch.float64)

    def test_silut_with_threshold(self) -> None:
        """silut:10.0 matches dpmodel numerically."""
        result = _torch_activation(self.x_torch, "silut:10.0")
        dp_fn = get_activation_fn("silut:10.0")
        expected = dp_fn(self.x_np)
        np.testing.assert_allclose(
            result.detach().numpy(), expected, rtol=1e-12, atol=1e-12
        )

    def test_silut_default_threshold(self) -> None:
        """Silut without parameter uses default threshold 3.0."""
        result = _torch_activation(self.x_torch, "silut")
        dp_fn = get_activation_fn("silut")
        expected = dp_fn(self.x_np)
        np.testing.assert_allclose(
            result.detach().numpy(), expected, rtol=1e-12, atol=1e-12
        )

    def test_silut_custom_silu_alias(self) -> None:
        """custom_silu:5.0 is an alias for silut:5.0."""
        result = _torch_activation(self.x_torch, "custom_silu:5.0")
        dp_fn = get_activation_fn("custom_silu:5.0")
        expected = dp_fn(self.x_np)
        np.testing.assert_allclose(
            result.detach().numpy(), expected, rtol=1e-12, atol=1e-12
        )

    def test_silut_gradient(self) -> None:
        """Gradient flows through both branches of silut."""
        x = self.x_torch.clone().requires_grad_(True)
        y = _torch_activation(x, "silut:3.0")
        loss = y.sum()
        loss.backward()
        grad = x.grad
        assert grad is not None
        # gradient should be finite everywhere
        assert torch.all(torch.isfinite(grad))
        # gradient should be non-zero for non-zero inputs
        nonzero_mask = self.x_np != 0.0
        assert torch.all(grad[nonzero_mask] != 0.0)

    def test_silut_make_fx(self) -> None:
        """make_fx can trace through silut activation."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            return _torch_activation(x, "silut:10.0")

        traced = make_fx(fn)(self.x_torch)
        result = traced(self.x_torch)
        expected = _torch_activation(self.x_torch, "silut:10.0")
        np.testing.assert_allclose(
            result.detach().numpy(), expected.detach().numpy(), rtol=1e-12, atol=1e-12
        )

    def test_silut_below_threshold_is_silu(self) -> None:
        """Below threshold, silut equals silu exactly."""
        threshold = 10.0
        x_below = torch.tensor([-5.0, 0.0, 1.0, 5.0, 9.9], dtype=torch.float64)
        result = _torch_activation(x_below, "silut:10.0")
        silu = x_below * torch.sigmoid(x_below)
        np.testing.assert_allclose(
            result.detach().numpy(), silu.detach().numpy(), rtol=1e-14, atol=1e-14
        )

    def test_silut_above_threshold_is_tanh_branch(self) -> None:
        """Above threshold, silut equals tanh(slope*(x-T))+const."""
        import math

        threshold = 3.0
        sig_t = 1.0 / (1.0 + math.exp(-threshold))
        slope = sig_t + threshold * sig_t * (1.0 - sig_t)
        const = threshold * sig_t

        x_above = torch.tensor([3.5, 5.0, 10.0, 20.0], dtype=torch.float64)
        result = _torch_activation(x_above, "silut:3.0")
        expected = torch.tanh(slope * (x_above - threshold)) + const
        np.testing.assert_allclose(
            result.detach().numpy(), expected.detach().numpy(), rtol=1e-14, atol=1e-14
        )

    def test_silut_export(self) -> None:
        """torch.export.export can trace through silut activation."""

        class SilutModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return _torch_activation(x, "silut:10.0")

        mod = SilutModule()
        exported = torch.export.export(mod, (self.x_torch,))
        result = exported.module()(self.x_torch)
        expected = _torch_activation(self.x_torch, "silut:10.0")
        np.testing.assert_allclose(
            result.detach().numpy(), expected.detach().numpy(), rtol=1e-12, atol=1e-12
        )
