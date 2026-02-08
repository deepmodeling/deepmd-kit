# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.utils.network import EmbeddingNet as DPEmbeddingNet
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.network import (
    EmbeddingNet,
    NativeLayer,
)

from ...seed import (
    GLOBAL_SEED,
)


def test_native_layer_clears_parameter_on_none() -> None:
    layer = NativeLayer(2, 3, trainable=True)
    assert layer.w is not None
    layer.w = None
    assert layer.w is None
    assert layer._parameters.get("w") is None


def test_native_layer_clears_buffer_on_none() -> None:
    layer = NativeLayer(2, 3, trainable=False)
    assert layer.w is not None
    layer.w = None
    assert layer.w is None
    assert layer._buffers.get("w") is None


class TestEmbeddingNetRefactor(unittest.TestCase):
    """Tests for the refactored EmbeddingNet pt_expt wrapper and integration."""

    def setUp(self) -> None:
        self.in_dim = 4
        self.neuron = [8, 16, 32]
        self.activation = "tanh"
        self.resnet_dt = True
        self.precision = "float64"

    def test_pt_expt_embedding_net_wraps_dpmodel(self) -> None:
        """Verify pt_expt EmbeddingNet correctly wraps dpmodel."""
        net = EmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )
        # Check it's a torch.nn.Module
        self.assertIsInstance(net, torch.nn.Module)
        # Check it's also a DPEmbeddingNet
        self.assertIsInstance(net, DPEmbeddingNet)
        # Check layers are converted to pt_expt NativeLayer (torch modules)
        self.assertIsInstance(net.layers, torch.nn.ModuleList)
        for layer in net.layers:
            self.assertIsInstance(layer, NativeLayer)
            self.assertIsInstance(layer, torch.nn.Module)

    def test_pt_expt_embedding_net_forward(self) -> None:
        """Test pt_expt EmbeddingNet forward pass returns torch.Tensor."""
        net = EmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )
        x = torch.randn(5, self.in_dim, dtype=torch.float64, device=env.DEVICE)
        out = net(x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (5, self.neuron[-1]))
        self.assertEqual(out.dtype, torch.float64)

    def test_serialization_round_trip_pt_expt(self) -> None:
        """Test pt_expt EmbeddingNet serialization/deserialization."""
        net = EmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )
        x = torch.randn(5, self.in_dim, dtype=torch.float64, device=env.DEVICE)
        out1 = net(x)

        # Serialize and deserialize
        serialized = net.serialize()
        net2 = EmbeddingNet.deserialize(serialized)

        # Verify layers are still pt_expt NativeLayer modules
        self.assertIsInstance(net2.layers, torch.nn.ModuleList)
        for layer in net2.layers:
            self.assertIsInstance(layer, NativeLayer)

        out2 = net2(x)
        np.testing.assert_allclose(
            out1.detach().cpu().numpy(),
            out2.detach().cpu().numpy(),
        )

    def test_deserialize_preserves_layer_type(self) -> None:
        """Test that deserialize uses type(obj.layers[0]) to preserve subclass layers.

        This is the key fix: dpmodel's deserialize no longer hardcodes
        super(EmbeddingNet, obj).__init__(layers), which would overwrite
        pt_expt's converted layers. Instead it uses type(obj.layers[0])
        to respect the subclass's layer type.
        """
        # Create pt_expt EmbeddingNet
        net = EmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )

        # Verify layers are pt_expt NativeLayer (torch modules)
        for layer in net.layers:
            self.assertIsInstance(layer, torch.nn.Module)
            self.assertTrue(hasattr(layer, "_parameters"))

        # Deserialize
        serialized = net.serialize()
        net2 = EmbeddingNet.deserialize(serialized)

        # Verify deserialized layers are STILL pt_expt NativeLayer, not dpmodel
        for layer in net2.layers:
            self.assertIsInstance(layer, torch.nn.Module)
            self.assertTrue(hasattr(layer, "_parameters"))
            # This would fail if deserialize used hardcoded dpmodel layers
            self.assertIsInstance(layer, NativeLayer)

    def test_cross_backend_consistency(self) -> None:
        """Test numerical consistency between dpmodel and pt_expt EmbeddingNet."""
        # Create both with same seed
        dp_net = DPEmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )
        pt_net = EmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )

        # Test forward pass
        rng = np.random.default_rng()
        x_np = rng.standard_normal((5, self.in_dim))
        x_torch = torch.from_numpy(x_np)

        out_dp = dp_net.call(x_np)
        out_pt = pt_net(x_torch).detach().cpu().numpy()

        np.testing.assert_allclose(out_dp, out_pt, rtol=1e-10, atol=1e-10)

    def test_registry_converts_dpmodel_to_pt_expt(self) -> None:
        """Test that the registry auto-converts dpmodel EmbeddingNet to pt_expt."""
        from deepmd.pt_expt.common import (
            try_convert_module,
        )

        # Create dpmodel EmbeddingNet
        dp_net = DPEmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )

        # Try to convert via registry
        converted = try_convert_module(dp_net)

        # Should return pt_expt EmbeddingNet
        self.assertIsNotNone(converted)
        self.assertIsInstance(converted, torch.nn.Module)
        self.assertIsInstance(converted, EmbeddingNet)

        # Verify layers are pt_expt NativeLayer
        for layer in converted.layers:
            self.assertIsInstance(layer, NativeLayer)
            self.assertIsInstance(layer, torch.nn.Module)

    def test_auto_conversion_in_setattr(self) -> None:
        """Test that dpmodel_setattr auto-converts EmbeddingNet attributes."""
        from deepmd.pt_expt.common import (
            dpmodel_setattr,
        )

        # Create a simple torch module
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = None

        obj = TestModule()

        # Create dpmodel EmbeddingNet
        dp_net = DPEmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )

        # Use dpmodel_setattr to set it
        handled, value = dpmodel_setattr(obj, "embedding_net", dp_net)

        # Should not be handled (returns converted value for caller to set)
        self.assertFalse(handled)
        # Value should be converted to pt_expt EmbeddingNet
        self.assertIsInstance(value, torch.nn.Module)
        self.assertIsInstance(value, EmbeddingNet)

    def test_trainable_parameter_handling(self) -> None:
        """Test that trainable parameters work correctly in pt_expt."""
        # Test with trainable=True
        net_trainable = EmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            trainable=True,
            seed=GLOBAL_SEED,
        )

        # Count trainable parameters
        param_count = sum(
            p.numel() for p in net_trainable.parameters() if p.requires_grad
        )
        self.assertGreater(param_count, 0)

        # Check all layer parameters are trainable
        for layer in net_trainable.layers:
            if layer.w is not None:
                self.assertTrue(layer.w.requires_grad)
            if layer.b is not None:
                self.assertTrue(layer.b.requires_grad)

        # Test with trainable=False
        net_frozen = EmbeddingNet(
            in_dim=self.in_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            trainable=False,
            seed=GLOBAL_SEED,
        )

        # Count trainable parameters (should be 0)
        param_count_frozen = sum(
            p.numel() for p in net_frozen.parameters() if p.requires_grad
        )
        self.assertEqual(param_count_frozen, 0)

        # Check all layer weights are buffers, not parameters
        for layer in net_frozen.layers:
            if layer.w is not None:
                self.assertFalse(layer.w.requires_grad)


class TestFittingNetRefactor(unittest.TestCase):
    """Tests for the refactored FittingNet pt_expt wrapper."""

    def setUp(self) -> None:
        self.in_dim = 4
        self.out_dim = 1
        self.neuron = [8, 16]
        self.activation = "tanh"
        self.resnet_dt = True
        self.precision = "float64"

    def test_pt_expt_fitting_net_wraps_dpmodel(self) -> None:
        """Verify pt_expt FittingNet correctly wraps dpmodel."""
        from deepmd.pt_expt.utils.network import (
            FittingNet,
        )

        net = FittingNet(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )
        # Check it's a torch.nn.Module
        self.assertIsInstance(net, torch.nn.Module)
        # Check layers are converted to pt_expt NativeLayer (torch modules)
        self.assertIsInstance(net.layers, torch.nn.ModuleList)
        for layer in net.layers:
            self.assertIsInstance(layer, torch.nn.Module)

    def test_pt_expt_fitting_net_forward(self) -> None:
        """Test pt_expt FittingNet forward pass returns torch.Tensor."""
        from deepmd.pt_expt.utils.network import (
            FittingNet,
        )

        net = FittingNet(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )
        x = torch.randn(5, self.in_dim, dtype=torch.float64, device=env.DEVICE)
        out = net(x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (5, self.out_dim))
        self.assertEqual(out.dtype, torch.float64)

    def test_serialization_round_trip_pt_expt(self) -> None:
        """Test pt_expt FittingNet serialization/deserialization."""
        from deepmd.pt_expt.utils.network import (
            FittingNet,
        )

        net = FittingNet(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )
        x = torch.randn(5, self.in_dim, dtype=torch.float64, device=env.DEVICE)
        out1 = net(x)

        # Serialize and deserialize
        serialized = net.serialize()
        net2 = FittingNet.deserialize(serialized)

        # Verify layers are still pt_expt NativeLayer modules
        self.assertIsInstance(net2.layers, torch.nn.ModuleList)
        for layer in net2.layers:
            self.assertIsInstance(layer, torch.nn.Module)

        out2 = net2(x)
        np.testing.assert_allclose(
            out1.detach().cpu().numpy(),
            out2.detach().cpu().numpy(),
        )

    def test_registry_converts_dpmodel_to_pt_expt(self) -> None:
        """Test that dpmodel FittingNet can be converted to pt_expt via registry."""
        from deepmd.dpmodel.utils.network import FittingNet as DPFittingNet
        from deepmd.pt_expt.common import (
            try_convert_module,
        )
        from deepmd.pt_expt.utils.network import (
            FittingNet,
        )

        # Create dpmodel FittingNet
        dp_net = DPFittingNet(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            neuron=self.neuron,
            activation_function=self.activation,
            resnet_dt=self.resnet_dt,
            precision=self.precision,
            seed=GLOBAL_SEED,
        )

        # Try to convert via registry
        converted = try_convert_module(dp_net)

        # Should return pt_expt FittingNet
        self.assertIsNotNone(converted)
        self.assertIsInstance(converted, torch.nn.Module)
        self.assertIsInstance(converted, FittingNet)

        # Verify layers are pt_expt modules
        for layer in converted.layers:
            self.assertIsInstance(layer, torch.nn.Module)
