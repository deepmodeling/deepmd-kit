# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import textwrap
import unittest
from copy import (
    deepcopy,
)

import numpy as np

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.utils import (
    EmbeddingNet,
    FittingNet,
    NativeLayer,
    NativeNet,
    NetworkCollection,
    load_dp_model,
    save_dp_model,
)
from deepmd.dpmodel.utils.serialization import (
    Node,
)


class TestNativeLayer(unittest.TestCase):
    def test_serialize_deserize(self) -> None:
        for (
            ni,
            no,
        ), bias, ut, activation_function, resnet, ashp, prec in itertools.product(
            [(5, 5), (5, 10), (5, 9), (9, 5)],
            [True, False],
            [True, False],
            ["tanh", "none"],
            [True, False],
            [None, [4], [3, 2]],
            ["float32", "float64", "single", "double"],
        ):
            nl0 = NativeLayer(
                ni,
                no,
                bias=bias,
                use_timestep=ut,
                activation_function=activation_function,
                resnet=resnet,
                precision=prec,
            )
            nl1 = NativeLayer.deserialize(nl0.serialize())
            inp_shap = [ni]
            if ashp is not None:
                inp_shap = ashp + inp_shap
            inp = np.arange(
                np.prod(inp_shap), dtype=get_xp_precision(np, prec)
            ).reshape(inp_shap)
            np.testing.assert_allclose(nl0.call(inp), nl1.call(inp))

    def test_shape_error(self) -> None:
        self.w0 = np.full((2, 3), 3.0)
        self.b0 = np.full((2,), 4.0)
        self.b1 = np.full((3,), 4.0)
        self.idt0 = np.full((2,), 4.0)
        with self.assertRaises(ValueError) as context:
            NativeLayer.deserialize(
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w0, "b": self.b0},
                }
            )
            assert "not equalt to shape of b" in context.exception
        with self.assertRaises(ValueError) as context:
            NativeLayer.deserialize(
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w0, "b": self.b1, "idt": self.idt0},
                }
            )
            assert "not equalt to shape of idt" in context.exception


class TestNativeNet(unittest.TestCase):
    def setUp(self) -> None:
        self.w0 = np.full((2, 3), 3.0)
        self.b0 = np.full((3,), 4.0)
        self.w1 = np.full((3, 4), 3.0)
        self.b1 = np.full((4,), 4.0)

    def test_serialize(self) -> None:
        network = NativeNet(
            [
                NativeLayer(2, 3).serialize(),
                NativeLayer(3, 4).serialize(),
            ]
        )
        network[1]["w"] = self.w1
        network[1]["b"] = self.b1
        network[0]["w"] = self.w0
        network[0]["b"] = self.b0
        network[1]["activation_function"] = "tanh"
        network[0]["activation_function"] = "tanh"
        network[1]["resnet"] = True
        network[0]["resnet"] = True
        jdata = network.serialize()
        np.testing.assert_array_equal(jdata["layers"][0]["@variables"]["w"], self.w0)
        np.testing.assert_array_equal(jdata["layers"][0]["@variables"]["b"], self.b0)
        np.testing.assert_array_equal(jdata["layers"][1]["@variables"]["w"], self.w1)
        np.testing.assert_array_equal(jdata["layers"][1]["@variables"]["b"], self.b1)
        np.testing.assert_array_equal(jdata["layers"][0]["activation_function"], "tanh")
        np.testing.assert_array_equal(jdata["layers"][1]["activation_function"], "tanh")
        np.testing.assert_array_equal(jdata["layers"][0]["resnet"], True)
        np.testing.assert_array_equal(jdata["layers"][1]["resnet"], True)

    def test_deserialize(self) -> None:
        network = NativeNet.deserialize(
            {
                "layers": [
                    {
                        "activation_function": "tanh",
                        "resnet": True,
                        "@variables": {"w": self.w0, "b": self.b0},
                    },
                    {
                        "activation_function": "tanh",
                        "resnet": True,
                        "@variables": {"w": self.w1, "b": self.b1},
                    },
                ],
            }
        )
        np.testing.assert_array_equal(network[0]["w"], self.w0)
        np.testing.assert_array_equal(network[0]["b"], self.b0)
        np.testing.assert_array_equal(network[1]["w"], self.w1)
        np.testing.assert_array_equal(network[1]["b"], self.b1)
        np.testing.assert_array_equal(network[0]["activation_function"], "tanh")
        np.testing.assert_array_equal(network[1]["activation_function"], "tanh")
        np.testing.assert_array_equal(network[0]["resnet"], True)
        np.testing.assert_array_equal(network[1]["resnet"], True)

    def test_shape_error(self) -> None:
        with self.assertRaises(ValueError) as context:
            NativeNet.deserialize(
                {
                    "layers": [
                        {
                            "activation_function": "tanh",
                            "resnet": True,
                            "@variables": {"w": self.w0, "b": self.b0},
                        },
                        {
                            "activation_function": "tanh",
                            "resnet": True,
                            "@variables": {"w": self.w0, "b": self.b0},
                        },
                    ],
                }
            )
            assert "does not match the dim of layer" in context.exception


class TestEmbeddingNet(unittest.TestCase):
    def test_embedding_net(self) -> None:
        for ni, act, idt, prec in itertools.product(
            [1, 10],
            ["tanh", "none"],
            [True, False],
            ["double", "single"],
        ):
            en0 = EmbeddingNet(
                ni,
                activation_function=act,
                precision=prec,
                resnet_dt=idt,
            )
            en1 = EmbeddingNet.deserialize(en0.serialize())
            inp = np.ones([ni], dtype=get_xp_precision(np, prec))
            np.testing.assert_allclose(en0.call(inp), en1.call(inp))

    def test_is_concrete_class(self) -> None:
        """Verify EmbeddingNet is a concrete class, not factory-generated."""
        in_dim = 4
        neuron = [8, 16, 32]
        net = EmbeddingNet(
            in_dim=in_dim,
            neuron=neuron,
            activation_function="tanh",
            resnet_dt=True,
            precision="float64",
        )
        # Check it's the actual EmbeddingNet class, not a dynamic class
        self.assertEqual(net.__class__.__name__, "EmbeddingNet")
        self.assertEqual(net.__class__.__module__, "deepmd.dpmodel.utils.network")
        # Verify it has the expected attributes
        self.assertEqual(net.in_dim, in_dim)
        self.assertEqual(net.neuron, neuron)
        self.assertEqual(net.activation_function, "tanh")
        self.assertEqual(net.resnet_dt, True)
        self.assertEqual(len(net.layers), len(neuron))

    def test_forward_pass(self) -> None:
        """Test EmbeddingNet forward pass produces correct shapes."""
        in_dim = 4
        neuron = [8, 16, 32]
        net = EmbeddingNet(
            in_dim=in_dim,
            neuron=neuron,
            activation_function="tanh",
            resnet_dt=True,
            precision="float64",
        )
        rng = np.random.default_rng()
        x = rng.standard_normal((5, in_dim))
        out = net.call(x)
        self.assertEqual(out.shape, (5, neuron[-1]))
        self.assertEqual(out.dtype, np.float64)

    def test_trainable_parameter_variants(self) -> None:
        """Test EmbeddingNet with different trainable configurations."""
        in_dim = 4
        neuron = [8, 16]

        # All trainable
        net_trainable = EmbeddingNet(
            in_dim=in_dim,
            neuron=neuron,
            trainable=True,
        )
        for layer in net_trainable.layers:
            self.assertTrue(layer.trainable)

        # All frozen
        net_frozen = EmbeddingNet(
            in_dim=in_dim,
            neuron=neuron,
            trainable=False,
        )
        for layer in net_frozen.layers:
            self.assertFalse(layer.trainable)

        # Mixed trainable
        net_mixed = EmbeddingNet(
            in_dim=in_dim,
            neuron=neuron,
            trainable=[True, False],
        )
        self.assertTrue(net_mixed.layers[0].trainable)
        self.assertFalse(net_mixed.layers[1].trainable)

    def test_empty_layers_round_trip(self) -> None:
        """Test EmbeddingNet with empty neuron list (edge case for deserialize).

        This tests the fix for IndexError when neuron=[] results in empty layers.
        The deserialize method should handle this case without trying to access
        layers[0] when the list is empty.
        """
        in_dim = 4
        neuron = []  # Empty neuron list

        # Create network with empty layers
        net = EmbeddingNet(
            in_dim=in_dim,
            neuron=neuron,
            activation_function="tanh",
            resnet_dt=True,
            precision="float64",
        )

        # Verify it has no layers
        self.assertEqual(len(net.layers), 0)

        # Serialize and deserialize
        serialized = net.serialize()
        net_restored = EmbeddingNet.deserialize(serialized)

        # Verify restored network also has no layers
        self.assertEqual(len(net_restored.layers), 0)
        self.assertEqual(net_restored.in_dim, in_dim)
        self.assertEqual(net_restored.neuron, neuron)

        # Verify forward pass works (should return input unchanged)
        rng = np.random.default_rng()
        x = rng.standard_normal((5, in_dim))
        out = net_restored.call(x)
        # With no layers, output should equal input
        np.testing.assert_allclose(out, x)


class TestFittingNet(unittest.TestCase):
    def test_fitting_net(self) -> None:
        for ni, no, act, idt, prec, bo in itertools.product(
            [1, 10],
            [1, 7],
            ["tanh", "none"],
            [True, False],
            ["double", "single"],
            [True, False],
        ):
            en0 = FittingNet(
                ni,
                no,
                activation_function=act,
                precision=prec,
                resnet_dt=idt,
                bias_out=bo,
            )
            en1 = FittingNet.deserialize(en0.serialize())
            inp = np.ones([ni], dtype=get_xp_precision(np, prec))
            en0.call(inp)
            en1.call(inp)
            np.testing.assert_allclose(en0.call(inp), en1.call(inp))

    def test_is_concrete_class(self) -> None:
        """Verify FittingNet is a concrete class, not factory-generated."""
        in_dim = 4
        out_dim = 1
        neuron = [8, 16]
        net = FittingNet(
            in_dim=in_dim,
            out_dim=out_dim,
            neuron=neuron,
            activation_function="tanh",
            resnet_dt=True,
            precision="float64",
            bias_out=True,
        )
        # Check it's the actual FittingNet class, not a dynamic class
        self.assertEqual(net.__class__.__name__, "FittingNet")
        self.assertEqual(net.__class__.__module__, "deepmd.dpmodel.utils.network")
        # Verify it has the expected attributes
        self.assertEqual(net.in_dim, in_dim)
        self.assertEqual(net.out_dim, out_dim)
        self.assertEqual(net.neuron, neuron)
        self.assertEqual(net.activation_function, "tanh")
        self.assertEqual(net.resnet_dt, True)
        self.assertEqual(net.bias_out, True)
        # FittingNet has len(neuron) embedding layers + 1 output layer
        self.assertEqual(len(net.layers), len(neuron) + 1)

    def test_forward_pass(self) -> None:
        """Test FittingNet forward pass produces correct output shape."""
        in_dim = 4
        out_dim = 3
        neuron = [8, 16, 32]
        net = FittingNet(
            in_dim=in_dim,
            out_dim=out_dim,
            neuron=neuron,
            activation_function="tanh",
            resnet_dt=True,
            precision="float64",
        )
        # Single sample
        rng = np.random.default_rng()
        x = rng.standard_normal(in_dim)
        out = net.call(x)
        self.assertEqual(out.shape, (out_dim,))

        # Batch of samples
        batch_size = 5
        x_batch = rng.standard_normal((batch_size, in_dim))
        out_batch = net.call(x_batch)
        self.assertEqual(out_batch.shape, (batch_size, out_dim))

    def test_trainable_parameter_variants(self) -> None:
        """Test FittingNet with different trainable configurations."""
        in_dim = 4
        out_dim = 2
        neuron = [8, 16]

        # Test 1: All layers trainable (default)
        net_all_trainable = FittingNet(
            in_dim=in_dim,
            out_dim=out_dim,
            neuron=neuron,
            trainable=True,
        )
        for layer in net_all_trainable.layers:
            self.assertTrue(layer.trainable)

        # Test 2: All layers frozen
        net_all_frozen = FittingNet(
            in_dim=in_dim,
            out_dim=out_dim,
            neuron=neuron,
            trainable=False,
        )
        for layer in net_all_frozen.layers:
            self.assertFalse(layer.trainable)

        # Test 3: Mixed trainable (embedding layers frozen, output layer trainable)
        trainable_list = [False, False, True]  # 2 embedding layers + 1 output layer
        net_mixed = FittingNet(
            in_dim=in_dim,
            out_dim=out_dim,
            neuron=neuron,
            trainable=trainable_list,
        )
        self.assertFalse(net_mixed.layers[0].trainable)  # First embedding layer
        self.assertFalse(net_mixed.layers[1].trainable)  # Second embedding layer
        self.assertTrue(net_mixed.layers[2].trainable)  # Output layer

        # Test 4: Serialize/deserialize preserves trainable
        serialized = net_mixed.serialize()
        net_restored = FittingNet.deserialize(serialized)
        for orig_layer, restored_layer in zip(
            net_mixed.layers, net_restored.layers, strict=True
        ):
            self.assertEqual(orig_layer.trainable, restored_layer.trainable)


class TestNetworkCollection(unittest.TestCase):
    def setUp(self) -> None:
        w0 = np.full((2, 3), 3.0)
        b0 = np.full((3,), 4.0)
        w1 = np.full((3, 4), 3.0)
        b1 = np.full((4,), 4.0)
        self.network = {
            "layers": [
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": w0, "b": b0},
                },
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": w1, "b": b1},
                },
            ],
        }

    def test_two_dim(self) -> None:
        networks = NetworkCollection(ndim=2, ntypes=2)
        networks[(0, 0)] = self.network
        networks[(1, 1)] = self.network
        networks[(0, 1)] = self.network
        with self.assertRaises(RuntimeError):
            networks.check_completeness()
        networks[(1, 0)] = self.network
        networks.check_completeness()
        np.testing.assert_equal(
            networks.serialize(),
            NetworkCollection.deserialize(networks.serialize()).serialize(),
        )
        np.testing.assert_equal(
            networks[(0, 0)].serialize(), networks.serialize()["networks"][0]
        )

    def test_one_dim(self) -> None:
        networks = NetworkCollection(ndim=1, ntypes=2)
        networks[(0,)] = self.network
        with self.assertRaises(RuntimeError):
            networks.check_completeness()
        networks[(1,)] = self.network
        networks.check_completeness()
        np.testing.assert_equal(
            networks.serialize(),
            NetworkCollection.deserialize(networks.serialize()).serialize(),
        )
        np.testing.assert_equal(
            networks[(0,)].serialize(), networks.serialize()["networks"][0]
        )

    def test_zero_dim(self) -> None:
        networks = NetworkCollection(ndim=0, ntypes=2)
        networks[()] = self.network
        networks.check_completeness()
        np.testing.assert_equal(
            networks.serialize(),
            NetworkCollection.deserialize(networks.serialize()).serialize(),
        )
        np.testing.assert_equal(
            networks[()].serialize(), networks.serialize()["networks"][0]
        )


class TestSaveLoadDPModel(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.full((3, 2), 3.0)
        self.b = np.full((3,), 4.0)
        self.model_dict = {
            "@class": "some_class",
            "type": "some_type",
            "layers": [
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w, "b": self.b},
                },
                {
                    "activation_function": "tanh",
                    "resnet": True,
                    "@variables": {"w": self.w, "b": self.b},
                },
            ],
        }
        self.filename = "test_dp_dpmodel.dp"
        self.filename_yaml = "test_dp_dpmodel.yaml"

    def test_save_load_model(self) -> None:
        save_dp_model(self.filename, {"model": deepcopy(self.model_dict)})
        model = load_dp_model(self.filename)
        np.testing.assert_equal(model["model"], self.model_dict)
        assert "software" in model
        assert "version" in model

    def test_save_load_model_yaml(self) -> None:
        save_dp_model(self.filename_yaml, {"model": deepcopy(self.model_dict)})
        model = load_dp_model(self.filename_yaml)
        np.testing.assert_equal(model["model"], self.model_dict)
        assert "software" in model
        assert "version" in model

    def test_node_display(self):
        disp_expected = textwrap.dedent("""\
            some_class some_type (size=18)
            └──layers -> ListNode (size=18)
               └──0, 1 -> Node (size=9)""")
        disp = str(Node.deserialize(self.model_dict))
        self.assertEqual(disp, disp_expected)

    def tearDown(self) -> None:
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.filename_yaml):
            os.remove(self.filename_yaml)
