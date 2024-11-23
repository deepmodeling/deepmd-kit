# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
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

    def tearDown(self) -> None:
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.filename_yaml):
            os.remove(self.filename_yaml)
