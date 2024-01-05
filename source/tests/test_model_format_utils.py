# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest
from copy import (
    deepcopy,
)

import numpy as np

from deepmd_utils.model_format import (
    NativeLayer,
    NativeNet,
    load_dp_model,
    save_dp_model,
)


class TestNativeLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.full((2, 3), 3.0)
        self.b = np.full((3,), 4.0)
        self.idt = np.full((3,), 5.0)

    def test_serialize_deserize(self):
        for ww, bb, idt, activation_function, resnet in itertools.product(
            [self.w], [self.b, None], [self.idt, None], ["tanh", "none"], [True, False]
        ):
            nl0 = NativeLayer(ww, bb, idt, activation_function, resnet)
            nl1 = NativeLayer.deserialize(nl0.serialize())
            inp = np.arange(self.w.shape[0])
            np.testing.assert_allclose(nl0.call(inp), nl1.call(inp))


class TestNativeNet(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.full((2, 3), 3.0)
        self.b = np.full((3,), 4.0)
        self.idt = np.full((3,), 5.0)

    def test_serialize(self):
        network = NativeNet()
        network[1]["w"] = self.w
        network[1]["b"] = self.b
        network[0]["w"] = self.w
        network[0]["b"] = self.b
        network[1]["activation_function"] = "tanh"
        network[0]["activation_function"] = "tanh"
        network[1]["resnet"] = True
        network[0]["resnet"] = True
        jdata = network.serialize()
        np.testing.assert_array_equal(jdata["layers"][0]["@variables"]["w"], self.w)
        np.testing.assert_array_equal(jdata["layers"][0]["@variables"]["b"], self.b)
        np.testing.assert_array_equal(jdata["layers"][1]["@variables"]["w"], self.w)
        np.testing.assert_array_equal(jdata["layers"][1]["@variables"]["b"], self.b)
        np.testing.assert_array_equal(jdata["layers"][0]["activation_function"], "tanh")
        np.testing.assert_array_equal(jdata["layers"][1]["activation_function"], "tanh")
        np.testing.assert_array_equal(jdata["layers"][0]["resnet"], True)
        np.testing.assert_array_equal(jdata["layers"][1]["resnet"], True)

    def test_deserialize(self):
        network = NativeNet.deserialize(
            {
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
        )
        np.testing.assert_array_equal(network[0]["w"], self.w)
        np.testing.assert_array_equal(network[0]["b"], self.b)
        np.testing.assert_array_equal(network[1]["w"], self.w)
        np.testing.assert_array_equal(network[1]["b"], self.b)
        np.testing.assert_array_equal(network[0]["activation_function"], "tanh")
        np.testing.assert_array_equal(network[1]["activation_function"], "tanh")
        np.testing.assert_array_equal(network[0]["resnet"], True)
        np.testing.assert_array_equal(network[1]["resnet"], True)


class TestDPModel(unittest.TestCase):
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
        self.filename = "test_dp_model_format.dp"

    def test_save_load_model(self):
        save_dp_model(self.filename, deepcopy(self.model_dict))
        model = load_dp_model(self.filename)
        np.testing.assert_equal(model["model"], self.model_dict)
        assert "software" in model
        assert "version" in model

    def tearDown(self) -> None:
        if os.path.exists(self.filename):
            os.remove(self.filename)
