# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest
from copy import (
    deepcopy,
)

import numpy as np

from deepmd_utils.model_format import (
    NativeNet,
    load_dp_model,
    save_dp_model,
)


class TestNativeNet(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.full((3, 2), 3.0)
        self.b = np.full((3,), 4.0)

    def test_serialize(self):
        network = NativeNet()
        network[1]["w"] = self.w
        network[1]["b"] = self.b
        network[0]["w"] = self.w
        network[0]["b"] = self.b
        jdata = network.serialize()
        np.testing.assert_array_equal(jdata["layers"][0]["w"], self.w)
        np.testing.assert_array_equal(jdata["layers"][0]["b"], self.b)
        np.testing.assert_array_equal(jdata["layers"][1]["w"], self.w)
        np.testing.assert_array_equal(jdata["layers"][1]["b"], self.b)

    def test_deserialize(self):
        network = NativeNet.deserialize(
            {
                "layers": [
                    {"w": self.w, "b": self.b},
                    {"w": self.w, "b": self.b},
                ]
            }
        )
        np.testing.assert_array_equal(network[0]["w"], self.w)
        np.testing.assert_array_equal(network[0]["b"], self.b)
        np.testing.assert_array_equal(network[1]["w"], self.w)
        np.testing.assert_array_equal(network[1]["b"], self.b)


class TestDPModel(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.full((3, 2), 3.0)
        self.b = np.full((3,), 4.0)
        self.model_dict = {
            "type": "some_type",
            "@variables": {
                "layers": [
                    {"w": self.w, "b": self.b},
                    {"w": self.w, "b": self.b},
                ]
            },
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
