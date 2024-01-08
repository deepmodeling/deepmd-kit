# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest
from copy import (
    deepcopy,
)

import numpy as np

from deepmd_utils.model_format import (
    EnvMat,
    EmbeddingNet,
    NativeLayer,
    NativeNet,
    load_dp_model,
    save_dp_model,
)


class TestNativeLayer(unittest.TestCase):
    def test_serialize_deserize(self):
        for (ni, no), bias, ut, activation_function, resnet, ashp, prec in itertools.product(
            [(5, 5), (5, 10), (5, 9), (9, 5)],
            [True, False],
            [True, False],
            ["tanh", "none"],
            [True, False],
            [None, [4], [3, 2]],
            ["float32", "float64", "default"],
        ):
            ww = np.full((ni, no), 3.0)
            bb = np.full((no,), 4.0) if bias else None
            idt = np.full((no,), 5.0) if ut else None
            nl0 = NativeLayer(ww, bb, idt, activation_function, resnet, prec)
            nl1 = NativeLayer.deserialize(nl0.serialize())
            inp_shap = [ww.shape[0]]
            if ashp is not None:
                inp_shap = ashp + inp_shap
            inp = np.arange(np.prod(inp_shap)).reshape(inp_shap)
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

    def test_embedding_net(self):
        for ni, idt, act in itertools.product(
            [1, 10],
            [True, False],
            ["tanh", "none"],
        ):
            en0 = EmbeddingNet(ni)
            en1 = EmbeddingNet.deserialize(en0.serialize())
            inp = np.ones([ni])
            np.testing.assert_allclose(en0.call(inp), en1.call(inp))


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


class TestEnvMat(unittest.TestCase):
  def setUp(self):
    # nloc == 3, nall == 4
    self.nloc = 3
    self.nall = 4
    self.coord_ext = np.array(
      [ [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, -2, 0],
       ], 
      dtype = np.float64,
    ).reshape([1, self.nall*3])
    self.atype_ext = np.array(
      [0, 0, 1, 0], dtype=int
    ).reshape([1, self.nall])
    # sel = [5, 2]
    self.nlist = np.array(
      [
        [1, 3, -1, -1, -1, 2, -1],
        [0,-1, -1, -1, -1, 2, -1],
        [0, 1, -1, -1, -1, 0, -1],
      ], 
      dtype=int,
    ).reshape([1, self.nloc, 7])
    self.rcut = .4
    self.rcut_smth = 2.2

  def test_self_consistency(
      self,
  ):
    em0 = EnvMat(self.rcut, self.rcut_smth)
    em1 = EnvMat.deserialize(em0.serialize())
    mm0, ww0 = em0.call(self.nlist, self.coord_ext)
    mm1, ww1 = em1.call(self.nlist, self.coord_ext)
    np.testing.assert_allclose(mm0, mm1)
    np.testing.assert_allclose(ww0, ww1)
