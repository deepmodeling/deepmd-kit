# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import sys
import unittest

import numpy as np

from deepmd.infer import (
    DeepPotential,
    calc_model_devi,
)
from deepmd.infer.model_devi import (
    make_model_devi,
)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from common import (
    del_data,
    gen_data,
    tests_path,
)

from deepmd.utils.convert import (
    convert_pbtxt_to_pb,
)


class TestMakeModelDevi(unittest.TestCase):
    def setUp(self):
        gen_data()
        self.data_dir = "system"
        with open(os.path.join(self.data_dir, "type_map.raw"), "w") as f:
            f.write("O\nH")
        coord = np.load(os.path.join(self.data_dir, "set.000/coord.npy"))
        box = np.load(os.path.join(self.data_dir, "set.000/box.npy"))
        self.atype = np.loadtxt(os.path.join(self.data_dir, "type.raw"))
        self.coord = np.vstack([coord, coord])
        self.box = np.vstack([box, box])
        self.freq = 10

        self.pbtxts = [
            os.path.join(tests_path, "infer/deeppot.pbtxt"),
            os.path.join(tests_path, "infer/deeppot-1.pbtxt"),
        ]
        self.graph_dirs = [pbtxt.replace("pbtxt", "pb") for pbtxt in self.pbtxts]
        for pbtxt, pb in zip(self.pbtxts, self.graph_dirs):
            convert_pbtxt_to_pb(pbtxt, pb)
        self.graphs = [DeepPotential(pb) for pb in self.graph_dirs]
        self.output = os.path.join(tests_path, "model_devi.out")
        self.expect = np.array(
            [
                0,
                1.670048e-01,
                4.182279e-04,
                8.048649e-02,
                5.095047e-01,
                4.584241e-01,
                4.819783e-01,
                1.594131e-02,
            ]
        )

    def test_calc_model_devi(self):
        model_devi = calc_model_devi(
            self.coord,
            None,
            self.atype,
            self.graphs,
            frequency=self.freq,
            fname=self.output,
        )
        self.assertAlmostEqual(model_devi[0][0], 0)
        self.assertAlmostEqual(model_devi[1][0], self.freq)
        np.testing.assert_almost_equal(model_devi[0][1:8], self.expect[1:8], 6)
        np.testing.assert_almost_equal(model_devi[0][1:8], model_devi[1][1:8], 6)
        self.assertTrue(os.path.isfile(self.output))

    def test_make_model_devi(self):
        make_model_devi(
            models=self.graph_dirs,
            system=self.data_dir,
            set_prefix="set",
            output=self.output,
            frequency=self.freq,
        )
        x = np.loadtxt(self.output)
        np.testing.assert_allclose(x, self.expect, 6)

    def tearDown(self):
        for pb in self.graph_dirs:
            os.remove(pb)
        os.remove(self.output)
        del_data()


class TestMakeModelDeviFparamAparam(unittest.TestCase):
    """Ensure dp model_devi accepts fparam and aparam."""

    @classmethod
    def setUpClass(cls):
        cls.pbtxts = [
            os.path.join(tests_path, "infer/fparam_aparam.pbtxt"),
        ]
        cls.graph_dirs = [pbtxt.replace("pbtxt", "pb") for pbtxt in cls.pbtxts]
        for pbtxt, pb in zip(cls.pbtxts, cls.graph_dirs):
            convert_pbtxt_to_pb(pbtxt, pb)
        cls.graphs = [DeepPotential(pb) for pb in cls.graph_dirs]

    @classmethod
    def tearDownClass(cls):
        for pb in cls.graph_dirs:
            os.remove(pb)
        cls.graphs = None

    def setUp(self):
        gen_data(dim_fparam=1)
        self.data_dir = "system"
        coord = np.load(os.path.join(self.data_dir, "set.000/coord.npy"))
        box = np.load(os.path.join(self.data_dir, "set.000/box.npy"))
        atype_ = np.loadtxt(os.path.join(self.data_dir, "type.raw"))
        self.atype = np.zeros_like(atype_)
        np.savetxt(os.path.join(self.data_dir, "type.raw"), self.atype)
        self.coord = np.vstack([coord, coord])
        self.box = np.vstack([box, box])
        self.freq = 10

        self.output = os.path.join(tests_path, "model_devi.out")
        self.expect = np.zeros(8)
        nframes = self.box.size // 9
        self.fparam = np.repeat([0.25852028], nframes).reshape((nframes, 1))
        self.aparam = np.repeat(self.fparam, self.atype.size).reshape((nframes, self.atype.size, 1))

    def test_calc_model_devi(self):
        model_devi = calc_model_devi(
            self.coord,
            None,
            self.atype,
            self.graphs,
            frequency=self.freq,
            fname=self.output,
            fparam=self.fparam,
            aparam=self.aparam,
        )
        self.assertAlmostEqual(model_devi[0][0], 0)
        self.assertAlmostEqual(model_devi[1][0], self.freq)
        np.testing.assert_almost_equal(model_devi[0][1:8], self.expect[1:8], 6)
        np.testing.assert_almost_equal(model_devi[0][1:8], model_devi[1][1:8], 6)
        self.assertTrue(os.path.isfile(self.output))

    def test_make_model_devi(self):
        make_model_devi(
            models=self.graph_dirs,
            system=self.data_dir,
            set_prefix="set",
            output=self.output,
            frequency=self.freq,
        )
        x = np.loadtxt(self.output)
        np.testing.assert_allclose(x, self.expect, 6)

    def tearDown(self):
        os.remove(self.output)
        del_data()
