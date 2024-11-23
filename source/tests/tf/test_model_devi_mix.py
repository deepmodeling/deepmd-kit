# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.env import (
    tf,
)
from deepmd.tf.infer import (
    DeepPotential,
    calc_model_devi,
)
from deepmd.tf.infer.model_devi import (
    make_model_devi,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

from .common import (
    del_data,
    gen_data,
    infer_path,
    tests_path,
)


@unittest.skipIf(
    parse_version(tf.__version__) < parse_version("1.15"),
    f"The current tf version {tf.__version__} is too low to run the new testing model.",
)
class TestMakeModelDeviMix(unittest.TestCase):
    def setUp(self) -> None:
        gen_data()
        self.data_dir = "system"
        with open(os.path.join(self.data_dir, "type_map.raw"), "w") as f:
            f.write("O\nH")
        coord = np.load(os.path.join(self.data_dir, "set.000/coord.npy"))
        box = np.load(os.path.join(self.data_dir, "set.000/box.npy"))
        self.atype = np.loadtxt(os.path.join(self.data_dir, "type.raw"))
        self.mixed_atype = np.vstack([self.atype, self.atype])
        self.mixed_atype[1][0] = 1.0
        self.mixed_atype[1][-1] = 0.0
        self.coord = np.vstack([coord, coord])
        self.box = np.vstack([box, box])
        self.freq = 10

        np.save(os.path.join(self.data_dir, "set.000/coord.npy"), self.coord)
        np.save(os.path.join(self.data_dir, "set.000/box.npy"), self.box)
        np.save(
            os.path.join(self.data_dir, "set.000/real_atom_types.npy"), self.mixed_atype
        )

        self.pbtxts = [
            os.path.join(infer_path, "se_atten_no_atten_1.pbtxt"),
            os.path.join(infer_path, "se_atten_no_atten_2.pbtxt"),
        ]
        self.graph_dirs = [pbtxt.replace("pbtxt", "pb") for pbtxt in self.pbtxts]
        for pbtxt, pb in zip(self.pbtxts, self.graph_dirs):
            convert_pbtxt_to_pb(pbtxt, pb)
        self.graphs = [DeepPotential(pb) for pb in self.graph_dirs]
        self.output = os.path.join(tests_path, "model_devi.out")
        self.expect = np.array(
            [
                [
                    0,
                    1.83881698e-01,
                    1.70085937e-04,
                    6.76134909e-02,
                    2.60789586e-01,
                    8.95047307e-02,
                    1.88914900e-01,
                    2.194880e-01,
                ],
                [
                    self.freq,
                    2.01793650e-01,
                    2.15067866e-04,
                    7.92811595e-02,
                    3.80715927e-01,
                    1.88116279e-01,
                    2.76809413e-01,
                    2.045956e-01,
                ],
            ]
        )

    def test_calc_model_devi(self) -> None:
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
        np.testing.assert_almost_equal(model_devi[0][1:8], self.expect[0][1:8], 6)
        np.testing.assert_almost_equal(model_devi[0][1:8], model_devi[1][1:8], 6)
        self.assertTrue(os.path.isfile(self.output))

    def test_calc_model_devi_mixed(self) -> None:
        model_devi = calc_model_devi(
            self.coord,
            None,
            self.mixed_atype,
            self.graphs,
            frequency=self.freq,
            fname=self.output,
            mixed_type=True,
        )
        self.assertAlmostEqual(model_devi[0][0], 0)
        self.assertAlmostEqual(model_devi[1][0], self.freq)
        np.testing.assert_almost_equal(model_devi[0][1:8], self.expect[0][1:8], 6)
        np.testing.assert_almost_equal(model_devi[1][1:8], self.expect[1][1:8], 6)
        self.assertTrue(os.path.isfile(self.output))

    def test_make_model_devi_mixed(self) -> None:
        make_model_devi(
            models=self.graph_dirs,
            system=self.data_dir,
            set_prefix="set",
            output=self.output,
            frequency=self.freq,
        )
        x = np.loadtxt(self.output)
        np.testing.assert_allclose(x, self.expect, 6)

    def tearDown(self) -> None:
        for pb in self.graph_dirs:
            os.remove(pb)
        os.remove(self.output)
        del_data()
