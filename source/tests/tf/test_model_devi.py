# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np

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


class TestMakeModelDevi(unittest.TestCase):
    def setUp(self) -> None:
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
            os.path.join(infer_path, "deeppot.pbtxt"),
            os.path.join(infer_path, "deeppot-1.pbtxt"),
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
        np.testing.assert_almost_equal(model_devi[0][1:8], self.expect[1:8], 6)
        np.testing.assert_almost_equal(model_devi[0][1:8], model_devi[1][1:8], 6)
        self.assertTrue(os.path.isfile(self.output))

    def test_make_model_devi(self) -> None:
        make_model_devi(
            models=self.graph_dirs,
            system=self.data_dir,
            set_prefix="set",
            output=self.output,
            frequency=self.freq,
        )
        x = np.loadtxt(self.output)
        np.testing.assert_allclose(x, self.expect, 6)

    def test_make_model_devi_real_erorr(self) -> None:
        make_model_devi(
            models=self.graph_dirs,
            system=self.data_dir,
            set_prefix="set",
            output=self.output,
            frequency=self.freq,
            real_error=True,
        )
        x = np.loadtxt(self.output)
        np.testing.assert_allclose(
            x,
            np.array(
                [
                    0.000000e00,
                    6.709021e-01,
                    1.634359e-03,
                    3.219720e-01,
                    2.018684e00,
                    1.829748e00,
                    1.956474e00,
                    1.550898e02,
                ]
            ),
            6,
        )

    def test_make_model_devi_atomic_relative(self) -> None:
        _, expected_f, expected_v = self.graphs[0].eval(
            self.coord[0], self.box[0], self.atype
        )
        _, expected_f2, expected_v2 = self.graphs[1].eval(
            self.coord[0], self.box[0], self.atype
        )
        expected_f = expected_f.reshape((-1, 3))
        expected_f2 = expected_f2.reshape((-1, 3))
        expected_v = expected_v.reshape((-1, 3))
        expected_v2 = expected_v2.reshape((-1, 3))
        relative = 1.0
        make_model_devi(
            models=self.graph_dirs,
            system=self.data_dir,
            set_prefix="set",
            output=self.output,
            frequency=self.freq,
            atomic=True,
            relative=relative,
        )
        md = np.loadtxt(self.output)
        # copy from lammps test
        norm = np.linalg.norm(np.mean([expected_f, expected_f2], axis=0), axis=1)
        expected_md_f = np.linalg.norm(
            np.std([expected_f, expected_f2], axis=0), axis=1
        )
        expected_md_f /= norm + relative
        np.testing.assert_allclose(md[8:], expected_md_f, 6)
        np.testing.assert_allclose(md[7], self.expect[7], 6)
        np.testing.assert_allclose(md[4], np.max(expected_md_f), 6)
        np.testing.assert_allclose(md[5], np.min(expected_md_f), 6)
        np.testing.assert_allclose(md[6], np.mean(expected_md_f), 6)
        expected_md_v = (
            np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0)
            / 6
        )
        np.testing.assert_allclose(md[1], np.max(expected_md_v), 6)
        np.testing.assert_allclose(md[2], np.min(expected_md_v), 6)
        np.testing.assert_allclose(md[3], np.sqrt(np.mean(np.square(expected_md_v))), 6)

    def test_make_model_devi_atomic_relative_v(self) -> None:
        _, expected_f, expected_v = self.graphs[0].eval(
            self.coord[0], self.box[0], self.atype
        )
        _, expected_f2, expected_v2 = self.graphs[1].eval(
            self.coord[0], self.box[0], self.atype
        )
        expected_f = expected_f.reshape((-1, 3))
        expected_f2 = expected_f2.reshape((-1, 3))
        expected_v = expected_v.reshape((-1, 3))
        expected_v2 = expected_v2.reshape((-1, 3))
        relative = 1.0
        make_model_devi(
            models=self.graph_dirs,
            system=self.data_dir,
            set_prefix="set",
            output=self.output,
            frequency=self.freq,
            atomic=True,
            relative_v=relative,
        )
        md = np.loadtxt(self.output)
        # copy from lammps test
        expected_md_f = np.linalg.norm(
            np.std([expected_f, expected_f2], axis=0), axis=1
        )
        np.testing.assert_allclose(md[8:], expected_md_f, 6)
        np.testing.assert_allclose(md[7], self.expect[7], 6)
        np.testing.assert_allclose(md[4], np.max(expected_md_f), 6)
        np.testing.assert_allclose(md[5], np.min(expected_md_f), 6)
        np.testing.assert_allclose(md[6], np.mean(expected_md_f), 6)
        expected_md_v = (
            np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0)
            / 6
        )
        norm = (
            np.abs(
                np.mean(
                    [np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0
                )
            )
            / 6
        )
        expected_md_v /= norm + relative
        np.testing.assert_allclose(md[1], np.max(expected_md_v), 6)
        np.testing.assert_allclose(md[2], np.min(expected_md_v), 6)
        np.testing.assert_allclose(md[3], np.sqrt(np.mean(np.square(expected_md_v))), 6)

    def tearDown(self) -> None:
        for pb in self.graph_dirs:
            os.remove(pb)
        os.remove(self.output)
        del_data()


class TestMakeModelDeviFparamAparam(unittest.TestCase):
    """Ensure dp model_devi accepts fparam and aparam."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pbtxts = [
            os.path.join(infer_path, "fparam_aparam.pbtxt"),
        ]
        cls.graph_dirs = [pbtxt.replace("pbtxt", "pb") for pbtxt in cls.pbtxts]
        for pbtxt, pb in zip(cls.pbtxts, cls.graph_dirs):
            convert_pbtxt_to_pb(pbtxt, pb)
        cls.graphs = [DeepPotential(pb) for pb in cls.graph_dirs]

    @classmethod
    def tearDownClass(cls) -> None:
        for pb in cls.graph_dirs:
            os.remove(pb)
        cls.graphs = None

    def setUp(self) -> None:
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
        self.aparam = np.repeat(self.fparam, self.atype.size).reshape(
            (nframes, self.atype.size, 1)
        )

    def test_calc_model_devi(self) -> None:
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

    def test_make_model_devi(self) -> None:
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
        os.remove(self.output)
        del_data()
