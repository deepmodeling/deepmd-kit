# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import ase
import numpy as np

from deepmd.infer.deep_eval import (
    DeepEval,
)

from ..consistent.common import (
    parameterized,
)
from .case import (
    get_cases,
)

default_places = 10


@parameterized(
    tuple(get_cases().keys()),  # key
    (".pb", ".pth"),  # model extension
)
class TestDeepPot(unittest.TestCase):
    # moved from tests/tf/test_deeppot_a,py

    @classmethod
    def setUpClass(cls):
        key, extension = cls.param
        cls.case = get_cases()[key]
        model_name = cls.case.get_model(extension)
        cls.dp = DeepEval(model_name)

    @classmethod
    def tearDownClass(cls):
        cls.dp = None

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 2)
        self.assertAlmostEqual(self.dp.get_rcut(), 6.0, places=default_places)
        self.assertEqual(self.dp.get_type_map(), ["O", "H"])
        self.assertEqual(self.dp.get_dim_fparam(), 0)
        self.assertEqual(self.dp.get_dim_aparam(), 0)

    def test_1frame(self):
        for ii, result in enumerate(self.case.results):
            ee, ff, vv = self.dp.eval(
                result.coord, result.box, result.atype, atomic=False
            )
            # check shape of the returns
            nframes = 1
            natoms = len(result.atype)
            self.assertEqual(ee.shape, (nframes, 1))
            self.assertEqual(ff.shape, (nframes, natoms, 3))
            self.assertEqual(vv.shape, (nframes, 9))
            # check values
            np.testing.assert_almost_equal(
                ff.ravel(),
                result.force.ravel(),
                default_places,
                err_msg=f"Result {ii} force",
            )
            expected_se = np.sum(result.atomic_energy.reshape([nframes, -1]), axis=1)
            np.testing.assert_almost_equal(
                ee.ravel(),
                expected_se.ravel(),
                default_places,
                err_msg=f"Result {ii} energy",
            )
            expected_sv = np.sum(result.atomic_virial.reshape([nframes, -1, 9]), axis=1)
            np.testing.assert_almost_equal(
                vv.ravel(),
                expected_sv.ravel(),
                default_places,
                err_msg=f"Result {ii} virial",
            )

    def test_1frame_atm(self):
        for ii, result in enumerate(self.case.results):
            ee, ff, vv, ae, av = self.dp.eval(
                result.coord, result.box, result.atype, atomic=True
            )
            # check shape of the returns
            nframes = 1
            natoms = len(result.atype)
            self.assertEqual(ee.shape, (nframes, 1))
            self.assertEqual(ff.shape, (nframes, natoms, 3))
            self.assertEqual(vv.shape, (nframes, 9))
            self.assertEqual(ae.shape, (nframes, natoms, 1))
            self.assertEqual(av.shape, (nframes, natoms, 9))
            # check values
            np.testing.assert_almost_equal(
                ff.ravel(),
                result.force.ravel(),
                default_places,
                err_msg=f"Result {ii} force",
            )
            np.testing.assert_almost_equal(
                ae.ravel(),
                result.atomic_energy.ravel(),
                default_places,
                err_msg=f"Result {ii} atomic energy",
            )
            np.testing.assert_almost_equal(
                av.ravel(),
                result.atomic_virial.ravel(),
                default_places,
                err_msg=f"Result {ii} atomic virial",
            )
            expected_se = np.sum(result.energy.reshape([nframes, -1]), axis=1)
            np.testing.assert_almost_equal(
                ee.ravel(),
                expected_se.ravel(),
                default_places,
                err_msg=f"Result {ii} energy",
            )
            expected_sv = np.sum(result.virial.reshape([nframes, -1, 9]), axis=1)
            np.testing.assert_almost_equal(
                vv.ravel(),
                expected_sv.ravel(),
                default_places,
                err_msg=f"Result {ii} virial",
            )

    def test_descriptor(self):
        _, extension = self.param
        if extension == ".pth":
            self.skipTest("eval_descriptor not supported for PyTorch models")
        for ii, result in enumerate(self.case.results):
            if result.descriptor is None:
                continue
            descpt = self.dp.eval_descriptor(result.coord, result.box, result.atype)
            expected_descpt = result.descriptor
            np.testing.assert_almost_equal(descpt.ravel(), expected_descpt.ravel())

    def test_2frame_atm(self):
        for ii, result in enumerate(self.case.results):
            coords2 = np.concatenate((result.coord, result.coord))
            box2 = np.concatenate((result.box, result.box))
            ee, ff, vv, ae, av = self.dp.eval(coords2, box2, result.atype, atomic=True)
            # check shape of the returns
            nframes = 2
            natoms = len(result.atype)
            self.assertEqual(ee.shape, (nframes, 1))
            self.assertEqual(ff.shape, (nframes, natoms, 3))
            self.assertEqual(vv.shape, (nframes, 9))
            self.assertEqual(ae.shape, (nframes, natoms, 1))
            self.assertEqual(av.shape, (nframes, natoms, 9))
            # check values
            expected_f = np.concatenate((result.force, result.force), axis=0)
            expected_e = np.concatenate(
                (result.atomic_energy, result.atomic_energy), axis=0
            )
            expected_v = np.concatenate(
                (result.atomic_virial, result.atomic_virial), axis=0
            )
            np.testing.assert_almost_equal(
                ff.ravel(), expected_f.ravel(), default_places
            )
            np.testing.assert_almost_equal(
                ae.ravel(), expected_e.ravel(), default_places
            )
            np.testing.assert_almost_equal(
                av.ravel(), expected_v.ravel(), default_places
            )
            expected_se = np.sum(expected_e.reshape([nframes, -1]), axis=1)
            np.testing.assert_almost_equal(
                ee.ravel(), expected_se.ravel(), default_places
            )
            expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis=1)
            np.testing.assert_almost_equal(
                vv.ravel(), expected_sv.ravel(), default_places
            )

    def test_zero_input(self):
        _, extension = self.param
        if extension == ".pb":
            from deepmd.tf.env import (
                tf,
            )

            if tf.test.is_gpu_available():
                # TODO: needs to fix
                self.skipTest("Segfault in GPUs")
        nframes = 1
        for box in [np.eye(3, dtype=np.float64), None]:
            ee, ff, vv = self.dp.eval(
                np.zeros([nframes, 0, 3], dtype=np.float64),
                box,
                np.zeros([0], dtype=int),
                atomic=False,
            )
            # check shape of the returns
            natoms = 0
            self.assertEqual(ee.shape, (nframes, 1))
            self.assertEqual(ff.shape, (nframes, natoms, 3))
            self.assertEqual(vv.shape, (nframes, 9))
            # check values
            np.testing.assert_almost_equal(ff.ravel(), 0, default_places)
            np.testing.assert_almost_equal(ee.ravel(), 0, default_places)
            np.testing.assert_almost_equal(vv.ravel(), 0, default_places)


@parameterized(
    tuple(get_cases().keys()),  # key
    (".pb",),  # model extension
)
class TestDeepPotNeighborList(TestDeepPot):
    @classmethod
    def setUpClass(cls):
        key, extension = cls.param
        cls.case = get_cases()[key]
        model_name = cls.case.get_model(extension)
        cls.dp = DeepEval(
            model_name,
            neighbor_list=ase.neighborlist.NewPrimitiveNeighborList(
                cutoffs=6, bothways=True
            ),
        )

    @unittest.skip("multiple frames not supported")
    def test_2frame_atm(self):
        pass

    @unittest.skip("Zero atoms not supported")
    def test_zero_input(self):
        pass
