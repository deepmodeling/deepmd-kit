# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import ase
import dpdata
import numpy as np

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)

from ..consistent.common import (
    parameterized,
)
from .case import (
    get_cases,
)

default_places = 7


@parameterized(
    (
        "se_e2_a",
        "se_e2_r",
        "fparam_aparam",
    ),  # key
    (".pb", ".pth"),  # model extension
)
class TestDeepPot(unittest.TestCase):
    # moved from tests/tf/test_deeppot_a.py

    @classmethod
    def setUpClass(cls) -> None:
        key, extension = cls.param
        cls.case = get_cases()[key]
        cls.model_name = cls.case.get_model(extension)
        cls.dp = DeepEval(cls.model_name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dp = None

    def setUp(self) -> None:
        key, extension = self.param
        if key == "se_e2_r" and extension == ".pth":
            self.skipTest(
                reason="se_e2_r type_one_side is not supported for PyTorch models"
            )

    def test_attrs(self) -> None:
        assert isinstance(self.dp, DeepPot)
        self.assertEqual(self.dp.get_ntypes(), self.case.ntypes)
        self.assertAlmostEqual(
            self.dp.get_rcut(), self.case.rcut, places=default_places
        )
        self.assertEqual(self.dp.get_type_map(), self.case.type_map)
        self.assertEqual(self.dp.get_dim_fparam(), self.case.dim_fparam)
        self.assertEqual(self.dp.get_dim_aparam(), self.case.dim_aparam)

    def test_1frame(self) -> None:
        for ii, result in enumerate(self.case.results):
            ee, ff, vv = self.dp.eval(
                result.coord,
                result.box,
                result.atype,
                atomic=False,
                fparam=result.fparam,
                aparam=result.aparam,
            )[:3]
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

    def test_1frame_atm(self) -> None:
        for ii, result in enumerate(self.case.results):
            ee, ff, vv, ae, av = self.dp.eval(
                result.coord,
                result.box,
                result.atype,
                atomic=True,
                fparam=result.fparam,
                aparam=result.aparam,
            )[:5]
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

    def test_descriptor(self) -> None:
        _, extension = self.param
        for ii, result in enumerate(self.case.results):
            if result.descriptor is None:
                continue
            descpt = self.dp.eval_descriptor(result.coord, result.box, result.atype)
            expected_descpt = result.descriptor
            np.testing.assert_almost_equal(descpt.ravel(), expected_descpt.ravel())
            # See #4533
            descpt = self.dp.eval_descriptor(result.coord, result.box, result.atype)
            expected_descpt = result.descriptor
            np.testing.assert_almost_equal(descpt.ravel(), expected_descpt.ravel())

    def test_2frame_atm(self) -> None:
        for ii, result in enumerate(self.case.results):
            coords2 = np.concatenate((result.coord, result.coord))
            if result.box is not None:
                box2 = np.concatenate((result.box, result.box))
            else:
                box2 = None
            ee, ff, vv, ae, av = self.dp.eval(
                coords2,
                box2,
                result.atype,
                atomic=True,
                fparam=result.fparam,
                aparam=result.aparam,
            )[:5]
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

    def test_zero_input(self) -> None:
        _, extension = self.param
        if extension == ".pb":
            from deepmd.tf.env import (
                tf,
            )

            if tf.test.is_gpu_available():
                # TODO: needs to fix
                self.skipTest("Segfault in GPUs")
        nframes = 1
        for box in [np.eye(3, dtype=np.float64).reshape(1, 3, 3), None]:
            ee, ff, vv = self.dp.eval(
                np.zeros([nframes, 0, 3], dtype=np.float64),
                box,
                np.zeros([0], dtype=int),
                atomic=False,
                fparam=np.zeros([self.case.dim_fparam], dtype=np.float64)
                if self.case.dim_fparam
                else None,
                aparam=np.zeros([0, self.case.dim_aparam], dtype=np.float64)
                if self.case.dim_aparam
                else None,
            )[:3]
            # check shape of the returns
            natoms = 0
            self.assertEqual(ee.shape, (nframes, 1))
            self.assertEqual(ff.shape, (nframes, natoms, 3))
            self.assertEqual(vv.shape, (nframes, 9))
            # check values
            np.testing.assert_almost_equal(ff.ravel(), 0, default_places)
            np.testing.assert_almost_equal(ee.ravel(), 0, default_places)
            np.testing.assert_almost_equal(vv.ravel(), 0, default_places)

    def test_ase(self) -> None:
        from ase import (
            Atoms,
        )

        from deepmd.calculator import (
            DP,
        )

        if self.case.dim_fparam or self.case.dim_aparam:
            self.skipTest("fparam and aparam not supported")

        for ii, result in enumerate(self.case.results):
            water = Atoms(
                np.array(self.case.type_map)[result.atype].tolist(),
                positions=result.coord.reshape((-1, 3)),
                cell=result.box.reshape((3, 3)) if result.box is not None else None,
                calculator=DP(self.model_name),
                pbc=result.box is not None,
            )
            ee = water.get_potential_energy()
            ff = water.get_forces()
            nframes = 1
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

    def test_dpdata_driver(self) -> None:
        if self.case.dim_fparam or self.case.dim_aparam:
            self.skipTest("fparam and aparam not supported")

        for ii, result in enumerate(self.case.results):
            nframes = 1
            # infer atom_numbs from atype
            atom_numbs = np.bincount(result.atype).tolist()
            system = dpdata.System(
                data={
                    "coords": result.coord.reshape((nframes, result.nloc, 3)),
                    "cells": np.zeros((nframes, 3, 3))
                    if result.box is None
                    else result.box.reshape((nframes, 3, 3)),
                    "atom_types": np.array(result.atype),
                    "orig": np.zeros((3,)),
                    "atom_names": self.case.type_map,
                    "atom_numbs": atom_numbs,
                    "nopbc": result.box is None,
                }
            )
            system_predicted = system.predict(self.dp, driver="dp")
            np.testing.assert_almost_equal(
                system_predicted["forces"].ravel(),
                result.force.ravel(),
                default_places,
                err_msg=f"Result {ii} force",
            )
            expected_se = np.sum(result.energy.reshape([nframes, -1]), axis=1)
            np.testing.assert_almost_equal(
                system_predicted["energies"].ravel(),
                expected_se.ravel(),
                default_places,
                err_msg=f"Result {ii} energy",
            )
            expected_sv = np.sum(result.virial.reshape([nframes, -1, 9]), axis=1)
            np.testing.assert_almost_equal(
                system_predicted["virials"].ravel(),
                expected_sv.ravel(),
                default_places,
                err_msg=f"Result {ii} virial",
            )

    def test_model_script_def(self) -> None:
        if self.case.model_def_script is not None:
            self.assertDictEqual(
                self.case.model_def_script, self.dp.get_model_def_script()
            )


@parameterized(
    ("se_e2_a",),  # key
    (".pb",),  # model extension
)
class TestDeepPotNeighborList(TestDeepPot):
    @classmethod
    def setUpClass(cls) -> None:
        key, extension = cls.param
        cls.case = get_cases()[key]
        model_name = cls.case.get_model(extension)
        cls.dp = DeepEval(
            model_name,
            neighbor_list=ase.neighborlist.NewPrimitiveNeighborList(
                cutoffs=cls.case.rcut, bothways=True
            ),
        )

    @unittest.skip("multiple frames not supported")
    def test_2frame_atm(self) -> None:
        pass

    @unittest.skip("Zero atoms not supported")
    def test_zero_input(self) -> None:
        pass
