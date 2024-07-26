# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil
import unittest

import dpdata
import numpy as np
from packaging.version import parse as parse_version

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    MODEL_VERSION,
)
from deepmd.tf.infer import (
    DeepPot,
)
from deepmd.tf.utils.convert import (
    convert_dp10_to_dp11,
    convert_dp012_to_dp10,
    convert_dp12_to_dp13,
    convert_dp13_to_dp20,
    convert_dp20_to_dp21,
    convert_pbtxt_to_pb,
    detect_model_version,
)

from .common import (
    infer_path,
    run_dp,
    tests_path,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10


class TestModelMajorCompatability(unittest.TestCase):
    def setUp(self):
        model_file = str(infer_path / os.path.join("deeppot.pbtxt"))
        with open(model_file) as fp:
            # data = fp.read().replace('\n', '')
            data = fp.read().split("\n")
            for ii in range(len(data)):
                if "model_attr/model_version" in data[ii]:
                    for jj in range(ii, len(data)):
                        if "string_val:" in data[jj]:
                            data[jj] = data[jj].replace(MODEL_VERSION, "0.0")
                            break
        self.version_pbtxt = str(tests_path / "deeppot-ver.pbtxt")
        self.version_pb = str(tests_path / "deeppot.pb")
        with open(self.version_pbtxt, "w") as fp:
            fp.write("\n".join(data))
        convert_pbtxt_to_pb(self.version_pbtxt, self.version_pb)

    def tearDown(self):
        os.remove(self.version_pbtxt)
        os.remove(self.version_pb)

    def test(self):
        with self.assertRaises(RuntimeError) as context:
            DeepPot(str(self.version_pb))
        self.assertTrue("incompatible" in str(context.exception))
        self.assertTrue(MODEL_VERSION in str(context.exception))
        self.assertTrue("0.0" in str(context.exception))


class TestModelMinorCompatability(unittest.TestCase):
    def setUp(self):
        model_file = str(infer_path / os.path.join("deeppot.pbtxt"))
        with open(model_file) as fp:
            # data = fp.read().replace('\n', '')
            data = fp.read().split("\n")
            for ii in range(len(data)):
                if "model_attr/model_version" in data[ii]:
                    for jj in range(ii, len(data)):
                        if "string_val:" in data[jj]:
                            data[jj] = data[jj].replace(MODEL_VERSION, "0.1000000")
                            break
        self.version_pbtxt = str(tests_path / "deeppot-ver.pbtxt")
        self.version_pb = str(tests_path / "deeppot.pb")
        with open(self.version_pbtxt, "w") as fp:
            fp.write("\n".join(data))
        convert_pbtxt_to_pb(self.version_pbtxt, self.version_pb)

    def tearDown(self):
        os.remove(self.version_pbtxt)
        os.remove(self.version_pb)

    def test(self):
        with self.assertRaises(RuntimeError) as context:
            DeepPot(self.version_pb)
        self.assertTrue("incompatible" in str(context.exception))
        self.assertTrue(MODEL_VERSION in str(context.exception))
        self.assertTrue("0.1000000" in str(context.exception))


# TestDeepPotAPBC: moved to infer/test_models.py


class TestDeepPotANoPBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deeppot.pbtxt")), "deeppot.pb"
        )
        cls.dp = DeepPot("deeppot.pb")

    def setUp(self):
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ]
        )
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = None
        self.expected_e = np.array(
            [
                -9.255934839310273787e01,
                -1.863253376736990106e02,
                -1.857237299341402945e02,
                -9.279308539717486326e01,
                -1.863708105823244239e02,
                -1.863635196514972563e02,
            ]
        )
        self.expected_f = np.array(
            [
                -2.161037360255332107e00,
                9.052994347015581589e-01,
                1.635379623977007979e00,
                2.161037360255332107e00,
                -9.052994347015581589e-01,
                -1.635379623977007979e00,
                -1.167128117249453811e-02,
                1.371975700096064992e-03,
                -1.575265180249604477e-03,
                6.226508593971802341e-01,
                -1.816734122009256991e-01,
                3.561766019664774907e-01,
                -1.406075393906316626e-02,
                3.789140061530929526e-01,
                -6.018777878642909140e-01,
                -5.969188242856223736e-01,
                -1.986125696522633155e-01,
                2.472764510780630642e-01,
            ]
        )
        self.expected_v = np.array(
            [
                -7.042445481792056761e-01,
                2.950213647777754078e-01,
                5.329418202437231633e-01,
                2.950213647777752968e-01,
                -1.235900311906896754e-01,
                -2.232594111831812944e-01,
                5.329418202437232743e-01,
                -2.232594111831813499e-01,
                -4.033073234276823849e-01,
                -8.949230984097404917e-01,
                3.749002169013777030e-01,
                6.772391014992630298e-01,
                3.749002169013777586e-01,
                -1.570527935667933583e-01,
                -2.837082722496912512e-01,
                6.772391014992631408e-01,
                -2.837082722496912512e-01,
                -5.125052659994422388e-01,
                4.858210330291591605e-02,
                -6.902596153269104431e-03,
                6.682612642430500391e-03,
                -5.612247004554610057e-03,
                9.767795567660207592e-04,
                -9.773758942738038254e-04,
                5.638322117219018645e-03,
                -9.483806049779926932e-04,
                8.493873281881353637e-04,
                -2.941738570564985666e-01,
                -4.482529909499673171e-02,
                4.091569840186781021e-02,
                -4.509020615859140463e-02,
                -1.013919988807244071e-01,
                1.551440772665269030e-01,
                4.181857726606644232e-02,
                1.547200233064863484e-01,
                -2.398213304685777592e-01,
                -3.218625798524068354e-02,
                -1.012438450438508421e-02,
                1.271639330380921855e-02,
                3.072814938490859779e-03,
                -9.556241797915024372e-02,
                1.512251983492413077e-01,
                -8.277872384009607454e-03,
                1.505412040827929787e-01,
                -2.386150620881526407e-01,
                -2.312295470054945568e-01,
                -6.631490213524345034e-02,
                7.932427266386249398e-02,
                -8.053754366323923053e-02,
                -3.294595881137418747e-02,
                4.342495071150231922e-02,
                1.004599500126941436e-01,
                4.450400364869536163e-02,
                -5.951077548033092968e-02,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        os.remove("deeppot.pb")
        cls.dp = None

    def test_1frame(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic=False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        # check values
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_1frame_atm(self):
        ee, ff, vv, ae, av = self.dp.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, natoms, 1))
        self.assertEqual(av.shape, (nframes, natoms, 9))
        # check values
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            ae.ravel(), self.expected_e.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            av.ravel(), self.expected_v.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_2frame_atm(self):
        coords2 = np.concatenate((self.coords, self.coords))
        ee, ff, vv, ae, av = self.dp.eval(coords2, self.box, self.atype, atomic=True)
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, natoms, 1))
        self.assertEqual(av.shape, (nframes, natoms, 9))
        # check values
        expected_f = np.concatenate((self.expected_f, self.expected_f), axis=0)
        expected_e = np.concatenate((self.expected_e, self.expected_e), axis=0)
        expected_v = np.concatenate((self.expected_v, self.expected_v), axis=0)
        np.testing.assert_almost_equal(ff.ravel(), expected_f.ravel(), default_places)
        np.testing.assert_almost_equal(ae.ravel(), expected_e.ravel(), default_places)
        np.testing.assert_almost_equal(av.ravel(), expected_v.ravel(), default_places)
        expected_se = np.sum(expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_dpdata_driver(self):
        nframes = 1
        system = dpdata.System(
            data={
                "coords": self.coords.reshape((nframes, -1, 3)),
                "cells": np.zeros((nframes, 3, 3)),
                "atom_types": np.array(self.atype),
                "orig": np.zeros((3,)),
                "atom_names": ["O", "H"],
                "atom_numbs": [2, 4],
                "nopbc": True,
            }
        )
        system_predicted = system.predict(self.dp, driver="dp")
        np.testing.assert_almost_equal(
            system_predicted["forces"].ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(
            system_predicted["energies"].ravel(), expected_se.ravel(), default_places
        )
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(
            system_predicted["virials"].ravel(), expected_sv.ravel(), default_places
        )


class TestDeepPotALargeBoxNoPBC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("deeppot.pbtxt")), "deeppot.pb"
        )
        cls.dp = DeepPot("deeppot.pb")

    def setUp(self):
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ]
        )
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([19.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])
        self.expected_e = np.array(
            [
                -9.255934839310273787e01,
                -1.863253376736990106e02,
                -1.857237299341402945e02,
                -9.279308539717486326e01,
                -1.863708105823244239e02,
                -1.863635196514972563e02,
            ]
        )
        self.expected_f = np.array(
            [
                -2.161037360255332107e00,
                9.052994347015581589e-01,
                1.635379623977007979e00,
                2.161037360255332107e00,
                -9.052994347015581589e-01,
                -1.635379623977007979e00,
                -1.167128117249453811e-02,
                1.371975700096064992e-03,
                -1.575265180249604477e-03,
                6.226508593971802341e-01,
                -1.816734122009256991e-01,
                3.561766019664774907e-01,
                -1.406075393906316626e-02,
                3.789140061530929526e-01,
                -6.018777878642909140e-01,
                -5.969188242856223736e-01,
                -1.986125696522633155e-01,
                2.472764510780630642e-01,
            ]
        )
        self.expected_v = np.array(
            [
                -7.042445481792056761e-01,
                2.950213647777754078e-01,
                5.329418202437231633e-01,
                2.950213647777752968e-01,
                -1.235900311906896754e-01,
                -2.232594111831812944e-01,
                5.329418202437232743e-01,
                -2.232594111831813499e-01,
                -4.033073234276823849e-01,
                -8.949230984097404917e-01,
                3.749002169013777030e-01,
                6.772391014992630298e-01,
                3.749002169013777586e-01,
                -1.570527935667933583e-01,
                -2.837082722496912512e-01,
                6.772391014992631408e-01,
                -2.837082722496912512e-01,
                -5.125052659994422388e-01,
                4.858210330291591605e-02,
                -6.902596153269104431e-03,
                6.682612642430500391e-03,
                -5.612247004554610057e-03,
                9.767795567660207592e-04,
                -9.773758942738038254e-04,
                5.638322117219018645e-03,
                -9.483806049779926932e-04,
                8.493873281881353637e-04,
                -2.941738570564985666e-01,
                -4.482529909499673171e-02,
                4.091569840186781021e-02,
                -4.509020615859140463e-02,
                -1.013919988807244071e-01,
                1.551440772665269030e-01,
                4.181857726606644232e-02,
                1.547200233064863484e-01,
                -2.398213304685777592e-01,
                -3.218625798524068354e-02,
                -1.012438450438508421e-02,
                1.271639330380921855e-02,
                3.072814938490859779e-03,
                -9.556241797915024372e-02,
                1.512251983492413077e-01,
                -8.277872384009607454e-03,
                1.505412040827929787e-01,
                -2.386150620881526407e-01,
                -2.312295470054945568e-01,
                -6.631490213524345034e-02,
                7.932427266386249398e-02,
                -8.053754366323923053e-02,
                -3.294595881137418747e-02,
                4.342495071150231922e-02,
                1.004599500126941436e-01,
                4.450400364869536163e-02,
                -5.951077548033092968e-02,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        os.remove("deeppot.pb")
        cls.dp = None

    def test_1frame(self):
        ee, ff, vv = self.dp.eval(self.coords, self.box, self.atype, atomic=False)
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        # check values
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_1frame_atm(self):
        ee, ff, vv, ae, av = self.dp.eval(
            self.coords, self.box, self.atype, atomic=True
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, natoms, 1))
        self.assertEqual(av.shape, (nframes, natoms, 9))
        # check values
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            ae.ravel(), self.expected_e.ravel(), default_places
        )
        np.testing.assert_almost_equal(
            av.ravel(), self.expected_v.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), default_places)

    def test_ase(self):
        from ase import (
            Atoms,
        )

        from deepmd.tf.calculator import (
            DP,
        )

        water = Atoms(
            "OHHOHH",
            positions=self.coords.reshape((-1, 3)),
            cell=self.box.reshape((3, 3)),
            calculator=DP("deeppot.pb"),
        )
        ee = water.get_potential_energy()
        ff = water.get_forces()
        nframes = 1
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)

    def test_dpdata_driver(self):
        nframes = 1
        system = dpdata.System(
            data={
                "coords": self.coords.reshape((nframes, -1, 3)),
                "cells": self.box.reshape((nframes, 3, 3)),
                "atom_types": np.array(self.atype),
                "orig": np.zeros((3,)),
                "atom_names": ["O", "H"],
                "atom_numbs": [2, 4],
            }
        )
        system_predicted = system.predict("deeppot.pb", driver="dp")
        np.testing.assert_almost_equal(
            system_predicted["forces"].ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(
            system_predicted["energies"].ravel(), expected_se.ravel(), default_places
        )
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(
            system_predicted["virials"].ravel(), expected_sv.ravel(), default_places
        )


class TestModelConvert(unittest.TestCase):
    def setUp(self):
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ]
        )
        self.atype = [0, 1, 1, 0, 1, 1]
        self.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])

    def test_convert_012(self):
        old_model = "deeppot.pb"
        new_model = "deeppot-new.pb"
        convert_pbtxt_to_pb(str(infer_path / "sea_012.pbtxt"), old_model)
        run_dp(f"dp convert-from 0.12 -i {old_model} -o {new_model}")
        dp = DeepPot(new_model)
        _ = dp.eval(self.coords, self.box, self.atype, atomic=True)
        os.remove(old_model)
        os.remove(new_model)

    def test_convert(self):
        old_model = "deeppot.pb"
        new_model = "deeppot-new.pb"
        convert_pbtxt_to_pb(str(infer_path / "sea_012.pbtxt"), old_model)
        run_dp(f"dp convert-from -i {old_model} -o {new_model}")
        dp = DeepPot(new_model)
        _ = dp.eval(self.coords, self.box, self.atype, atomic=True)
        os.remove(old_model)
        os.remove(new_model)

    def test_detect(self):
        old_model = "deeppot.pb"
        new_model_txt = "deeppot_new.pbtxt"
        new_model_pb = "deeppot_new.pb"
        convert_pbtxt_to_pb(str(infer_path / "sea_012.pbtxt"), old_model)
        version = detect_model_version(old_model)
        self.assertEqual(version, parse_version("0.12"))
        os.remove(old_model)
        shutil.copyfile(str(infer_path / "sea_012.pbtxt"), new_model_txt)
        convert_dp012_to_dp10(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("1.0"))
        os.remove(new_model_pb)
        convert_dp10_to_dp11(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("1.3"))
        os.remove(new_model_pb)
        convert_dp12_to_dp13(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("1.3"))
        os.remove(new_model_pb)
        convert_dp13_to_dp20(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("2.0"))
        os.remove(new_model_pb)
        convert_dp20_to_dp21(new_model_txt)
        convert_pbtxt_to_pb(new_model_txt, new_model_pb)
        version = detect_model_version(new_model_pb)
        self.assertEqual(version, parse_version("2.1"))
        os.remove(new_model_pb)
        os.remove(new_model_txt)


class TestTypeEmbed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("se_e2_a_tebd.pbtxt")),
            "se_e2_a_tebd.pb",
        )
        cls.dp = DeepPot("se_e2_a_tebd.pb")

    def test_eval_typeebd(self):
        expected_typeebd = np.array(
            [
                [
                    -0.4602908199,
                    -0.9440795817,
                    -0.857044451,
                    -0.3448434537,
                    -0.6310194663,
                    -0.9765837147,
                    -0.3945653821,
                    0.8973716518,
                ],
                [
                    -0.7239568558,
                    -0.9672733137,
                    -0.420987752,
                    -0.4542931277,
                    -0.79586188,
                    -0.9615886543,
                    -0.6864800369,
                    0.9477863254,
                ],
            ]
        )

        eval_typeebd = self.dp.eval_typeebd()
        np.testing.assert_almost_equal(eval_typeebd, expected_typeebd, default_places)


class FparamAparamCommonTest:
    """Test fparam and aparam."""

    def setUp(self):
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ]
        )
        self.atype = [0, 0, 0, 0, 0, 0]
        self.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])
        self.fparam = 0.25852028
        self.aparam = np.repeat(self.fparam, len(self.atype))
        self.expected_e = np.array(
            [
                -1.038271183039953804e-01,
                -7.285433575272914908e-02,
                -9.467600174099155552e-02,
                -1.467050086239614082e-01,
                -7.660561620618722145e-02,
                -7.277295998502930630e-02,
            ]
        )
        self.expected_f = np.array(
            [
                6.622266817497907132e-02,
                5.278739055693523058e-02,
                2.265727495541422845e-02,
                -2.606047850915838363e-02,
                -4.538811686410718776e-02,
                1.058247569147072187e-02,
                1.679392490937766935e-01,
                -2.257828022687320690e-03,
                -4.490145670355452645e-02,
                -1.148364103573685929e-01,
                -1.169790466695089237e-02,
                6.140402504113953025e-02,
                -8.078778132132799494e-02,
                -5.838878056243369807e-02,
                6.773639989682191109e-02,
                -1.247724708090079161e-02,
                6.494523955924384750e-02,
                -1.174787188812918687e-01,
            ]
        )
        self.expected_v = np.array(
            [
                -1.589185553287162656e-01,
                2.586163333170100279e-03,
                -1.575127933809472624e-04,
                -1.855360380105876630e-02,
                1.949822090859933826e-02,
                -1.006552056166355388e-02,
                3.177029853276916449e-02,
                1.714349636720383010e-03,
                -1.290389175187874483e-03,
                -8.553510339477603253e-02,
                -5.654637257232508415e-03,
                -1.286954833787038420e-02,
                2.464156457499515687e-02,
                -2.398202886026797043e-02,
                -1.957110465239037672e-02,
                2.233492928605742764e-02,
                6.107843207824020099e-03,
                1.707078295947736047e-03,
                -1.653994088976195043e-01,
                3.894358678172111371e-02,
                -2.169595969759342477e-02,
                6.819704294738503786e-03,
                -5.018242039618424008e-03,
                2.640664428663210429e-03,
                -1.985298275686078057e-03,
                -3.638421609610945767e-02,
                2.342932331075030239e-02,
                -8.501331914753691710e-02,
                -2.181253413538992297e-03,
                4.311300069651782287e-03,
                -1.910329328333908129e-03,
                -1.808810159508548836e-03,
                -1.540075281450827612e-03,
                -1.173703213175551763e-02,
                -2.596306629910121507e-03,
                6.705025662372287101e-03,
                -9.038455005073858795e-02,
                3.011717773578577451e-02,
                -5.083054073419784880e-02,
                -2.951210292616929069e-03,
                2.342445652898489383e-02,
                -4.091207474993674431e-02,
                -1.648470649301832236e-02,
                -2.872261885460645689e-02,
                4.763924972552112391e-02,
                -8.300036532764677732e-02,
                1.020429228955421243e-03,
                -1.026734151199098881e-03,
                5.678534096113684732e-02,
                1.273635718045938205e-02,
                -1.530143225195957322e-02,
                -1.061671865629566225e-01,
                -2.486859433265622629e-02,
                2.875323131744185121e-02,
            ]
        )
        self.places = default_places

    def test_attrs(self):
        self.assertEqual(self.dp.get_ntypes(), 1)
        self.assertAlmostEqual(self.dp.get_rcut(), 6.0, places=self.places)
        self.assertEqual(self.dp.get_dim_fparam(), 1)
        self.assertEqual(self.dp.get_dim_aparam(), 1)

    def test_1frame(self):
        ee, ff, vv = self.dp.eval(
            self.coords,
            self.box,
            self.atype,
            fparam=self.fparam,
            aparam=self.aparam,
            atomic=False,
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        # check values
        np.testing.assert_almost_equal(ff.ravel(), self.expected_f.ravel(), self.places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), self.places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), self.places)

    def test_1frame_atm(self):
        ee, ff, vv, ae, av = self.dp.eval(
            self.coords,
            self.box,
            self.atype,
            fparam=self.fparam,
            aparam=self.aparam,
            atomic=True,
        )
        # check shape of the returns
        nframes = 1
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, natoms, 1))
        self.assertEqual(av.shape, (nframes, natoms, 9))
        # check values
        np.testing.assert_almost_equal(ff.ravel(), self.expected_f.ravel(), self.places)
        np.testing.assert_almost_equal(ae.ravel(), self.expected_e.ravel(), self.places)
        np.testing.assert_almost_equal(av.ravel(), self.expected_v.ravel(), self.places)
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), self.places)
        expected_sv = np.sum(self.expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), self.places)

    def test_2frame_atm_single_param(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee, ff, vv, ae, av = self.dp.eval(
            coords2,
            box2,
            self.atype,
            fparam=self.fparam,
            aparam=self.aparam,
            atomic=True,
        )
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, natoms, 1))
        self.assertEqual(av.shape, (nframes, natoms, 9))
        # check values
        expected_f = np.concatenate((self.expected_f, self.expected_f), axis=0)
        expected_e = np.concatenate((self.expected_e, self.expected_e), axis=0)
        expected_v = np.concatenate((self.expected_v, self.expected_v), axis=0)
        np.testing.assert_almost_equal(ff.ravel(), expected_f.ravel(), self.places)
        np.testing.assert_almost_equal(ae.ravel(), expected_e.ravel(), self.places)
        np.testing.assert_almost_equal(av.ravel(), expected_v.ravel(), self.places)
        expected_se = np.sum(expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), self.places)
        expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), self.places)

    def test_2frame_atm_all_param(self):
        coords2 = np.concatenate((self.coords, self.coords))
        box2 = np.concatenate((self.box, self.box))
        ee, ff, vv, ae, av = self.dp.eval(
            coords2,
            box2,
            self.atype,
            fparam=np.repeat(self.fparam, 2),
            aparam=np.repeat(self.aparam, 2),
            atomic=True,
        )
        # check shape of the returns
        nframes = 2
        natoms = len(self.atype)
        self.assertEqual(ee.shape, (nframes, 1))
        self.assertEqual(ff.shape, (nframes, natoms, 3))
        self.assertEqual(vv.shape, (nframes, 9))
        self.assertEqual(ae.shape, (nframes, natoms, 1))
        self.assertEqual(av.shape, (nframes, natoms, 9))
        # check values
        expected_f = np.concatenate((self.expected_f, self.expected_f), axis=0)
        expected_e = np.concatenate((self.expected_e, self.expected_e), axis=0)
        expected_v = np.concatenate((self.expected_v, self.expected_v), axis=0)
        np.testing.assert_almost_equal(ff.ravel(), expected_f.ravel(), self.places)
        np.testing.assert_almost_equal(ae.ravel(), expected_e.ravel(), self.places)
        np.testing.assert_almost_equal(av.ravel(), expected_v.ravel(), self.places)
        expected_se = np.sum(expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), self.places)
        expected_sv = np.sum(expected_v.reshape([nframes, -1, 9]), axis=1)
        np.testing.assert_almost_equal(vv.ravel(), expected_sv.ravel(), self.places)


class TestFparamAparam(FparamAparamCommonTest, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(infer_path / os.path.join("fparam_aparam.pbtxt")),
            "fparam_aparam.pb",
        )
        cls.dp = DeepPot("fparam_aparam.pb")

    @classmethod
    def tearDownClass(cls):
        os.remove("fparam_aparam.pb")
        cls.dp = None


# TestDeepPotAPBCNeighborList: moved to infer/test_models.py
