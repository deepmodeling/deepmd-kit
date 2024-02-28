# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import sys
import unittest
from pathlib import (
    Path,
)

import numpy as np
from ase import (
    Atoms,
)
from ase.calculators.calculator import (
    FileIOCalculator,
)
from ase.calculators.socketio import (
    SocketIOCalculator,
)

from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

tests_path = Path(__file__).parent.parent.parent / "tests"
default_places = 6


class DPiPICalculator(FileIOCalculator):
    def __init__(self, model: str, use_unix: bool = True, **kwargs):
        self.xyz_file = "test_ipi.xyz"
        self.config_file = "config.json"
        config = {
            "verbose": False,
            "use_unix": use_unix,
            "port": 31415,
            "host": "localhost",
            "graph_file": model,
            "coord_file": self.xyz_file,
            "atom_type": {
                "O": 0,
                "H": 1,
            },
        }
        with open(self.config_file, "w") as f:
            json.dump(config, f)
        command = "dp_ipi " + self.config_file
        FileIOCalculator.__init__(
            self, command=command, label=self.config_file, **kwargs
        )

    def write_input(self, atoms, **kwargs):
        atoms.write(self.xyz_file, format="xyz")


class TestDPIPI(unittest.TestCase):
    # copy from test_deeppot_a.py
    @classmethod
    def setUpClass(cls):
        cls.model_file = "deeppot.pb"
        convert_pbtxt_to_pb(
            str(tests_path / os.path.join("infer", "deeppot.pbtxt")), "deeppot.pb"
        )

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

    def test_ase_unix(self):
        with SocketIOCalculator(
            DPiPICalculator(self.model_file), log=sys.stdout, unixsocket="localhost"
        ) as calc:
            water = Atoms(
                "OHHOHH",
                positions=self.coords.reshape((-1, 3)),
                cell=self.box.reshape((3, 3)),
                calculator=calc,
            )
            ee = water.get_potential_energy()
            ff = water.get_forces()
        nframes = 1
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)

    def test_ase_nounix(self):
        with SocketIOCalculator(
            DPiPICalculator(self.model_file, use_unix=False),
            log=sys.stdout,
        ) as calc:
            water = Atoms(
                "OHHOHH",
                positions=self.coords.reshape((-1, 3)),
                cell=self.box.reshape((3, 3)),
                calculator=calc,
            )
            ee = water.get_potential_energy()
            ff = water.get_forces()
        nframes = 1
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)

    def test_normalize_coords(self):
        # coordinate nomarlization should happen inside the interface
        cell = self.box.reshape((3, 3))
        coord = self.coords.reshape((-1, 3))
        # random unwrap coords
        coord[0] += np.array([3, 0, 0]) @ cell
        coord[1] += np.array([0, -3, 0]) @ cell
        coord[2] += np.array([0, 0, 3]) @ cell
        coord[3] += np.array([-3, 0, 0]) @ cell
        coord[4] += np.array([0, 3, 0]) @ cell
        coord[5] += np.array([0, 0, -3]) @ cell
        with SocketIOCalculator(
            DPiPICalculator(self.model_file, use_unix=False),
            log=sys.stdout,
        ) as calc:
            water = Atoms(
                "OHHOHH",
                positions=coord,
                cell=cell,
                calculator=calc,
            )
            ee = water.get_potential_energy()
            ff = water.get_forces()
        nframes = 1
        np.testing.assert_almost_equal(
            ff.ravel(), self.expected_f.ravel(), default_places
        )
        expected_se = np.sum(self.expected_e.reshape([nframes, -1]), axis=1)
        np.testing.assert_almost_equal(ee.ravel(), expected_se.ravel(), default_places)


class TestDPIPIPt(TestDPIPI):
    @classmethod
    def setUpClass(cls):
        cls.model_file = str(tests_path / "infer" / "deeppot_sea.pth")

    def setUp(self):
        super().setUp()

        self.box = np.array([13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0])
        self.expected_e = np.array(
            [
                -93.016873944029,
                -185.923296645958,
                -185.927096544970,
                -93.019371018039,
                -185.926179995548,
                -185.924351901852,
            ]
        )
        self.expected_f = np.array(
            [
                0.006277522211,
                -0.001117962774,
                0.000618580445,
                0.009928999655,
                0.003026035654,
                -0.006941982227,
                0.000667853212,
                -0.002449963843,
                0.006506463508,
                -0.007284129115,
                0.000530662205,
                -0.000028806821,
                0.000068097781,
                0.006121331983,
                -0.009019754602,
                -0.009658343745,
                -0.006110103225,
                0.008865499697,
            ]
        )
        self.expected_v = np.array(
            [
                -0.000155238009,
                0.000116605516,
                -0.007869862476,
                0.000465578340,
                0.008182547185,
                -0.002398713212,
                -0.008112887338,
                -0.002423738425,
                0.007210716605,
                -0.019203504012,
                0.001724938709,
                0.009909211091,
                0.001153857542,
                -0.001600015103,
                -0.000560024090,
                0.010727836276,
                -0.001034836404,
                -0.007973454377,
                -0.021517399106,
                -0.004064359664,
                0.004866398692,
                -0.003360038617,
                -0.007241406162,
                0.005920941051,
                0.004899151657,
                0.006290788591,
                -0.006478820311,
                0.001921504710,
                0.001313470921,
                -0.000304091236,
                0.001684345981,
                0.004124109256,
                -0.006396084465,
                -0.000701095618,
                -0.006356507032,
                0.009818550859,
                -0.015230664587,
                -0.000110244376,
                0.000690319396,
                0.000045953023,
                -0.005726548770,
                0.008769818495,
                -0.000572380210,
                0.008860603423,
                -0.013819348050,
                -0.021227082558,
                -0.004977781343,
                0.006646239696,
                -0.005987066507,
                -0.002767831232,
                0.003746502525,
                0.007697590397,
                0.003746130152,
                -0.005172634748,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.dp = None
