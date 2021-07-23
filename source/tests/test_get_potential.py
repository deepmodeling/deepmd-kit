"""Test if `DeepPotential` facto function returns the right type of potential."""

import unittest
from pathlib import Path

from deepmd.infer import (DeepDipole, DeepGlobalPolar, DeepPolar, DeepPot,
                          DeepPotential, DeepWFC)

from infer.convert2pb import convert_pbtxt_to_pb


class TestGetPotential(unittest.TestCase):

    def setUp(self):
        self.work_dir = Path(__file__).parent / "infer"

        convert_pbtxt_to_pb(
            str(self.work_dir / "deeppot.pbtxt"),
            str(self.work_dir / "deep_pot.pb")
        )

        convert_pbtxt_to_pb(
            str(self.work_dir / "deepdipole.pbtxt"),
            str(self.work_dir / "deep_dipole.pb")
        )

        convert_pbtxt_to_pb(
            str(self.work_dir / "deeppolar.pbtxt"),
            str(self.work_dir / "deep_polar.pb")
        )

        # TODO add model files for globalpolar and WFC
        # convert_pbtxt_to_pb(
        #     str(self.work_dir / "deepglobalpolar.pbtxt"),
        #     str(self.work_dir / "deep_globalpolar.pb")
        # )

        # convert_pbtxt_to_pb(
        #     str(self.work_dir / "deepwfc.pbtxt"),
        #     str(self.work_dir / "deep_wfc.pb")
        # )

    def tearDown(self):
        for f in self.work_dir.glob("*.pb"):
            f.unlink()

    def test_factory(self):

        msg = "Returned wrong type of potential. Expected: {}, got: {}"

        dp = DeepPotential(self.work_dir / "deep_dipole.pb")
        self.assertIsInstance(dp, DeepDipole, msg.format(DeepDipole, type(dp)))

        dp = DeepPotential(self.work_dir / "deep_polar.pb")
        self.assertIsInstance(dp, DeepPolar, msg.format(DeepPolar, type(dp)))

        dp = DeepPotential(self.work_dir / "deep_pot.pb")
        self.assertIsInstance(dp, DeepPot, msg.format(DeepPot, type(dp)))

        # TODO add model files for globalpolar and WFC
        # dp = DeepPotential(self.work_dir / "deep_globalpolar.pb")
        # self.assertIsInstance(
        #     dp, DeepGlobalPolar, msg.format(DeepGlobalPolar, type(dp))
        # )

        # dp = DeepPotential(self.work_dir / "deep_wfc.pb")
        # self.assertIsInstance(dp, DeepWFC, msg.format(DeepWFC, type(dp)))
