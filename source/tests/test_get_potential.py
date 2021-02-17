import unittest
from pathlib import Path
from shutil import rmtree

from deepmd.infer import DeepPotential, DeepDipole, DeepGlobalPolar, DeepPolar, DeepPot, DeepWFC
from deepmd.env import tf


class TestGetPotential(unittest.TestCase):

    def setUp(self):
        self.work_dir = Path(__file__).parent / "test_get_potential"
        self.work_dir.mkdir(exist_ok=True)
        # TODO create all types of graphs
        ...

    def tearDown(self):
        rmtree(self.work_dir)

    def test_merge_all_stat(self):

        dp = DeepPotential(self.work_dir / "deep_pot_model.pb")
        self.assertIsInstance(dp, DeepPot, "Returned wrong type of potential")

        dp = DeepPotential(self.work_dir / "deep_polar_model.pb")
        self.assertIsInstance(dp, DeepPolar, "Returned wrong type of potential")

        dp = DeepPotential(self.work_dir / "deep_global_polar_model.pb")
        self.assertIsInstance(dp, DeepGlobalPolar, "Returned wrong type of potential")

        dp = DeepPotential(self.work_dir / "deep_wfc_model.pb")
        self.assertIsInstance(dp, DeepWFC, "Returned wrong type of potential")

        dp = DeepPotential(self.work_dir / "deep_dipole_model.pb")
        self.assertIsInstance(dp, DeepDipole, "Returned wrong type of potential")
