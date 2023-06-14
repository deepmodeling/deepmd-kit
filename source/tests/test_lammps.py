import os
import subprocess
import unittest
from pathlib import (
    Path,
)

from deepmd.utils.convert import (
    convert_pbtxt_to_pb,
)


@unittest.skipIf(
    os.environ.get("CIBUILDWHEEL", "0") != "1",
    "Only test under cibuildwheel environment",
)
class TestLAMMPS(unittest.TestCase):
    """Test LAMMPS in cibuildwheel environment."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = (Path(__file__).parent / "infer").absolute()

        convert_pbtxt_to_pb(
            str(cls.work_dir / "deeppot.pbtxt"), str(cls.work_dir / "deep_pot.pb")
        )

    def test_lmp(self):
        in_file = (self.work_dir / "in.test").absolute()
        subprocess.check_call(["lmp", "-in", str(in_file)], cwd=str(self.work_dir))
