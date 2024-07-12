# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import subprocess
import unittest

from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

from .common import (
    infer_path,
)


@unittest.skipIf(
    os.environ.get("CIBUILDWHEEL", "0") != "1",
    "Only test under cibuildwheel environment",
)
class TestLAMMPS(unittest.TestCase):
    """Test LAMMPS in cibuildwheel environment."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = infer_path

        convert_pbtxt_to_pb(
            str(cls.work_dir / "deeppot.pbtxt"), str(cls.work_dir / "deep_pot.pb")
        )

    def test_lmp(self):
        in_file = (self.work_dir / "in.test").absolute()
        subprocess.check_call(["lmp", "-in", str(in_file)], cwd=str(self.work_dir))
