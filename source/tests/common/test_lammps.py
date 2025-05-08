# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import subprocess
import unittest

from deepmd.tf.utils.convert import (  # noqa: TID253
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
    def setUpClass(cls) -> None:
        cls.work_dir = infer_path

        convert_pbtxt_to_pb(
            str(cls.work_dir / "deeppot.pbtxt"), str(cls.work_dir / "deep_pot.pb")
        )

    @unittest.skipIf(
        os.environ.get("CUDA_VERSION", "").startswith("11"),
        "CUDA 11.x wheel uses PyTorch 2.3 which is not ABI compatible with TensorFlow",
    )
    def test_lmp(self) -> None:
        in_file = (self.work_dir / "in.test").absolute()
        subprocess.check_call(["lmp", "-in", str(in_file)], cwd=str(self.work_dir))
