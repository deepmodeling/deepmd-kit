# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test if `DeepPotential` facto function returns the right type of potential."""

import tempfile
import unittest

from deepmd.infer.deep_polar import (
    DeepGlobalPolar,
)
from deepmd.infer.deep_wfc import (
    DeepWFC,
)
from deepmd.tf.infer import (
    DeepDipole,
    DeepPolar,
    DeepPot,
    DeepPotential,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)

from .common import (
    infer_path,
)


class TestGetPotential(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = infer_path

        convert_pbtxt_to_pb(
            str(self.work_dir / "deeppot.pbtxt"), str(self.work_dir / "deep_pot.pb")
        )

        convert_pbtxt_to_pb(
            str(self.work_dir / "deepdipole.pbtxt"),
            str(self.work_dir / "deep_dipole.pb"),
        )

        convert_pbtxt_to_pb(
            str(self.work_dir / "deeppolar.pbtxt"), str(self.work_dir / "deep_polar.pb")
        )

        with open(self.work_dir / "deeppolar.pbtxt") as f:
            deeppolar_pbtxt = f.read()

        # not an actual globalpolar and wfc model, but still good enough for testing factory
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write(deeppolar_pbtxt.replace("polar", "global_polar"))
            f.flush()
            convert_pbtxt_to_pb(f.name, str(self.work_dir / "deep_globalpolar.pb"))

        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write(deeppolar_pbtxt.replace("polar", "wfc"))
            f.flush()
            convert_pbtxt_to_pb(f.name, str(self.work_dir / "deep_wfc.pb"))

    def tearDown(self) -> None:
        for f in self.work_dir.glob("*.pb"):
            f.unlink()

    def test_factory(self) -> None:
        msg = "Returned wrong type of potential. Expected: {}, got: {}"

        dp = DeepPotential(self.work_dir / "deep_dipole.pb")
        self.assertIsInstance(dp, DeepDipole, msg.format(DeepDipole, type(dp)))

        dp = DeepPotential(self.work_dir / "deep_polar.pb")
        self.assertIsInstance(dp, DeepPolar, msg.format(DeepPolar, type(dp)))

        dp = DeepPotential(self.work_dir / "deep_pot.pb")
        self.assertIsInstance(dp, DeepPot, msg.format(DeepPot, type(dp)))

        dp = DeepPotential(self.work_dir / "deep_globalpolar.pb")
        self.assertIsInstance(
            dp, DeepGlobalPolar, msg.format(DeepGlobalPolar, type(dp))
        )

        dp = DeepPotential(self.work_dir / "deep_wfc.pb")
        self.assertIsInstance(dp, DeepWFC, msg.format(DeepWFC, type(dp)))
