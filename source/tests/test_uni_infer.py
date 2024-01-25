# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the universal Python inference interface."""

import os
import unittest

from common import (
    tests_path,
)

from deepmd.tf.infer.deep_pot import DeepPot as DeepPotTF
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)
from deepmd_utils.infer.deep_pot import DeepPot as DeepPot


class TestUniversalInfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        convert_pbtxt_to_pb(
            str(tests_path / os.path.join("infer", "deeppot-r.pbtxt")), "deeppot.pb"
        )

    def test_deep_pot(self):
        dp = DeepPot("deeppot.pb")
        self.assertIsInstance(dp, DeepPotTF)
