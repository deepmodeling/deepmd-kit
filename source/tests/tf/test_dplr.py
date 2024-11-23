# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)


class TestDPLR(unittest.TestCase):
    def setUp(self) -> None:
        # a bit strange path, need to move to the correct directory
        pbtxt_file = (
            Path(__file__).parent.parent.parent / "lmp" / "tests" / "lrmodel.pbtxt"
        )
        convert_pbtxt_to_pb(pbtxt_file, "lrmodel.pb")

        self.expected_e_lr_efield_variable = -40.56538550
        self.expected_f_lr_efield_variable = np.array(
            [
                [0.35019748, 0.27802691, -0.38443156],
                [-0.42115581, -0.20474826, -0.02701100],
                [-0.56357653, 0.34154004, 0.78389512],
                [0.21023870, -0.60684635, -0.39875165],
                [0.78732106, 0.00610023, 0.17197636],
                [-0.36302488, 0.18592742, -0.14567727],
            ]
        )

        self.box = np.eye(3).reshape(1, 9) * 20.0
        self.coord = np.array(
            [
                [1.25545000, 1.27562200, 0.98873000],
                [0.96101000, 3.25750000, 1.33494000],
                [0.66417000, 1.31153700, 1.74354000],
                [1.29187000, 0.33436000, 0.73085000],
                [1.88885000, 3.51130000, 1.42444000],
                [0.51617000, 4.04330000, 0.90904000],
                [1.25545000, 1.27562200, 0.98873000],
                [0.96101000, 3.25750000, 1.33494000],
            ]
        ).reshape(1, 8, 3)
        self.atype = np.array([0, 0, 1, 1, 1, 1, 2, 2])

    def test_eval(self) -> None:
        dp = DeepPot("lrmodel.pb")
        e, f, v, ae, av = dp.eval(
            self.coord[:, :6], self.box, self.atype[:6], atomic=True
        )
        np.testing.assert_allclose(e, self.expected_e_lr_efield_variable, atol=1e-6)
        np.testing.assert_allclose(
            f.ravel(), self.expected_f_lr_efield_variable.ravel(), atol=1e-6
        )
