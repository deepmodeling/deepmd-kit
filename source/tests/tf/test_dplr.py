# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.tf.common import (
    clear_session,
)
from deepmd.tf.utils.convert import (
    convert_pbtxt_to_pb,
)


class TestDPLR(unittest.TestCase):
    def setUp(self) -> None:
        clear_session()
        # a bit strange path, need to move to the correct directory
        pbtxt_file = (
            Path(__file__).parent.parent.parent / "lmp" / "tests" / "lrmodel.pbtxt"
        )
        convert_pbtxt_to_pb(pbtxt_file, "lrmodel.pb")
        pbtxt_file = (
            Path(__file__).parent.parent.parent / "lmp" / "tests" / "lrdipole.pbtxt"
        )
        convert_pbtxt_to_pb(pbtxt_file, "lrdipole.pb")

        self.expected_e_lr_efield_variable = -35.713836
        self.expected_f_lr_efield_variable = np.array(
            [
                [1.479351, 0.430085, 2.234422],
                [-1.636212, 1.001423, 5.771218],
                [-1.046306, 5.068619, 0.037626],
                [1.043115, -0.01208, -3.296184],
                [1.819341, -4.589436, -1.725136],
                [-1.659289, -1.898612, -3.021946],
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
        e, f, v = dp.eval(self.coord[:, :6], self.box, self.atype[:6], atomic=False)
        np.testing.assert_allclose(e, self.expected_e_lr_efield_variable, atol=1e-6)
        np.testing.assert_allclose(
            f.ravel(), self.expected_f_lr_efield_variable.ravel(), atol=1e-6
        )
