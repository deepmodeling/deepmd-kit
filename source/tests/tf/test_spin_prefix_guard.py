# SPDX-License-Identifier: LGPL-3.0-or-later
"""The TF Spin helper must reject non-prefix use_spin layouts.

The legacy TensorFlow spin implementation assumes spin-enabled types form a
contiguous prefix of the type map: the SE-A ``sel`` extension takes the first
``ntypes_spin`` selections, and the coordinate/force splitting and bias merging
address the virtual block with a dense real->virtual offset. A non-prefix layout
such as ``use_spin=[False, True]`` silently reads the wrong real/virtual type
ranges (or raises deep inside the graph), so it must be rejected up front with a
clear error.
"""

import unittest

from deepmd.tf.utils.spin import (
    Spin,
)


class TestSpinPrefixGuard(unittest.TestCase):
    def test_non_prefix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Spin(use_spin=[False, True], spin_norm=[1.0], virtual_len=[0.4])

    def test_non_prefix_rejected_middle(self) -> None:
        with self.assertRaises(ValueError):
            Spin(
                use_spin=[True, False, True],
                spin_norm=[1.0, 1.0],
                virtual_len=[0.4, 0.4],
            )

    def test_prefix_accepted(self) -> None:
        # spin-enabled types first: the supported layout
        Spin(use_spin=[True, False], spin_norm=[1.0], virtual_len=[0.4])
        self.assertEqual(Spin(use_spin=[True, True]).ntypes_spin, 2)

    def test_all_non_spin_accepted(self) -> None:
        # no spin types at all is not a non-prefix violation
        self.assertEqual(Spin(use_spin=[False, False]).ntypes_spin, 0)


if __name__ == "__main__":
    unittest.main()
