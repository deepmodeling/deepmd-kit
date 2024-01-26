# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.entrypoints.main import (
    main,
)


class TestLKF(unittest.TestCase):
    def test_lkf(self):
        main(["train", "tests/water/lkf.json"])


if __name__ == "__main__":
    unittest.main()
