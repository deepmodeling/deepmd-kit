# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import (
    Path,
)

from deepmd.pt.entrypoints.main import (
    main,
)


class TestLKF(unittest.TestCase):
    def test_lkf(self):
        main(["train", str(Path(__file__).parent / "water/lkf.json")])


if __name__ == "__main__":
    unittest.main()
