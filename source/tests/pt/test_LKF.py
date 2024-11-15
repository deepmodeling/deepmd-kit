# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from pathlib import (
    Path,
)

from deepmd.pt.entrypoints.main import (
    main,
)


class TestLKF(unittest.TestCase):
    def test_lkf(self) -> None:
        with open(str(Path(__file__).parent / "water/lkf.json")) as fin:
            content = fin.read()
        self.config = json.loads(content)
        self.config["training"]["training_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/data_0")
        ]
        self.config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/data_0")
        ]
        self.input_json = "test_lkf.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)
        main(["train", self.input_json])

    def tearDown(self) -> None:
        os.remove(self.input_json)


if __name__ == "__main__":
    unittest.main()
