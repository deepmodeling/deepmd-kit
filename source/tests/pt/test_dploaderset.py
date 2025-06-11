# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest
from pathlib import (
    Path,
)

from deepmd.common import (
    expand_sys_str,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)


class TestSampler(unittest.TestCase):
    def setUp(self) -> None:
        with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
            content = fin.read()
        config = json.loads(content)
        data_file = [
            str(Path(__file__).parent / "model/water/data/data_0"),
        ]
        config["training"]["training_data"]["systems"] = data_file
        config["training"]["validation_data"]["systems"] = data_file
        model_config = config["model"]
        self.rcut = model_config["descriptor"]["rcut"]
        self.rcut_smth = model_config["descriptor"]["rcut_smth"]
        self.sel = model_config["descriptor"]["sel"]
        self.batch_size = config["training"]["training_data"]["batch_size"]
        self.systems = config["training"]["validation_data"]["systems"]
        self.type_map = model_config["type_map"]
        if isinstance(self.systems, str):
            self.systems = expand_sys_str(self.systems)

    def get_batch_sizes(self, batch_size) -> int:
        dataset = DpLoaderSet(
            self.systems,
            batch_size,
            self.type_map,
            seed=10,
            shuffle=False,
        )
        return dataset.batch_sizes[0]

    def test_batchsize(self) -> None:
        # 192 atoms, 1 system
        assert len(self.systems) == 1

        # test: batch_size:int
        self.assertEqual(self.get_batch_sizes(3), 3)

        # test: batch_size:list[int]
        self.assertEqual(self.get_batch_sizes([3]), 3)

        # test: batch_size:str = "auto"
        self.assertEqual(self.get_batch_sizes("auto:384"), 2)
        self.assertEqual(self.get_batch_sizes("auto:383"), 2)
        self.assertEqual(self.get_batch_sizes("auto:193"), 2)
        self.assertEqual(self.get_batch_sizes("auto:192"), 1)
        self.assertEqual(self.get_batch_sizes("auto:191"), 1)
        self.assertEqual(self.get_batch_sizes("auto:32"), 1)
        self.assertEqual(self.get_batch_sizes("auto"), 1)

        # test: batch_size:str = "max"
        self.assertEqual(self.get_batch_sizes("max:384"), 2)
        self.assertEqual(self.get_batch_sizes("max:383"), 1)
        self.assertEqual(self.get_batch_sizes("max:193"), 1)
        self.assertEqual(self.get_batch_sizes("max:192"), 1)
        self.assertEqual(self.get_batch_sizes("max:191"), 1)

        # test: batch_size:str = "filter"
        self.assertEqual(self.get_batch_sizes("filter:193"), 1)
        self.assertEqual(self.get_batch_sizes("filter:192"), 1)
        with self.assertLogs(logger="deepmd") as cm:
            self.assertRaises(ValueError, self.get_batch_sizes, "filter:191")
        self.assertIn("Remove 1 systems with more than 191 atoms", cm.output[-1])

        # test: unknown batch_size: str
        with self.assertRaises(ValueError) as context:
            self.get_batch_sizes("unknown")
        self.assertIn("Unsupported batch size rule: unknown", str(context.exception))


if __name__ == "__main__":
    unittest.main()
