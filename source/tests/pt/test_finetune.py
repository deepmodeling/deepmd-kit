# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    change_energy_bias_lower,
)

from .model.test_permutation import (
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
)


class FinetuneTest:
    def test_finetune_change_energy_bias(self):
        # get model
        model = get_model(self.model_config)
        model.fitting_net.bias_atom_e = torch.rand_like(model.fitting_net.bias_atom_e)
        energy_bias_before = deepcopy(
            model.fitting_net.bias_atom_e.detach().cpu().numpy().reshape(-1)
        )
        bias_atom_e_input = deepcopy(
            model.fitting_net.bias_atom_e.detach().cpu().numpy().reshape(-1)
        )
        model = torch.jit.script(model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(model, tmp_model.name)
        dp = DeepEval(tmp_model.name)
        ntest = 10
        origin_type_map = ["O", "H"]
        full_type_map = ["O", "H", "B"]

        # change energy bias
        energy_bias_after = change_energy_bias_lower(
            self.data,
            dp,
            origin_type_map=origin_type_map,
            full_type_map=full_type_map,
            bias_atom_e=bias_atom_e_input,
            bias_shift="delta",
            ntest=ntest,
        )

        # get ground-truth energy bias change
        sorter = np.argsort(full_type_map)
        idx_type_map = sorter[
            np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
        ]
        test_data = self.data.get_test()
        atom_nums = np.tile(np.bincount(test_data["type"][0])[idx_type_map], (ntest, 1))
        energy = dp.eval(
            test_data["coord"][:ntest], test_data["box"][:ntest], test_data["type"][0]
        )[0]
        energy_diff = test_data["energy"][:ntest] - energy
        finetune_shift = (
            energy_bias_after[idx_type_map] - energy_bias_before[idx_type_map]
        )
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        # check values
        np.testing.assert_almost_equal(finetune_shift, ground_truth_shift, decimal=10)

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestEnergyModelSeA(unittest.TestCase, FinetuneTest):
    def setUp(self):
        self.data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.data = DeepmdDataSystem(
            self.data_file,
            batch_size=1,
            test_size=1,
        )
        self.data.add("energy", ndof=1, atomic=False, must=True, high_prec=True)
        self.model_config = model_se_e2_a

    def tearDown(self) -> None:
        FinetuneTest.tearDown(self)


class TestEnergyModelDPA1(unittest.TestCase, FinetuneTest):
    def setUp(self):
        self.data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.data = DeepmdDataSystem(
            self.data_file,
            batch_size=1,
            test_size=1,
        )
        self.data.add("energy", ndof=1, atomic=False, must=True, high_prec=True)
        self.model_config = model_dpa1

    def tearDown(self) -> None:
        FinetuneTest.tearDown(self)


class TestEnergyModelDPA2(unittest.TestCase, FinetuneTest):
    def setUp(self):
        self.data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.data = DeepmdDataSystem(
            self.data_file,
            batch_size=1,
            test_size=1,
        )
        self.data.add("energy", ndof=1, atomic=False, must=True, high_prec=True)
        self.model_config = model_dpa2

    def tearDown(self) -> None:
        FinetuneTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()