# SPDX-License-Identifier: LGPL-3.0-or-later
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
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .model.test_permutation import (
    model_dpa2,
    model_se_e2_a,
    model_zbl,
    model_dos_bias,
)
from .test_stat import (
    dos_data_requirement,
    energy_data_requirement,
)


class FinetuneTest:
    def test_finetune_change_out_bias(self):
        # get model
        model = get_model(self.model_config)
        fitting_net = model.get_fitting_net()
        fitting_net["bias_atom_e"] = torch.rand_like(fitting_net["bias_atom_e"])
        energy_bias_before = deepcopy(to_numpy_array(fitting_net["bias_atom_e"]))

        # prepare original model for test
        dp = torch.jit.script(model)
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        torch.jit.save(dp, tmp_model.name)
        dp = DeepEval(tmp_model.name)
        origin_type_map = ["O", "H"]
        full_type_map = ["O", "H", "B"]

        # change energy bias
        model.atomic_model.change_out_bias(
            self.sampled,
            bias_adjust_mode="change-by-statistic",
            origin_type_map=origin_type_map,
            full_type_map=full_type_map,
        )
        energy_bias_after = deepcopy(to_numpy_array(fitting_net["bias_atom_e"]))

        # get ground-truth energy bias change
        sorter = np.argsort(full_type_map)
        idx_type_map = sorter[
            np.searchsorted(full_type_map, origin_type_map, sorter=sorter)
        ]
        ntest = 1
        atom_nums = np.tile(
            np.bincount(to_numpy_array(self.sampled[0]["atype"][0]))[idx_type_map],
            (ntest, 1),
        )
        energy = dp.eval(
            to_numpy_array(self.sampled[0]["coord"][:ntest]),
            to_numpy_array(self.sampled[0]["box"][:ntest]),
            to_numpy_array(self.sampled[0]["atype"][0]),
        )[0]
        energy_diff = to_numpy_array(self.sampled[0][self.var_name][:ntest]) - energy
        finetune_shift = (
            energy_bias_after[idx_type_map] - energy_bias_before[idx_type_map]
        )
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[0]
        # check values
        np.testing.assert_almost_equal(finetune_shift, ground_truth_shift, decimal=10)


class TestDOSModelSeA(unittest.TestCase, FinetuneTest):
    def setUp(self):
        self.data_file = [str(Path(__file__).parent / "dos/data/global_system")]
        self.model_config = model_dos_bias
        self.data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.model_config["type_map"],
        )
        self.data.add_data_requirement(dos_data_requirement)
        self.sampled = make_stat_input(
            self.data.systems,
            self.data.dataloaders,
            nbatches=1,
        )
        self.var_name = "dos"


class TestEnergyModelSeA(unittest.TestCase, FinetuneTest):
    def setUp(self):
        self.data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.model_config = model_se_e2_a
        self.data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.model_config["type_map"],
        )
        self.data.add_data_requirement(energy_data_requirement)
        self.sampled = make_stat_input(
            self.data.systems,
            self.data.dataloaders,
            nbatches=1,
        )
        self.var_name = "energy"


@unittest.skip("change bias not implemented yet.")
class TestEnergyZBLModelSeA(unittest.TestCase, FinetuneTest):
    def setUp(self):
        self.data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.model_config = model_zbl
        self.data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.model_config["type_map"],
        )
        self.data.add_data_requirement(energy_data_requirement)
        self.sampled = make_stat_input(
            self.data.systems,
            self.data.dataloaders,
            nbatches=1,
        )
        self.var_name = "energy"


class TestEnergyModelDPA2(unittest.TestCase, FinetuneTest):
    def setUp(self):
        self.data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.model_config = model_dpa2
        self.data = DpLoaderSet(
            self.data_file,
            batch_size=1,
            type_map=self.model_config["type_map"],
        )
        self.data.add_data_requirement(energy_data_requirement)
        self.sampled = make_stat_input(
            self.data.systems,
            self.data.dataloaders,
            nbatches=1,
        )
        self.var_name = "energy"


if __name__ == "__main__":
    unittest.main()
