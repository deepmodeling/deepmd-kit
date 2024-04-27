# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
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
from deepmd.utils.data import (
    DataRequirementItem,
)

from .model.test_permutation import (
    model_dpa2,
    model_se_e2_a,
    model_zbl,
)

energy_data_requirement = [
    DataRequirementItem(
        "energy",
        ndof=1,
        atomic=False,
        must=False,
        high_prec=True,
    ),
    DataRequirementItem(
        "force",
        ndof=3,
        atomic=True,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "virial",
        ndof=9,
        atomic=False,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "atom_ener",
        ndof=1,
        atomic=True,
        must=False,
        high_prec=False,
    ),
    DataRequirementItem(
        "atom_pref",
        ndof=1,
        atomic=True,
        must=False,
        high_prec=False,
        repeat=3,
    ),
]


class FinetuneTest:
    def test_finetune_change_out_bias(self):
        # get model
        model = get_model(self.model_config)
        atomic_model = model.atomic_model
        atomic_model["out_bias"] = torch.rand_like(atomic_model["out_bias"])
        energy_bias_before = to_numpy_array(atomic_model["out_bias"])[0].ravel()

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
        )
        energy_bias_after = to_numpy_array(atomic_model["out_bias"])[0].ravel()

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
        energy_diff = to_numpy_array(self.sampled[0]["energy"][:ntest]) - energy
        finetune_shift = (
            energy_bias_after[idx_type_map] - energy_bias_before[idx_type_map]
        )
        ground_truth_shift = np.linalg.lstsq(atom_nums, energy_diff, rcond=None)[
            0
        ].reshape(-1)

        # check values
        np.testing.assert_almost_equal(finetune_shift, ground_truth_shift, decimal=10)


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


if __name__ == "__main__":
    unittest.main()
