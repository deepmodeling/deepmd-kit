# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import json
import unittest
from pathlib import (
    Path,
)
from typing import (
    Optional,
)

import numpy as np
import paddle

from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.utils.data import (
    DeepmdData,
)

from ...seed import (
    GLOBAL_SEED,
)


class CheckSymmetry(DeepmdData):
    def __init__(
        self,
        sys_path: str,
        type_map: Optional[list[str]] = None,
    ):
        super().__init__(sys_path=sys_path, type_map=type_map)
        self.add("energy", 1, atomic=False, must=False, high_prec=True)
        self.add("force", 3, atomic=True, must=False, high_prec=False)
        self.add("virial", 9, atomic=False, must=False, high_prec=False)

    def get_disturb(self, index, atom_index, axis_index, delta):
        for i in range(
            0, len(self.dirs) + 1
        ):  # note: if different sets can be merged, prefix sum is unused to calculate
            if index < self.prefix_sum[i]:
                break
        frames = self._load_set(self.dirs[i - 1])
        tmp = copy.deepcopy(frames["coord"].reshape(self.nframes, -1, 3))
        tmp[:, atom_index, axis_index] += delta
        frames["coord"] = tmp
        frame = self._get_subdata(frames, index - self.prefix_sum[i - 1])
        frame = self.reformat_data_torch(frame)
        return frame


def get_data(batch):
    inputs = {}
    for key in ["coord", "atype", "box"]:
        inputs[key] = batch[key].unsqueeze(0).to(env.DEVICE)
    return inputs


class TestForceGrad(unittest.TestCase):
    def setUp(self):
        with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
            self.config = json.load(fin)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.system_index = 0
        self.batch_index = 0
        self.get_dataset(self.system_index, self.batch_index)
        self.get_model()

    def get_model(self):
        self.model = get_model(self.config["model"]).to(env.DEVICE)

    def get_dataset(self, system_index=0, batch_index=0):
        systems = self.config["training"]["training_data"]["systems"]
        rcut = self.config["model"]["descriptor"]["rcut"]
        sel = self.config["model"]["descriptor"]["sel"]
        sec = paddle.cumsum(paddle.to_tensor(sel), axis=0)
        type_map = self.config["model"]["type_map"]
        self.dpdatasystem = CheckSymmetry(
            sys_path=systems[system_index], type_map=type_map
        )
        self.origin_batch = self.dpdatasystem.get_item_paddle(batch_index)

    @unittest.skip("it can be replaced by autodiff")
    def test_force_grad(self, threshold=1e-2, delta0=1e-6, seed=20):
        rng = np.random.default_rng(GLOBAL_SEED)
        result0 = self.model(**get_data(self.origin_batch))
        np.random.default_rng(seed)
        errors = np.zeros((self.dpdatasystem.natoms, 3))
        for atom_index in range(self.dpdatasystem.natoms):
            for axis_index in range(3):
                delta = rng.random() * delta0
                disturb_batch = self.dpdatasystem.get_disturb(
                    self.batch_index, atom_index, axis_index, delta
                )
                disturb_result = self.model(**get_data(disturb_batch))
                disturb_force = -(disturb_result["energy"] - result0["energy"]) / delta
                disturb_error = (
                    result0["force"][0, atom_index, axis_index] - disturb_force
                )
                errors[atom_index, axis_index] = disturb_error.detach().cpu().numpy()
        self.assertTrue(np.abs(errors).max() < threshold, msg=str(np.abs(errors).max()))


if __name__ == "__main__":
    unittest.main()
