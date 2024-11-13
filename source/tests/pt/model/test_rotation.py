# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest
from pathlib import (
    Path,
)
from typing import (
    Optional,
)

import numpy as np
import torch
from scipy.stats import (
    special_ortho_group,
)

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DeepmdData,
)


class CheckSymmetry(DeepmdData):
    def __init__(
        self,
        sys_path: str,
        type_map: Optional[list[str]] = None,
    ) -> None:
        super().__init__(sys_path=sys_path, type_map=type_map)
        self.add("energy", 1, atomic=False, must=False, high_prec=True)
        self.add("force", 3, atomic=True, must=False, high_prec=False)
        self.add("virial", 9, atomic=False, must=False, high_prec=False)

    def get_rotation(self, index, rotation_matrix):
        for i in range(
            0, len(self.dirs) + 1
        ):  # note: if different sets can be merged, prefix sum is unused to calculate
            if index < self.prefix_sum[i]:
                break
        frames = self._load_set(self.dirs[i - 1])
        frames["coord"] = np.dot(
            rotation_matrix, frames["coord"].reshape(-1, 3).T
        ).T.reshape(self.nframes, -1)
        frames["box"] = np.dot(
            rotation_matrix, frames["box"].reshape(-1, 3).T
        ).T.reshape(self.nframes, -1)
        frames["force"] = np.dot(
            rotation_matrix, frames["force"].reshape(-1, 3).T
        ).T.reshape(self.nframes, -1)
        frame = self._get_subdata(frames, index - self.prefix_sum[i - 1])
        frame = self.reformat_data_torch(frame)
        return frame


def get_data(batch):
    inputs = {}
    for key in ["coord", "atype", "box"]:
        inputs[key] = torch.as_tensor(batch[key], device=env.DEVICE)
        inputs[key] = inputs[key].unsqueeze(0).to(env.DEVICE)
    return inputs


class TestRotation(unittest.TestCase):
    def setUp(self) -> None:
        with open(str(Path(__file__).parent / "water/se_e2_a.json")) as fin:
            self.config = json.load(fin)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.rotation = special_ortho_group.rvs(3)
        with torch.device("cpu"):
            self.get_dataset(0)
        self.get_model()

    def get_model(self) -> None:
        self.model = get_model(self.config["model"]).to(env.DEVICE)

    def get_dataset(self, system_index=0, batch_index=0) -> None:
        systems = self.config["training"]["training_data"]["systems"]
        type_map = self.config["model"]["type_map"]
        dpdatasystem = CheckSymmetry(sys_path=systems[system_index], type_map=type_map)
        self.origin_batch = dpdatasystem.get_item_torch(batch_index)
        self.rotated_batch = dpdatasystem.get_rotation(batch_index, self.rotation)

    def test_rotation(self) -> None:
        result1 = self.model(**get_data(self.origin_batch))
        result2 = self.model(**get_data(self.rotated_batch))
        rotation = torch.from_numpy(self.rotation).to(env.DEVICE)
        torch.testing.assert_close(result1["energy"], result2["energy"])
        if "force" in result1:
            torch.testing.assert_close(
                result2["force"][0], torch.matmul(rotation, result1["force"][0].T).T
            )
        if "virial" in result1:
            torch.testing.assert_close(
                result2["virial"][0].view([3, 3]),
                torch.matmul(
                    torch.matmul(rotation, result1["virial"][0].view([3, 3]).T),
                    rotation.T,
                ),
            )


if __name__ == "__main__":
    unittest.main()
