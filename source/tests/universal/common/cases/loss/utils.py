# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np

from deepmd.utils.data import (
    DataRequirementItem,
)

from .....seed import (
    GLOBAL_SEED,
)


class LossTestCase:
    """Common test case for loss function."""

    def setUp(self):
        pass

    def test_label_keys(self):
        module = self.forward_wrapper(self.module)
        label_requirement = self.module.label_requirement
        label_dict = {item.key: item for item in label_requirement}
        label_keys = sorted(label_dict.keys())
        label_keys_expected = sorted(
            [key for key in self.key_to_pref_map if self.key_to_pref_map[key] > 0]
        )
        np.testing.assert_equal(label_keys_expected, label_keys)

    def test_forward(self):
        module = self.forward_wrapper(self.module)
        label_requirement = self.module.label_requirement
        label_dict = {item.key: item for item in label_requirement}
        label_keys = sorted(label_dict.keys())
        natoms = 5
        nframes = 2

        def fake_model():
            model_predict = {
                data_key: fake_input(
                    label_dict[data_key], natoms=natoms, nframes=nframes
                )
                for data_key in label_keys
            }
            if "atom_ener" in model_predict:
                model_predict["atom_energy"] = model_predict.pop("atom_ener")
            model_predict.update(
                {"mask_mag": np.ones([nframes, natoms, 1], dtype=np.bool_)}
            )
            return model_predict

        labels = {
            data_key: fake_input(label_dict[data_key], natoms=natoms, nframes=nframes)
            for data_key in label_keys
        }
        labels.update({"find_" + data_key: 1.0 for data_key in label_keys})

        _, loss, more_loss = module(
            {},
            fake_model,
            labels,
            natoms,
            1.0,
        )


def fake_input(data_item: DataRequirementItem, natoms=5, nframes=2) -> np.ndarray:
    ndof = data_item.ndof
    atomic = data_item.atomic
    repeat = data_item.repeat
    rng = np.random.default_rng(seed=GLOBAL_SEED)
    dtype = data_item.dtype if data_item.dtype is not None else np.float64
    if atomic:
        data = rng.random([nframes, natoms, ndof], dtype)
    else:
        data = rng.random([nframes, ndof], dtype)
    if repeat != 1:
        data = np.repeat(data, repeat).reshape([nframes, -1])
    return data
