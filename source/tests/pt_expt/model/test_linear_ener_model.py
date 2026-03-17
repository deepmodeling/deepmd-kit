# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import torch

from deepmd.pt_expt.model.dp_linear_model import (
    LinearEnergyModel,
)
from deepmd.pt_expt.model.get_model import (
    get_linear_model,
    get_standard_model,
)
from deepmd.pt_expt.utils import (
    env,
)

_sub_model_1 = {
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [3, 6],
        "axis_neuron": 2,
        "attn": 8,
        "attn_layer": 2,
        "attn_dotr": True,
        "attn_mask": False,
        "activation_function": "tanh",
        "scaling_factor": 1.0,
        "normalize": False,
        "temperature": 1.0,
        "set_davg_zero": True,
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [5, 5],
        "resnet_dt": True,
        "seed": 1,
    },
}
_sub_model_2 = copy.deepcopy(_sub_model_1)
_sub_model_2["descriptor"]["seed"] = 2
_sub_model_2["fitting_net"]["seed"] = 2

_type_map = ["O", "H"]


class TestLinearEnerWeights(unittest.TestCase):
    """Test that weights parameter affects energy, force, and virial."""

    def setUp(self) -> None:
        self.device = env.DEVICE

        # Build individual standard models for reference
        std_data_1 = copy.deepcopy(_sub_model_1)
        std_data_1["type_map"] = copy.deepcopy(_type_map)
        std_data_2 = copy.deepcopy(_sub_model_2)
        std_data_2["type_map"] = copy.deepcopy(_type_map)
        self.std_model_1 = get_standard_model(std_data_1)
        self.std_model_2 = get_standard_model(std_data_2)

        # Build linear models with different weights
        def _make_linear(weights):
            data = {
                "type_map": copy.deepcopy(_type_map),
                "models": [copy.deepcopy(_sub_model_1), copy.deepcopy(_sub_model_2)],
                "weights": weights,
            }
            return get_linear_model(data)

        self.model_mean = _make_linear("mean")
        self.model_sum = _make_linear("sum")
        self.model_custom = _make_linear([0.3, 0.7])

        # Sync sub-model weights so linear models use the same params as std models
        for linear_model in [self.model_mean, self.model_sum, self.model_custom]:
            linear_model.atomic_model.models[0].load_state_dict(
                self.std_model_1.atomic_model.state_dict()
            )
            linear_model.atomic_model.models[1].load_state_dict(
                self.std_model_2.atomic_model.state_dict()
            )

        # Test inputs
        generator = torch.Generator(device=self.device).manual_seed(20)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(
            3, dtype=torch.float64, device=self.device
        )
        self.cell = cell.unsqueeze(0)
        natoms = 6
        coord = torch.rand(
            [natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0)
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1, 1]], dtype=torch.int64, device=self.device
        )
        self.box = self.cell.reshape(1, 9)

    def _eval(self, model):
        coord = self.coord.clone().detach().requires_grad_(True)
        ret = model(
            coord,
            self.atype,
            box=self.box,
        )
        return {k: v.detach().cpu().numpy() for k, v in ret.items()}

    def test_mean_weights(self) -> None:
        ret1 = self._eval(self.std_model_1)
        ret2 = self._eval(self.std_model_2)
        ret_mean = self._eval(self.model_mean)
        for key in ["energy", "force", "virial"]:
            expected = 0.5 * ret1[key] + 0.5 * ret2[key]
            np.testing.assert_allclose(ret_mean[key], expected, atol=1e-10)

    def test_sum_weights(self) -> None:
        ret1 = self._eval(self.std_model_1)
        ret2 = self._eval(self.std_model_2)
        ret_sum = self._eval(self.model_sum)
        for key in ["energy", "force", "virial"]:
            expected = ret1[key] + ret2[key]
            np.testing.assert_allclose(ret_sum[key], expected, atol=1e-10)

    def test_custom_weights(self) -> None:
        ret1 = self._eval(self.std_model_1)
        ret2 = self._eval(self.std_model_2)
        ret_custom = self._eval(self.model_custom)
        for key in ["energy", "force", "virial"]:
            expected = 0.3 * ret1[key] + 0.7 * ret2[key]
            np.testing.assert_allclose(ret_custom[key], expected, atol=1e-10)


class TestLinearUpdateSel(unittest.TestCase):
    """Test that update_sel writes updated sub-model configs back."""

    @patch("deepmd.pt_expt.model.dp_linear_model.DPModelCommon.update_sel")
    def test_updated_sel_written_back(self, mock_update_sel) -> None:
        """Verify that update_sel returns configs with updated sel values."""

        def side_effect(train_data, type_map, sub_jdata):
            updated = copy.deepcopy(sub_jdata)
            updated["descriptor"]["sel"] = 99
            return updated, 0.5

        mock_update_sel.side_effect = side_effect

        local_jdata = {
            "type_map": ["O", "H"],
            "models": [
                {
                    "descriptor": {"type": "se_atten", "sel": 10, "rcut": 4.0},
                    "fitting_net": {"neuron": [5, 5]},
                },
                {
                    "descriptor": {"type": "se_atten", "sel": 10, "rcut": 4.0},
                    "fitting_net": {"neuron": [5, 5]},
                },
            ],
            "weights": "mean",
        }

        result, min_dist = LinearEnergyModel.update_sel(
            train_data=None,
            type_map=["O", "H"],
            local_jdata=local_jdata,
        )

        for idx, sub_model in enumerate(result["models"]):
            self.assertEqual(
                sub_model["descriptor"]["sel"],
                99,
                f"Sub-model {idx} sel was not updated in returned config",
            )


if __name__ == "__main__":
    unittest.main()
