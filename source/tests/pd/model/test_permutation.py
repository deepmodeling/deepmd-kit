# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import os
import unittest

import numpy as np
import paddle

from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)
from ..common import (
    eval_model,
)

CUR_DIR = os.path.dirname(__file__)

dtype = paddle.float64

model_se_e2_a = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
}

model_dos = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
        "type": "dos",
        "numb_dos": 250,
    },
    "data_stat_nbatch": 20,
}

model_zbl = {
    "type_map": ["O", "H", "B"],
    "use_srtab": f"{CUR_DIR}/water/data/zbl_tab_potential/H2O_tab_potential.txt",
    "smin_alpha": 0.1,
    "sw_rmin": 0.2,
    "sw_rmax": 4.0,
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "attn": 64,
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
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
}


model_spin = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92, 4],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
    "data_stat_nbatch": 20,
    "spin": {
        "use_spin": [True, False, False],
        "virtual_scale": [0.3140],
        "_comment": " that's all",
    },
}

model_dpa2 = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "dpa2",
        "repinit": {
            "rcut": 6.0,
            "rcut_smth": 2.0,
            "nsel": 100,
            "neuron": [2, 4, 8],
            "axis_neuron": 4,
            "activation_function": "tanh",
        },
        "repformer": {
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "nsel": 40,
            "nlayers": 12,
            "g1_dim": 8,
            "g2_dim": 5,
            "attn2_hidden": 3,
            "attn2_nhead": 1,
            "attn1_hidden": 5,
            "attn1_nhead": 1,
            "axis_neuron": 4,
            "update_h2": False,
            "update_g1_has_conv": True,
            "update_g1_has_grrg": True,
            "update_g1_has_drrd": True,
            "update_g1_has_attn": True,
            "update_g2_has_g1g1": True,
            "update_g2_has_attn": True,
            "attn2_has_gate": True,
        },
        "seed": 1,
        "add_tebd_to_repinit_out": False,
    },
    "fitting_net": {
        "neuron": [24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
}

model_dpa3 = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "dpa3",
        "repflow": {
            "n_dim": 20,
            "e_dim": 10,
            "a_dim": 8,
            "nlayers": 6,
            "e_rcut": 6.0,
            "e_rcut_smth": 3.0,
            "e_sel": 20,
            "a_rcut": 4.0,
            "a_rcut_smth": 2.0,
            "a_sel": 10,
            "axis_neuron": 4,
            "a_compress_rate": 1,
            "a_compress_e_rate": 2,
            "a_compress_use_split": True,
            "update_angle": True,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
            "smooth_edge_update": True,
        },
        "activation_function": "silut:10.0",
        "use_tebd_bias": False,
        "precision": "float32",
        "concat_output_tebd": False,
    },
    "fitting_net": {
        "neuron": [24, 24],
        "resnet_dt": True,
        "precision": "float32",
        "activation_function": "silut:10.0",
        "seed": 1,
    },
}

model_dpa2tebd = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "dpa2",
        "repinit": {
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "nsel": 100,
            "neuron": [2, 4, 8],
            "axis_neuron": 4,
            "activation_function": "tanh",
            "three_body_sel": 40,
            "three_body_rcut": 4.0,
            "three_body_rcut_smth": 3.5,
            "use_three_body": True,
        },
        "repformer": {
            "rcut": 4.0,
            "rcut_smth": 0.5,
            "nsel": 40,
            "nlayers": 6,
            "g1_dim": 8,
            "g2_dim": 5,
            "attn2_hidden": 3,
            "attn2_nhead": 1,
            "attn1_hidden": 5,
            "attn1_nhead": 1,
            "axis_neuron": 4,
            "update_h2": False,
            "update_g1_has_conv": True,
            "update_g1_has_grrg": True,
            "update_g1_has_drrd": True,
            "update_g1_has_attn": False,
            "update_g2_has_g1g1": False,
            "update_g2_has_attn": True,
            "update_style": "res_residual",
            "update_residual": 0.01,
            "update_residual_init": "norm",
            "attn2_has_gate": True,
            "use_sqrt_nnei": True,
            "g1_out_conv": True,
            "g1_out_mlp": True,
        },
        "seed": 1,
        "add_tebd_to_repinit_out": False,
    },
    "fitting_net": {
        "neuron": [24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
}

model_dpa1 = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_atten",
        "sel": 40,
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "attn": 64,
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
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "seed": 1,
    },
}


model_hybrid = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "hybrid",
        "list": [
            {
                "type": "se_atten",
                "sel": 120,
                "rcut_smth": 0.5,
                "rcut": 6.0,
                "neuron": [25, 50, 100],
                "axis_neuron": 16,
                "attn": 128,
                "attn_layer": 0,
                "attn_dotr": True,
                "attn_mask": False,
                "activation_function": "tanh",
                "scaling_factor": 1.0,
                "normalize": True,
                "temperature": 1.0,
                "seed": 1,
            },
            {
                "type": "dpa2",
                "repinit": {
                    "rcut": 6.0,
                    "rcut_smth": 2.0,
                    "nsel": 30,
                    "neuron": [2, 4, 8],
                    "axis_neuron": 4,
                    "activation_function": "tanh",
                },
                "repformer": {
                    "rcut": 4.0,
                    "rcut_smth": 0.5,
                    "nsel": 10,
                    "nlayers": 12,
                    "g1_dim": 8,
                    "g2_dim": 5,
                    "attn2_hidden": 3,
                    "attn2_nhead": 1,
                    "attn1_hidden": 5,
                    "attn1_nhead": 1,
                    "axis_neuron": 4,
                    "update_h2": False,
                    "update_g1_has_conv": True,
                    "update_g1_has_grrg": True,
                    "update_g1_has_drrd": True,
                    "update_g1_has_attn": True,
                    "update_g2_has_g1g1": True,
                    "update_g2_has_attn": True,
                    "attn2_has_gate": True,
                },
                "seed": 1,
                "add_tebd_to_repinit_out": False,
            },
        ],
    },
    "fitting_net": {
        "neuron": [240, 240, 240],
        "resnet_dt": True,
        "seed": 1,
        "_comment": " that's all",
    },
    "_comment": " that's all",
}

model_property = {
    "type_map": ["H", "C", "N", "O"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [3, 3, 3, 3],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [25, 50, 100],
        "resnet_dt": False,
        "axis_neuron": 16,
        "seed": 1,
    },
    "fitting_net": {
        "type": "property",
        "property_name": "band_property",
        "task_dim": 3,
        "neuron": [24, 24, 24],
        "resnet_dt": True,
        "intensive": True,
        "seed": 1,
    },
}


class PermutationTest:
    def test(
        self,
    ) -> None:
        natoms = 5
        generator = paddle.seed(GLOBAL_SEED)
        cell = paddle.rand([3, 3], dtype=dtype)
        cell = (cell + cell.T) + 5.0 * paddle.eye(3)
        coord = paddle.rand([natoms, 3], dtype=dtype)
        spin = paddle.rand([natoms, 3], dtype=dtype)
        coord = paddle.matmul(coord, cell)
        atype = paddle.to_tensor([0, 0, 0, 1, 1], dtype=paddle.int32)
        idx_perm = [1, 0, 4, 3, 2]
        test_spin = getattr(self, "test_spin", False)
        if not test_spin:
            test_keys = ["energy", "force", "virial"]
        else:
            test_keys = ["energy", "force", "force_mag", "virial"]
        result_0 = eval_model(
            self.model,
            coord.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            spins=spin.unsqueeze(0),
        )
        ret0 = {key: result_0[key].squeeze(0) for key in test_keys}
        result_1 = eval_model(
            self.model,
            coord[idx_perm].unsqueeze(0),
            cell.unsqueeze(0),
            atype[idx_perm],
            spins=spin[idx_perm].unsqueeze(0),
        )
        ret1 = {key: result_1[key].squeeze(0) for key in test_keys}
        prec = 1e-10
        for key in test_keys:
            if key in ["energy"]:
                np.testing.assert_allclose(
                    ret0[key].numpy(), ret1[key].numpy(), rtol=prec, atol=prec
                )
            elif key in ["force", "force_mag"]:
                np.testing.assert_allclose(
                    ret0[key][idx_perm].numpy(), ret1[key].numpy(), rtol=prec, atol=prec
                )
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    np.testing.assert_allclose(
                        ret0[key], ret1[key], rtol=prec, atol=prec
                    )
            else:
                raise RuntimeError(f"Unexpected test key {key}")


class TestEnergyModelSeA(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestDOSModelSeA(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dos)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA2(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestForceModelDPA2(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelHybrid(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestForceModelHybrid(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hybrid)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelZBL(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelSpinSeA(unittest.TestCase, PermutationTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_spin)
        self.type_split = False
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


# class TestEnergyFoo(unittest.TestCase):
#   def test(self):
#     model_params = model_dpau
#     self.model = EnergyModelDPAUni(model_params).to(env.DEVICE)

#     natoms = 5
#     cell = paddle.rand([3, 3], dtype=dtype)
#     cell = (cell + cell.T) + 5. * paddle.eye(3)
#     coord = paddle.rand([natoms, 3], dtype=dtype)
#     coord = paddle.matmul(coord, cell)
#     atype = paddle.to_tensor([0, 0, 0, 1, 1])
#     idx_perm = [1, 0, 4, 3, 2]
#     ret0 = infer_model(self.model, coord, cell, atype, type_split=True)


if __name__ == "__main__":
    unittest.main()
