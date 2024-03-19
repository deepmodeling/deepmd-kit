# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.infer.deep_eval import (
    eval_model,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)

from .test_permutation import (  # model_dpau,
    model_dos,
    model_dpa1,
    model_dpa2,
    model_hybrid,
    model_se_e2_a,
    model_spin,
    model_zbl,
)

dtype = torch.float64


class RotTest:
    def test(
        self,
    ):
        prec = 1e-10
        natoms = 5
        cell = 10.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        coord = 2 * torch.rand([natoms, 3], dtype=dtype, device=env.DEVICE)
        spin = 2 * torch.rand([natoms, 3], dtype=dtype, device=env.DEVICE)
        shift = torch.tensor([4, 4, 4], dtype=dtype, device=env.DEVICE)
        atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE)
        from scipy.stats import (
            special_ortho_group,
        )

        test_spin = getattr(self, "test_spin", False)
        if not test_spin:
            test_keys = ["energy", "force", "virial"]
        else:
            test_keys = ["energy", "force", "force_mag"]
        rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype, device=env.DEVICE)

        # rotate only coord and shift to the center of cell
        coord_rot = torch.matmul(coord, rmat)
        spin_rot = torch.matmul(spin, rmat)
        result_0 = eval_model(
            self.model,
            (coord + shift).unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            spins=spin.unsqueeze(0),
        )
        ret0 = {key: result_0[key].squeeze(0) for key in test_keys}
        result_1 = eval_model(
            self.model,
            (coord_rot + shift).unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            spins=spin_rot.unsqueeze(0),
        )
        ret1 = {key: result_1[key].squeeze(0) for key in test_keys}
        for key in test_keys:
            if key in ["energy"]:
                torch.testing.assert_close(ret0[key], ret1[key], rtol=prec, atol=prec)
            elif key in ["force", "force_mag"]:
                torch.testing.assert_close(
                    torch.matmul(ret0[key], rmat), ret1[key], rtol=prec, atol=prec
                )
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    torch.testing.assert_close(
                        torch.matmul(
                            rmat.T, torch.matmul(ret0[key].view([3, 3]), rmat)
                        ),
                        ret1[key].view([3, 3]),
                        rtol=prec,
                        atol=prec,
                    )
            else:
                raise RuntimeError(f"Unexpected test key {key}")
        # rotate coord and cell
        torch.manual_seed(0)
        cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        coord = torch.rand([natoms, 3], dtype=dtype, device=env.DEVICE)
        coord = torch.matmul(coord, cell)
        spin = torch.rand([natoms, 3], dtype=dtype, device=env.DEVICE)
        atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE)
        coord_rot = torch.matmul(coord, rmat)
        spin_rot = torch.matmul(spin, rmat)
        cell_rot = torch.matmul(cell, rmat)
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
            coord_rot.unsqueeze(0),
            cell_rot.unsqueeze(0),
            atype,
            spins=spin_rot.unsqueeze(0),
        )
        ret1 = {key: result_1[key].squeeze(0) for key in test_keys}
        for key in test_keys:
            if key in ["energy"]:
                torch.testing.assert_close(ret0[key], ret1[key], rtol=prec, atol=prec)
            elif key in ["force", "force_mag"]:
                torch.testing.assert_close(
                    torch.matmul(ret0[key], rmat), ret1[key], rtol=prec, atol=prec
                )
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    torch.testing.assert_close(
                        torch.matmul(
                            rmat.T, torch.matmul(ret0[key].view([3, 3]), rmat)
                        ),
                        ret1[key].view([3, 3]),
                        rtol=prec,
                        atol=prec,
                    )


class TestEnergyModelSeA(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestDOSModelSeA(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dos)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA2(unittest.TestCase, RotTest):
    def setUp(self):
        model_params_sample = copy.deepcopy(model_dpa2)
        model_params_sample["descriptor"]["rcut"] = model_params_sample["descriptor"][
            "repinit_rcut"
        ]
        model_params_sample["descriptor"]["sel"] = model_params_sample["descriptor"][
            "repinit_nsel"
        ]
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestForceModelDPA2(unittest.TestCase, RotTest):
    def setUp(self):
        model_params_sample = copy.deepcopy(model_dpa2)
        model_params_sample["descriptor"]["rcut"] = model_params_sample["descriptor"][
            "repinit_rcut"
        ]
        model_params_sample["descriptor"]["sel"] = model_params_sample["descriptor"][
            "repinit_nsel"
        ]
        model_params = copy.deepcopy(model_dpa2)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelHybrid(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestForceModelHybrid(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelZBL(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinSeA(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_spin)
        self.type_split = False
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
