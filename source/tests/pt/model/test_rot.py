# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.infer.deep_eval import (
    eval_model,
)
from deepmd.pt.model.model import (
    get_model,
    get_zbl_model,
)
from deepmd.pt.utils import (
    env,
)

from .test_permutation import (  # model_dpau,
    model_dpa1,
    model_dpa2,
    model_hybrid,
    model_se_e2_a,
    model_zbl,
)

dtype = torch.float64


class RotTest:
    def test(
        self,
    ):
        prec = 1e-10
        natoms = 5
        cell = 10.0 * torch.eye(3, dtype=dtype).to(env.DEVICE)
        coord = 2 * torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
        shift = torch.tensor([4, 4, 4], dtype=dtype).to(env.DEVICE)
        atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
        from scipy.stats import (
            special_ortho_group,
        )

        rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype).to(env.DEVICE)

        # rotate only coord and shift to the center of cell
        coord_rot = torch.matmul(coord, rmat)
        e0, f0, v0 = eval_model(
            self.model, (coord + shift).unsqueeze(0), cell.unsqueeze(0), atype
        )
        ret0 = {
            "energy": e0.squeeze(0),
            "force": f0.squeeze(0),
            "virial": v0.squeeze(0),
        }
        e1, f1, v1 = eval_model(
            self.model, (coord_rot + shift).unsqueeze(0), cell.unsqueeze(0), atype
        )
        ret1 = {
            "energy": e1.squeeze(0),
            "force": f1.squeeze(0),
            "virial": v1.squeeze(0),
        }
        torch.testing.assert_close(ret0["energy"], ret1["energy"], rtol=prec, atol=prec)
        torch.testing.assert_close(
            torch.matmul(ret0["force"], rmat), ret1["force"], rtol=prec, atol=prec
        )
        if not hasattr(self, "test_virial") or self.test_virial:
            torch.testing.assert_close(
                torch.matmul(rmat.T, torch.matmul(ret0["virial"].view([3, 3]), rmat)),
                ret1["virial"].view([3, 3]),
                rtol=prec,
                atol=prec,
            )

        # rotate coord and cell
        torch.manual_seed(0)
        cell = torch.rand([3, 3], dtype=dtype).to(env.DEVICE)
        cell = (cell + cell.T) + 5.0 * torch.eye(3).to(env.DEVICE)
        coord = torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
        coord_rot = torch.matmul(coord, rmat)
        cell_rot = torch.matmul(cell, rmat)
        e0, f0, v0 = eval_model(
            self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype
        )
        ret0 = {
            "energy": e0.squeeze(0),
            "force": f0.squeeze(0),
            "virial": v0.squeeze(0),
        }
        e1, f1, v1 = eval_model(
            self.model, coord_rot.unsqueeze(0), cell_rot.unsqueeze(0), atype
        )
        ret1 = {
            "energy": e1.squeeze(0),
            "force": f1.squeeze(0),
            "virial": v1.squeeze(0),
        }
        torch.testing.assert_close(ret0["energy"], ret1["energy"], rtol=prec, atol=prec)
        torch.testing.assert_close(
            torch.matmul(ret0["force"], rmat), ret1["force"], rtol=prec, atol=prec
        )
        if not hasattr(self, "test_virial") or self.test_virial:
            torch.testing.assert_close(
                torch.matmul(rmat.T, torch.matmul(ret0["virial"].view([3, 3]), rmat)),
                ret1["virial"].view([3, 3]),
                rtol=prec,
                atol=prec,
            )


class TestEnergyModelSeA(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_se_e2_a)
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


@unittest.skip("hybrid not supported at the moment")
class TestEnergyModelHybrid(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("hybrid not supported at the moment")
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
        sampled = make_sample(model_params)
        self.type_split = False
        self.model = get_zbl_model(model_params, sampled).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
