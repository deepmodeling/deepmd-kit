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
    model_dpa1,
    model_dpa2,
    model_hybrid,
    model_se_e2_a,
    model_zbl,
)

dtype = torch.float64


class SmoothTest:
    def test(
        self,
    ):
        # displacement of atoms
        epsilon = 1e-5 if self.epsilon is None else self.epsilon
        # required prec. relative prec is not checked.
        rprec = 0
        aprec = 1e-5 if self.aprec is None else self.aprec

        natoms = 10
        cell = 8.6 * torch.eye(3, dtype=dtype).to(env.DEVICE)
        atype = torch.randint(0, 3, [natoms])
        coord0 = (
            torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    4.0 - 0.5 * epsilon,
                    0.0,
                    0.0,
                    0.0,
                    4.0 - 0.5 * epsilon,
                    0.0,
                ],
                dtype=dtype,
            )
            .view([-1, 3])
            .to(env.DEVICE)
        )
        coord1 = torch.rand([natoms - coord0.shape[0], 3], dtype=dtype).to(env.DEVICE)
        coord1 = torch.matmul(coord1, cell)
        coord = torch.concat([coord0, coord1], dim=0)

        coord0 = torch.clone(coord)
        coord1 = torch.clone(coord)
        coord1[1][0] += epsilon
        coord2 = torch.clone(coord)
        coord2[2][1] += epsilon
        coord3 = torch.clone(coord)
        coord3[1][0] += epsilon
        coord3[2][1] += epsilon

        e0, f0, v0 = eval_model(
            self.model, coord0.unsqueeze(0), cell.unsqueeze(0), atype
        )
        ret0 = {
            "energy": e0.squeeze(0),
            "force": f0.squeeze(0),
            "virial": v0.squeeze(0),
        }
        e1, f1, v1 = eval_model(
            self.model, coord1.unsqueeze(0), cell.unsqueeze(0), atype
        )
        ret1 = {
            "energy": e1.squeeze(0),
            "force": f1.squeeze(0),
            "virial": v1.squeeze(0),
        }
        e2, f2, v2 = eval_model(
            self.model, coord2.unsqueeze(0), cell.unsqueeze(0), atype
        )
        ret2 = {
            "energy": e2.squeeze(0),
            "force": f2.squeeze(0),
            "virial": v2.squeeze(0),
        }
        e3, f3, v3 = eval_model(
            self.model, coord3.unsqueeze(0), cell.unsqueeze(0), atype
        )
        ret3 = {
            "energy": e3.squeeze(0),
            "force": f3.squeeze(0),
            "virial": v3.squeeze(0),
        }

        def compare(ret0, ret1):
            torch.testing.assert_close(
                ret0["energy"], ret1["energy"], rtol=rprec, atol=aprec
            )
            # plus 1. to avoid the divided-by-zero issue
            torch.testing.assert_close(
                1.0 + ret0["force"], 1.0 + ret1["force"], rtol=rprec, atol=aprec
            )
            if not hasattr(self, "test_virial") or self.test_virial:
                torch.testing.assert_close(
                    1.0 + ret0["virial"], 1.0 + ret1["virial"], rtol=rprec, atol=aprec
                )

        compare(ret0, ret1)
        compare(ret1, ret2)
        compare(ret0, ret3)


class TestEnergyModelSeA(unittest.TestCase, SmoothTest):
    def setUp(self):
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = None, None


# @unittest.skip("dpa-1 not smooth at the moment")
class TestEnergyModelDPA1(unittest.TestCase, SmoothTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)
        # less degree of smoothness,
        # error can be systematically removed by reducing epsilon
        self.epsilon = 1e-5
        self.aprec = 1e-5


class TestEnergyModelDPA2(unittest.TestCase, SmoothTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa2)
        model_params["descriptor"]["repinit_rcut"] = 8
        model_params["descriptor"]["repinit_rcut_smth"] = 3.5
        model_params_sample = copy.deepcopy(model_params)
        #######################################################
        # dirty hack here! the interface of dataload should be
        # redesigned to support specifying rcut and sel
        #######################################################
        model_params_sample["descriptor"]["rcut"] = model_params_sample["descriptor"][
            "repinit_rcut"
        ]
        model_params_sample["descriptor"]["sel"] = model_params_sample["descriptor"][
            "repinit_nsel"
        ]
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = 1e-5, 1e-4


class TestEnergyModelDPA2_1(unittest.TestCase, SmoothTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa2)
        model_params["fitting_net"]["type"] = "ener"
        model_params_sample = copy.deepcopy(model_params)
        model_params_sample["descriptor"]["rcut"] = model_params_sample["descriptor"][
            "repinit_rcut"
        ]
        model_params_sample["descriptor"]["sel"] = model_params_sample["descriptor"][
            "repinit_nsel"
        ]
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = None, None


class TestEnergyModelDPA2_2(unittest.TestCase, SmoothTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa2)
        model_params["fitting_net"]["type"] = "ener"
        model_params_sample = copy.deepcopy(model_params)
        model_params_sample["descriptor"]["rcut"] = model_params_sample["descriptor"][
            "repinit_rcut"
        ]
        model_params_sample["descriptor"]["sel"] = model_params_sample["descriptor"][
            "repinit_nsel"
        ]
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = None, None


@unittest.skip("hybrid not supported at the moment")
class TestEnergyModelHybrid(unittest.TestCase, SmoothTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = None, None


class TestEnergyModelZBL(unittest.TestCase, SmoothTest):
    def setUp(self):
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = None, None


# class TestEnergyFoo(unittest.TestCase):
#   def test(self):
#     model_params = model_dpau
#     self.model = EnergyModelDPAUni(model_params).to(env.DEVICE)

#     natoms = 5
#     cell = torch.rand([3, 3], dtype=dtype)
#     cell = (cell + cell.T) + 5. * torch.eye(3)
#     coord = torch.rand([natoms, 3], dtype=dtype)
#     coord = torch.matmul(coord, cell)
#     atype = torch.IntTensor([0, 0, 0, 1, 1])
#     idx_perm = [1, 0, 4, 3, 2]
#     ret0 = infer_model(self.model, coord, cell, atype, type_split=True)


if __name__ == "__main__":
    unittest.main()
