# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
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
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .test_permutation import (
    model_dpa1,
    model_dpa2,
    model_hybrid,
    model_se_e2_a,
    model_zbl,
)

dtype = torch.float64


class NullTest:
    def test_nloc_1(
        self,
    ):
        natoms = 1
        # torch.manual_seed(1000)
        cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE)
        # large box to exclude images
        cell = (cell + cell.T) + 100.0 * torch.eye(3, device=env.DEVICE)
        coord = torch.rand([natoms, 3], dtype=dtype, device=env.DEVICE)
        atype = torch.tensor([0], dtype=torch.int32, device=env.DEVICE)
        test_keys = ["energy", "force", "virial"]
        result = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
        ret0 = {key: result[key].squeeze(0) for key in test_keys}
        prec = 1e-10
        expect_e_shape = [1]
        expect_f = torch.zeros([natoms, 3], dtype=dtype, device=env.DEVICE)
        expect_v = torch.zeros([9], dtype=dtype, device=env.DEVICE)
        self.assertEqual(list(ret0["energy"].shape), expect_e_shape)
        self.assertFalse(np.isnan(to_numpy_array(ret0["energy"])[0]))
        torch.testing.assert_close(ret0["force"], expect_f, rtol=prec, atol=prec)
        if not hasattr(self, "test_virial") or self.test_virial:
            torch.testing.assert_close(ret0["virial"], expect_v, rtol=prec, atol=prec)

    def test_nloc_2_far(
        self,
    ):
        natoms = 2
        cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE)
        # large box to exclude images
        cell = (cell + cell.T) + 3000.0 * torch.eye(3, device=env.DEVICE)
        coord = torch.rand([1, 3], dtype=dtype, device=env.DEVICE)
        # 2 far-away atoms
        coord = torch.cat([coord, coord + 100.0], dim=0)
        atype = torch.tensor([0, 2], dtype=torch.int32, device=env.DEVICE)
        test_keys = ["energy", "force", "virial"]
        result = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
        ret0 = {key: result[key].squeeze(0) for key in test_keys}
        prec = 1e-10
        expect_e_shape = [1]
        expect_f = torch.zeros([natoms, 3], dtype=dtype, device=env.DEVICE)
        expect_v = torch.zeros([9], dtype=dtype, device=env.DEVICE)
        self.assertEqual(list(ret0["energy"].shape), expect_e_shape)
        self.assertFalse(np.isnan(to_numpy_array(ret0["energy"])[0]))
        torch.testing.assert_close(ret0["force"], expect_f, rtol=prec, atol=prec)
        if not hasattr(self, "test_virial") or self.test_virial:
            torch.testing.assert_close(ret0["virial"], expect_v, rtol=prec, atol=prec)


class TestEnergyModelSeA(unittest.TestCase, NullTest):
    def setUp(self):
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, NullTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA2(unittest.TestCase, NullTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestForceModelDPA2(unittest.TestCase, NullTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa2)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelHybrid(unittest.TestCase, NullTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestForceModelHybrid(unittest.TestCase, NullTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelZBL(unittest.TestCase, NullTest):
    def setUp(self):
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_zbl_model(model_params).to(env.DEVICE)
