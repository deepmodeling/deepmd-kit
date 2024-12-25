# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
import paddle

from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
)

from ...seed import (
    GLOBAL_SEED,
)
from ..common import (
    eval_model,
)
from .test_permutation import (
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
)

dtype = paddle.float64


class NullTest:
    def test_nloc_1(
        self,
    ) -> None:
        natoms = 1
        generator = paddle.seed(GLOBAL_SEED)
        # paddle.seed(1000)
        cell = paddle.rand([3, 3], dtype=dtype).to(device=env.DEVICE)
        # large box to exclude images
        cell = (cell + cell.T) + 100.0 * paddle.eye(3).to(device=env.DEVICE)
        coord = paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        atype = paddle.to_tensor([0], dtype=paddle.int32).to(device=env.DEVICE)
        test_keys = ["energy", "force", "virial"]
        result = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
        ret0 = {key: result[key].squeeze(0) for key in test_keys}
        prec = 1e-10
        expect_e_shape = [1]
        expect_f = paddle.zeros([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        expect_v = paddle.zeros([9], dtype=dtype).to(device=env.DEVICE)
        self.assertEqual(list(ret0["energy"].shape), expect_e_shape)
        self.assertFalse(np.isnan(to_numpy_array(ret0["energy"])[0]))
        np.testing.assert_allclose(
            ret0["force"].numpy(), expect_f.numpy(), rtol=prec, atol=prec
        )
        if not hasattr(self, "test_virial") or self.test_virial:
            np.testing.assert_allclose(
                ret0["virial"].numpy(), expect_v.numpy(), rtol=prec, atol=prec
            )

    def test_nloc_2_far(
        self,
    ) -> None:
        natoms = 2
        generator = paddle.seed(GLOBAL_SEED)
        cell = paddle.rand([3, 3], dtype=dtype).to(device=env.DEVICE)
        # large box to exclude images
        cell = (cell + cell.T) + 3000.0 * paddle.eye(3).to(device=env.DEVICE)
        coord = paddle.rand([1, 3], dtype=dtype).to(device=env.DEVICE)
        # 2 far-away atoms
        coord = paddle.concat([coord, coord + 100.0], axis=0)
        atype = paddle.to_tensor([0, 2], dtype=paddle.int32).to(device=env.DEVICE)
        test_keys = ["energy", "force", "virial"]
        result = eval_model(self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype)
        ret0 = {key: result[key].squeeze(0) for key in test_keys}
        prec = 1e-10
        expect_e_shape = [1]
        expect_f = paddle.zeros([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        expect_v = paddle.zeros([9], dtype=dtype).to(device=env.DEVICE)
        self.assertEqual(list(ret0["energy"].shape), expect_e_shape)
        self.assertFalse(np.isnan(to_numpy_array(ret0["energy"])[0]))
        np.testing.assert_allclose(
            ret0["force"].numpy(), expect_f.numpy(), rtol=prec, atol=prec
        )
        if not hasattr(self, "test_virial") or self.test_virial:
            np.testing.assert_allclose(
                ret0["virial"].numpy(), expect_v.numpy(), rtol=prec, atol=prec
            )


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
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)
