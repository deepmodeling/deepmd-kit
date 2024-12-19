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

from ...seed import (
    GLOBAL_SEED,
)
from ..common import (
    eval_model,
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

dtype = paddle.float64


class RotTest:
    def test(
        self,
    ):
        generator = paddle.seed(GLOBAL_SEED)
        prec = 1e-9
        natoms = 5
        cell = 10.0 * paddle.eye(3, dtype=dtype).to(device=env.DEVICE)
        coord = 2 * paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        spin = 2 * paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        shift = paddle.to_tensor([4, 4, 4], dtype=dtype).to(device=env.DEVICE)
        atype = paddle.to_tensor([0, 0, 0, 1, 1], dtype=paddle.int32).to(
            device=env.DEVICE
        )
        from scipy.stats import (
            special_ortho_group,
        )

        test_spin = getattr(self, "test_spin", False)
        if not test_spin:
            test_keys = ["energy", "force", "virial"]
        else:
            test_keys = ["energy", "force", "force_mag"]
        rmat = paddle.to_tensor(special_ortho_group.rvs(3), dtype=dtype).to(
            device=env.DEVICE
        )

        # rotate only coord and shift to the center of cell
        coord_rot = paddle.matmul(coord, rmat)
        spin_rot = paddle.matmul(spin, rmat)
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
                np.testing.assert_allclose(
                    ret0[key].numpy(), ret1[key].numpy(), rtol=prec, atol=prec
                )
            elif key in ["force", "force_mag"]:
                np.testing.assert_allclose(
                    paddle.matmul(ret0[key], rmat).numpy(),
                    ret1[key].numpy(),
                    rtol=prec,
                    atol=prec,
                )
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    np.testing.assert_allclose(
                        paddle.matmul(
                            rmat.T, paddle.matmul(ret0[key].reshape([3, 3]), rmat)
                        ).numpy(),
                        ret1[key].reshape([3, 3]).numpy(),
                        rtol=prec,
                        atol=prec,
                    )
            else:
                raise RuntimeError(f"Unexpected test key {key}")
        # rotate coord and cell
        paddle.seed(0)
        cell = paddle.rand([3, 3], dtype=dtype).to(device=env.DEVICE)
        cell = (cell + cell.T) + 5.0 * paddle.eye(3).to(device=env.DEVICE)
        coord = paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        coord = paddle.matmul(coord, cell)
        spin = paddle.rand([natoms, 3], dtype=dtype).to(device=env.DEVICE)
        atype = paddle.to_tensor([0, 0, 0, 1, 1], dtype=paddle.int32).to(
            device=env.DEVICE
        )
        coord_rot = paddle.matmul(coord, rmat)
        spin_rot = paddle.matmul(spin, rmat)
        cell_rot = paddle.matmul(cell, rmat)
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
                np.testing.assert_allclose(
                    ret0[key].numpy(), ret1[key].numpy(), rtol=prec, atol=prec
                )
            elif key in ["force", "force_mag"]:
                np.testing.assert_allclose(
                    paddle.matmul(ret0[key], rmat).numpy(),
                    ret1[key].numpy(),
                    rtol=prec,
                    atol=prec,
                )
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    np.testing.assert_allclose(
                        paddle.matmul(
                            rmat.T, paddle.matmul(ret0[key].reshape([3, 3]), rmat)
                        ).numpy(),
                        ret1[key].reshape([3, 3]).numpy(),
                        rtol=prec,
                        atol=prec,
                    )


class TestEnergyModelSeA(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
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
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestForceModelDPA2(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_dpa2)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelHybrid(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestForceModelHybrid(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_hybrid)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelZBL(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("Skip for not implemented yet")
class TestEnergyModelSpinSeA(unittest.TestCase, RotTest):
    def setUp(self):
        model_params = copy.deepcopy(model_spin)
        self.type_split = False
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
