# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
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

dtype = torch.float64


class TransTest:
    def test(
        self,
    ) -> None:
        natoms = 5
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE, generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        coord = torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        coord = torch.matmul(coord, cell)
        spin = torch.rand(
            [natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE)
        shift = (
            torch.rand([3], dtype=dtype, device=env.DEVICE, generator=generator) - 0.5
        ) * 2.0
        coord_s = torch.matmul(
            torch.remainder(torch.matmul(coord + shift, torch.linalg.inv(cell)), 1.0),
            cell,
        )
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
            coord_s.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            spins=spin.unsqueeze(0),
        )
        ret1 = {key: result_1[key].squeeze(0) for key in test_keys}
        prec = 1e-7
        for key in test_keys:
            if key in ["energy", "force", "force_mag"]:
                torch.testing.assert_close(ret0[key], ret1[key], rtol=prec, atol=prec)
            elif key == "virial":
                if not hasattr(self, "test_virial") or self.test_virial:
                    torch.testing.assert_close(
                        ret0[key], ret1[key], rtol=prec, atol=prec
                    )
            else:
                raise RuntimeError(f"Unexpected test key {key}")


class TestEnergyModelSeA(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestDOSModelSeA(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dos)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA2(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestForceModelDPA2(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelHybrid(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestForceModelHybrid(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hybrid)
        model_params["fitting_net"]["type"] = "direct_force_ener"
        self.type_split = True
        self.test_virial = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelZBL(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinSeA(unittest.TestCase, TransTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_spin)
        self.type_split = False
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
