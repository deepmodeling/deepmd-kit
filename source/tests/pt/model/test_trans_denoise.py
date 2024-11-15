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
from .test_permutation_denoise import (
    model_dpa1,
    model_dpa2,
    model_hybrid,
)

dtype = torch.float64


class TransDenoiseTest:
    def test(
        self,
    ) -> None:
        natoms = 5
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, generator=generator).to(env.DEVICE)
        cell = (cell + cell.T) + 5.0 * torch.eye(3).to(env.DEVICE)
        coord = torch.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
        shift = (torch.rand([3], dtype=dtype, generator=generator) - 0.5).to(
            env.DEVICE
        ) * 2.0
        coord_s = torch.matmul(
            torch.remainder(torch.matmul(coord + shift, torch.linalg.inv(cell)), 1.0),
            cell,
        )
        updated_c0, logits0 = eval_model(
            self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        updated_c0 = updated_c0 - coord.unsqueeze(0)
        ret0 = {"updated_coord": updated_c0.squeeze(0), "logits": logits0.squeeze(0)}
        updated_c1, logits1 = eval_model(
            self.model, coord_s.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        updated_c1 = updated_c1 - coord_s.unsqueeze(0)
        ret1 = {"updated_coord": updated_c1.squeeze(0), "logits": logits1.squeeze(0)}
        prec = 1e-10
        torch.testing.assert_close(
            ret0["updated_coord"], ret1["updated_coord"], rtol=prec, atol=prec
        )
        torch.testing.assert_close(ret0["logits"], ret1["logits"], rtol=prec, atol=prec)


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA1(unittest.TestCase, TransDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA2(unittest.TestCase, TransDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("hybrid not supported at the moment")
class TestDenoiseModelHybrid(unittest.TestCase, TransDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
