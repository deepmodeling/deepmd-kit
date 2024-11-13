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
)

dtype = torch.float64


class RotDenoiseTest:
    def test(
        self,
    ) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        prec = 1e-10
        natoms = 5
        cell = 10.0 * torch.eye(3, dtype=dtype).to(env.DEVICE)
        coord = 2 * torch.rand(
            [natoms, 3], dtype=dtype, generator=generator, device=env.DEVICE
        )
        shift = torch.tensor([4, 4, 4], dtype=dtype).to(env.DEVICE)
        atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
        from scipy.stats import (
            special_ortho_group,
        )

        rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype).to(env.DEVICE)

        # rotate only coord and shift to the center of cell
        coord_rot = torch.matmul(coord, rmat)
        update_c0, logits0 = eval_model(
            self.model,
            (coord + shift).unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            denoise=True,
        )
        update_c0 = update_c0 - (coord + shift).unsqueeze(0)
        ret0 = {"updated_coord": update_c0.squeeze(0), "logits": logits0.squeeze(0)}
        update_c1, logits1 = eval_model(
            self.model,
            (coord_rot + shift).unsqueeze(0),
            cell.unsqueeze(0),
            atype,
            denoise=True,
        )
        update_c1 = update_c1 - (coord_rot + shift).unsqueeze(0)
        ret1 = {"updated_coord": update_c1.squeeze(0), "logits": logits1.squeeze(0)}
        torch.testing.assert_close(
            torch.matmul(ret0["updated_coord"], rmat),
            ret1["updated_coord"],
            rtol=prec,
            atol=prec,
        )
        torch.testing.assert_close(ret0["logits"], ret1["logits"], rtol=prec, atol=prec)

        # rotate coord and cell
        torch.manual_seed(0)
        cell = torch.rand([3, 3], dtype=dtype, generator=generator).to(env.DEVICE)
        cell = (cell + cell.T) + 5.0 * torch.eye(3).to(env.DEVICE)
        coord = torch.rand([natoms, 3], dtype=dtype, generator=generator).to(env.DEVICE)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1]).to(env.DEVICE)
        coord_rot = torch.matmul(coord, rmat)
        cell_rot = torch.matmul(cell, rmat)
        update_c0, logits0 = eval_model(
            self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        ret0 = {"updated_coord": update_c0.squeeze(0), "logits": logits0.squeeze(0)}
        update_c1, logits1 = eval_model(
            self.model,
            coord_rot.unsqueeze(0),
            cell_rot.unsqueeze(0),
            atype,
            denoise=True,
        )
        ret1 = {"updated_coord": update_c1.squeeze(0), "logits": logits1.squeeze(0)}
        torch.testing.assert_close(ret0["logits"], ret1["logits"], rtol=prec, atol=prec)
        torch.testing.assert_close(
            torch.matmul(ret0["updated_coord"], rmat),
            ret1["updated_coord"],
            rtol=prec,
            atol=prec,
        )


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA1(unittest.TestCase, RotDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA2(unittest.TestCase, RotDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


# @unittest.skip("hybrid not supported at the moment")
# class TestEnergyModelHybrid(unittest.TestCase, TestRotDenoise):
#     def setUp(self):
#         model_params = copy.deepcopy(model_hybrid_denoise)
#         self.type_split = True
#         self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
