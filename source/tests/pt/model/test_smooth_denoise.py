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
    model_dpa2,
)

dtype = torch.float64


class SmoothDenoiseTest:
    def test(
        self,
    ) -> None:
        # displacement of atoms
        epsilon = 1e-5 if self.epsilon is None else self.epsilon
        # required prec. relative prec is not checked.
        rprec = 0
        aprec = 1e-5 if self.aprec is None else self.aprec

        natoms = 10
        cell = 8.6 * torch.eye(3, dtype=dtype).to(env.DEVICE)
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        atype = torch.randint(0, 3, [natoms], generator=generator, device=env.DEVICE)
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
        coord1 = torch.rand(
            [natoms - coord0.shape[0], 3], dtype=dtype, generator=generator
        ).to(env.DEVICE)
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

        update_c0, logits0 = eval_model(
            self.model, coord0.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        ret0 = {"updated_coord": update_c0.squeeze(0), "logits": logits0.squeeze(0)}
        update_c1, logits1 = eval_model(
            self.model, coord1.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        ret1 = {"updated_coord": update_c1.squeeze(0), "logits": logits1.squeeze(0)}
        update_c2, logits2 = eval_model(
            self.model, coord2.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        ret2 = {"updated_coord": update_c2.squeeze(0), "logits": logits2.squeeze(0)}
        update_c3, logits3 = eval_model(
            self.model, coord3.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        ret3 = {"updated_coord": update_c3.squeeze(0), "logits": logits3.squeeze(0)}

        def compare(ret0, ret1) -> None:
            torch.testing.assert_close(
                ret0["updated_coord"], ret1["updated_coord"], rtol=rprec, atol=aprec
            )
            torch.testing.assert_close(
                ret0["logits"], ret1["logits"], rtol=rprec, atol=aprec
            )

        compare(ret0, ret1)
        compare(ret1, ret2)
        compare(ret0, ret3)


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA2(unittest.TestCase, SmoothDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        model_params["descriptor"]["sel"] = 8
        model_params["descriptor"]["rcut_smth"] = 3.5
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = None, None
        self.epsilon = 1e-7
        self.aprec = 1e-5


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA2_1(unittest.TestCase, SmoothDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        # model_params["descriptor"]["combine_grrg"] = True
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)
        self.epsilon, self.aprec = None, None
        self.epsilon = 1e-7
        self.aprec = 1e-5


# @unittest.skip("hybrid not supported at the moment")
# class TestDenoiseModelHybrid(unittest.TestCase, TestSmoothDenoise):
#     def setUp(self):
#         model_params = copy.deepcopy(model_hybrid_denoise)
#         self.type_split = True
#         self.model = get_model(model_params).to(env.DEVICE)
#         self.epsilon, self.aprec = None, None
#         self.epsilon = 1e-7
#         self.aprec = 1e-5


if __name__ == "__main__":
    unittest.main()
