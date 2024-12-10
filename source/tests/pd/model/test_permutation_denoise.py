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
    get_generator,
)

from ...seed import (
    GLOBAL_SEED,
)
from ..common import (
    eval_model,
)
from .test_permutation import (  # model_dpau,
    model_dpa1,
    model_dpa2,
    model_hybrid,
)

dtype = paddle.float64

model_dpa1 = copy.deepcopy(model_dpa1)
model_dpa2 = copy.deepcopy(model_dpa2)
model_hybrid = copy.deepcopy(model_hybrid)
model_dpa1["type_map"] = ["O", "H", "B", "MASKED_TOKEN"]
model_dpa1.pop("fitting_net")
model_dpa2["type_map"] = ["O", "H", "B", "MASKED_TOKEN"]
model_dpa2.pop("fitting_net")
model_hybrid["type_map"] = ["O", "H", "B", "MASKED_TOKEN"]
model_hybrid.pop("fitting_net")


class PermutationDenoiseTest:
    def test(
        self,
    ) -> None:
        generator = get_generator(GLOBAL_SEED)
        natoms = 5
        cell = paddle.rand([3, 3], dtype=dtype).to(env.DEVICE)
        cell = (cell + cell.T) + 5.0 * paddle.eye(3).to(env.DEVICE)
        coord = paddle.rand([natoms, 3], dtype=dtype).to(env.DEVICE)
        coord = paddle.matmul(coord, cell)
        atype = paddle.to_tensor([0, 0, 0, 1, 1]).to(env.DEVICE)
        idx_perm = [1, 0, 4, 3, 2]
        updated_c0, logits0 = eval_model(
            self.model, coord.unsqueeze(0), cell.unsqueeze(0), atype, denoise=True
        )
        ret0 = {"updated_coord": updated_c0.squeeze(0), "logits": logits0.squeeze(0)}
        updated_c1, logits1 = eval_model(
            self.model,
            coord[idx_perm].unsqueeze(0),
            cell.unsqueeze(0),
            atype[idx_perm],
            denoise=True,
        )
        ret1 = {"updated_coord": updated_c1.squeeze(0), "logits": logits1.squeeze(0)}
        prec = 1e-10
        np.testing.assert_allclose(
            ret0["updated_coord"][idx_perm].numpy(),
            ret1["updated_coord"].numpy(),
            rtol=prec,
            atol=prec,
        )
        np.testing.assert_allclose(
            ret0["logits"][idx_perm].numpy(),
            ret1["logits"].numpy(),
            rtol=prec,
            atol=prec,
        )


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA1(unittest.TestCase, PermutationDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


@unittest.skip("support of the denoise is temporally disabled")
class TestDenoiseModelDPA2(unittest.TestCase, PermutationDenoiseTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(
            model_params,
        ).to(env.DEVICE)


# @unittest.skip("hybrid not supported at the moment")
# class TestDenoiseModelHybrid(unittest.TestCase, TestPermutationDenoise):
#     def setUp(self):
#         model_params = copy.deepcopy(model_hybrid_denoise)
#         self.type_split = True
#         self.model = get_model(model_params).to(env.DEVICE)


if __name__ == "__main__":
    unittest.main()
