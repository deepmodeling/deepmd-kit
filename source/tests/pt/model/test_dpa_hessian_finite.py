# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_permutation import (
    model_dpa2,
    model_dpa3,
)

dtype = torch.float64


class TestDPAHessianFinite(unittest.TestCase):
    def _build_inputs(self):
        natoms = 5
        cell = 4.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        coord = 3.0 * torch.rand(
            [1, natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        atype = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int64, device=env.DEVICE)
        return coord.view(1, natoms * 3), atype, cell.view(1, 9)

    def _assert_hessian_finite(self, model_params):
        model = get_model(copy.deepcopy(model_params)).to(env.DEVICE)
        model.enable_hessian()
        model.requires_hessian("energy")
        coord, atype, cell = self._build_inputs()
        ret = model.forward_common(coord, atype, box=cell)
        hessian = to_numpy_array(ret["energy_derv_r_derv_r"])
        self.assertTrue(np.isfinite(hessian).all())

    def test_dpa2_direct_dist_hessian_is_finite(self):
        model_params = copy.deepcopy(model_dpa2)
        model_params["descriptor"]["repformer"]["direct_dist"] = True
        self._assert_hessian_finite(model_params)

    def test_dpa3_hessian_is_finite(self):
        model_params = copy.deepcopy(model_dpa3)
        model_params["descriptor"]["precision"] = "float64"
        model_params["fitting_net"]["precision"] = "float64"
        self._assert_hessian_finite(model_params)


if __name__ == "__main__":
    unittest.main()
