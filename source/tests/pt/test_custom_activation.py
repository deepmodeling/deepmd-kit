# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    CustomSilu,
    CustomSiluScript,
    to_numpy_array,
)

from ..consistent.common import (
    parameterized,
)


@parameterized(
    (3.0, 10.0),
)
class TestCustomSilu(unittest.TestCase):
    def setUp(self) -> None:
        (self.threshold,) = self.param
        self.custom_silu_naive = CustomSilu(threshold=self.threshold)
        self.custom_silu_script = CustomSiluScript(threshold=self.threshold)

    def test_naive_consistent_with_script(self) -> None:
        def get_compare(cust_silu_tmp):
            x_tmp = torch.arange(
                -60.0,
                60.0,
                0.1,
                device=env.DEVICE,
                requires_grad=True,
                dtype=torch.float64,
            )
            y_tmp = cust_silu_tmp(x_tmp)
            dy_tmp = torch.autograd.grad(y_tmp, x_tmp, y_tmp * 10.0, create_graph=True)[
                0
            ]
            dy2_tmp = torch.autograd.grad(dy_tmp, x_tmp, dy_tmp * 10.0)[0]
            return (
                to_numpy_array(y_tmp),
                to_numpy_array(dy_tmp),
                to_numpy_array(dy2_tmp),
            )

        rtol = 1e-8
        atol = 1e-8
        naive_y, naive_dy, naive_dy2 = get_compare(self.custom_silu_naive)
        script_y, script_dy, script_dy2 = get_compare(self.custom_silu_script)
        np.testing.assert_allclose(naive_y, script_y, rtol=rtol, atol=atol)
        np.testing.assert_allclose(naive_dy, script_dy, rtol=rtol, atol=atol)
        np.testing.assert_allclose(naive_dy2, script_dy2, rtol=rtol, atol=atol)
