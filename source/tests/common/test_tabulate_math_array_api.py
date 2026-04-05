# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.utils import tabulate_math as tm

from ..consistent.common import (
    INSTALLED_ARRAY_API_STRICT,
)

if INSTALLED_ARRAY_API_STRICT:
    from .. import array_api_strict as _array_api_strict  # noqa: F401
    from ..array_api_strict.common import (
        to_array_api_strict_array,
    )


class TestTabulateMathArrayAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.xbar_np = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
        self.y_np = np.tanh(self.xbar_np)
        self.w_np = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)

    @unittest.skipUnless(
        INSTALLED_ARRAY_API_STRICT, "array_api_strict is not installed"
    )
    @unittest.skipUnless(
        sys.version_info >= (3, 9), "array_api_strict doesn't support Python<=3.8"
    )
    def test_chain_rule_helpers_array_api_strict_consistent_with_numpy(self) -> None:
        xbar = to_array_api_strict_array(self.xbar_np)
        y = to_array_api_strict_array(self.y_np)
        w = to_array_api_strict_array(self.w_np)

        dy_s = tm.unaggregated_dy_dx_s(y, w, xbar, 1)
        dy2_s = tm.unaggregated_dy2_dx_s(y, dy_s, w, xbar, 1)
        dy = tm.unaggregated_dy_dx(y, w, dy_s, xbar, 1)
        dy2 = tm.unaggregated_dy2_dx(y, w, dy_s, dy2_s, xbar, 1)

        dy_s_ref = tm.unaggregated_dy_dx_s(self.y_np, self.w_np, self.xbar_np, 1)
        dy2_s_ref = tm.unaggregated_dy2_dx_s(
            self.y_np,
            dy_s_ref,
            self.w_np,
            self.xbar_np,
            1,
        )
        dy_ref = tm.unaggregated_dy_dx(
            self.y_np,
            self.w_np,
            dy_s_ref,
            self.xbar_np,
            1,
        )
        dy2_ref = tm.unaggregated_dy2_dx(
            self.y_np,
            self.w_np,
            dy_s_ref,
            dy2_s_ref,
            self.xbar_np,
            1,
        )

        np.testing.assert_allclose(to_numpy_array(dy_s), dy_s_ref, atol=1e-10)
        np.testing.assert_allclose(to_numpy_array(dy2_s), dy2_s_ref, atol=1e-10)
        np.testing.assert_allclose(to_numpy_array(dy), dy_ref, atol=1e-10)
        np.testing.assert_allclose(to_numpy_array(dy2), dy2_ref, atol=1e-10)

    @unittest.skipUnless(
        INSTALLED_ARRAY_API_STRICT, "array_api_strict is not installed"
    )
    @unittest.skipUnless(
        sys.version_info >= (3, 9), "array_api_strict doesn't support Python<=3.8"
    )
    def test_stable_sigmoid_and_silu_grad_array_api_strict_consistent_with_numpy(
        self,
    ) -> None:
        xbar_np = np.array([[-1000.0, -1.0, 0.0, 1.0, 1000.0]], dtype=np.float64)
        xbar = to_array_api_strict_array(xbar_np)

        stable = tm._stable_sigmoid(xbar)
        silu_grad = tm.grad(xbar, stable, 7)

        stable_ref = tm._stable_sigmoid(xbar_np)
        silu_grad_ref = tm.grad(xbar_np, stable_ref, 7)

        np.testing.assert_allclose(to_numpy_array(stable), stable_ref, atol=1e-10)
        np.testing.assert_allclose(
            to_numpy_array(silu_grad),
            silu_grad_ref,
            atol=1e-10,
        )
