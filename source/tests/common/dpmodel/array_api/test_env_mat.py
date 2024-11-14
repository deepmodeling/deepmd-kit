# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import array_api_strict as xp

from deepmd.dpmodel.utils.env_mat import (
    compute_smooth_weight,
)

from .utils import (
    ArrayAPITest,
)


class TestEnvMat(unittest.TestCase, ArrayAPITest):
    def test_compute_smooth_weight(self) -> None:
        d = xp.arange(10, dtype=xp.float64)
        w = compute_smooth_weight(
            d,
            4.0,
            6.0,
        )
        self.assert_namespace_equal(w, d)
        self.assert_device_equal(w, d)
        self.assert_dtype_equal(w, d)
