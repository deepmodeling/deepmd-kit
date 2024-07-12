# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

if sys.version_info >= (3, 9):
    import array_api_strict as xp
else:
    raise unittest.SkipTest("array_api_strict doesn't support Python<=3.8")

from deepmd.dpmodel.utils.env_mat import (
    compute_smooth_weight,
)

from .utils import (
    ArrayAPITest,
)


class TestEnvMat(unittest.TestCase, ArrayAPITest):
    def test_compute_smooth_weight(self):
        self.set_array_api_version(compute_smooth_weight)
        d = xp.arange(10, dtype=xp.float64)
        w = compute_smooth_weight(
            d,
            4.0,
            6.0,
        )
        self.assert_namespace_equal(w, d)
        self.assert_device_equal(w, d)
        self.assert_dtype_equal(w, d)
