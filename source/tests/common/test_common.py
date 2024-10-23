# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import array_api_compat
import ml_dtypes
import numpy as np

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)


class TestGetXPPrecision(unittest.TestCase):
    def test(self):
        aa = np.zeros(3)
        xp = array_api_compat.array_namespace(aa)
        self.assertTrue(get_xp_precision(xp, "float16"), xp.float16)
        self.assertTrue(get_xp_precision(xp, "float32"), xp.float32)
        self.assertTrue(get_xp_precision(xp, "float64"), xp.float64)
        self.assertTrue(get_xp_precision(xp, "single"), xp.float32)
        self.assertTrue(get_xp_precision(xp, "double"), xp.float64)
        self.assertTrue(get_xp_precision(xp, "global"), GLOBAL_NP_FLOAT_PRECISION)
        self.assertTrue(get_xp_precision(xp, "default"), GLOBAL_NP_FLOAT_PRECISION)
        self.assertTrue(get_xp_precision(xp, "bfloat16"), ml_dtypes.bfloat16)
