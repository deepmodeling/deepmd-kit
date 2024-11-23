# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from ..seed import (
    GLOBAL_SEED,
)


class TestCvt(unittest.TestCase):
    def test_to_numpy(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        foo = rng.normal([3, 4])
        for ptp, npp in zip(
            [torch.float16, torch.float32, torch.float64],
            [np.float16, np.float32, np.float64],
        ):
            foo = foo.astype(npp)
            bar = to_torch_tensor(foo)
            self.assertEqual(bar.dtype, ptp)
            onk = to_numpy_array(bar)
            self.assertEqual(onk.dtype, npp)
        with self.assertRaises(ValueError) as ee:
            foo = foo.astype(np.int8)
            bar = to_torch_tensor(foo)
        with self.assertRaises(ValueError) as ee:
            bar = to_torch_tensor(foo)
            bar = to_numpy_array(bar.int())
