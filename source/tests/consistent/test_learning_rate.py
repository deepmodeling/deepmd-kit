# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.learning_rate import (
    BaseLR,
)

from .common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    parameterized,
)

if INSTALLED_PT:
    import array_api_compat.torch as torch_xp
    import torch
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict as xp


@parameterized(
    (
        {
            "type": "exp",
            "start_lr": 1e-3,
            "stop_lr": 1e-8,
            "decay_steps": 1000,
            "stop_steps": 1000000,
        },
        {
            "type": "cosine",
            "start_lr": 1e-3,
            "stop_lr": 1e-8,
            "decay_steps": 1000,
            "stop_steps": 1000000,
        },
    ),
)
class TestActivationFunctionConsistent(unittest.TestCase):
    def setUp(self) -> None:
        (lr_param,) = self.param
        self.lr = BaseLR(**lr_param)
        self.step = 500000
        self.ref = self.lr.value(self.step, xp=np)

    def compare_test_with_ref(self, xp: Any) -> None:
        test = self.lr.value(self.step, xp=xp)
        np.testing.assert_allclose(self.ref, to_numpy_array(test), atol=1e-10)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        with torch.device("cpu"):
            self.compare_test_with_ref(torch_xp)

    @unittest.skipUnless(
        INSTALLED_ARRAY_API_STRICT, "array_api_strict is not installed"
    )
    @unittest.skipUnless(
        sys.version_info >= (3, 9), "array_api_strict doesn't support Python<=3.8"
    )
    def test_array_api_strict(self) -> None:
        self.compare_test_with_ref(xp)

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_consistent_with_ref(self) -> None:
        self.compare_test_with_ref(jnp)
