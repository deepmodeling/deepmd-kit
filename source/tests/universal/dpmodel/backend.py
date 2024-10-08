# SPDX-License-Identifier: LGPL-3.0-or-later
from functools import (
    cache,
)

import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)

from ..common.backend import (
    BackendTestCase,
)


class DPTestCase(BackendTestCase):
    """Common test case."""

    module: NativeOP
    """DP module to test."""

    def forward_wrapper(self, x):
        return x

    def forward_wrapper_cpu_ref(self, x):
        return x

    @classmethod
    def convert_to_numpy(cls, xx: np.ndarray) -> np.ndarray:
        return xx

    @classmethod
    def convert_from_numpy(cls, xx: np.ndarray) -> np.ndarray:
        return xx

    @classmethod
    @cache
    def deserialized_module(cls):
        return cls.module.deserialize(cls.module.serialize())

    @property
    def modules_to_test(self):
        modules = [
            self.module,
            self.deserialized_module,
        ]
        return modules
