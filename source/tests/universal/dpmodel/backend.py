# SPDX-License-Identifier: LGPL-3.0-or-later
from functools import (
    lru_cache,
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
    @lru_cache(maxsize=1)
    def _get_deserialized_module(cls):
        return cls.module.deserialize(cls.module.serialize())

    @property
    def deserialized_module(self):
        if hasattr(self.__class__, "module"):
            return self._get_deserialized_module()
        return self.module.deserialize(self.module.serialize())

    @property
    def modules_to_test(self):
        modules = [
            self.module,
            self.deserialized_module,
        ]
        return modules

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "module"):
            del cls.module
        cls._get_deserialized_module.cache_clear()
