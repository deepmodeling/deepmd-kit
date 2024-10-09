# SPDX-License-Identifier: LGPL-3.0-or-later
from functools import (
    lru_cache,
)

import numpy as np
import torch

from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from ..common.backend import (
    BackendTestCase,
)


class PTTestCase(BackendTestCase):
    """Common test case."""

    module: "torch.nn.Module"
    """PT module to test."""

    @classmethod
    @lru_cache(maxsize=1)
    def _get_script_module(cls):
        with torch.jit.optimized_execution(False):
            return torch.jit.script(cls.module)

    @property
    def script_module(self):
        if hasattr(self.__class__, "module"):
            return self._get_script_module()
        with torch.jit.optimized_execution(False):
            return torch.jit.script(self.module)

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
    def tearDownClass(cls):
        super().tearDownClass()
        if hasattr(cls, "module"):
            del cls.module
        cls._get_deserialized_module.cache_clear()
        cls._get_script_module.cache_clear()

    def test_jit(self):
        if getattr(self, "skip_test_jit", False):
            self.skipTest("Skip test jit.")
        self.script_module

    @classmethod
    def convert_to_numpy(cls, xx: torch.Tensor) -> np.ndarray:
        return to_numpy_array(xx)

    @classmethod
    def convert_from_numpy(cls, xx: np.ndarray) -> torch.Tensor:
        return to_torch_tensor(xx)

    def forward_wrapper_cpu_ref(self, module):
        module.to("cpu")
        return self.forward_wrapper(module, on_cpu=True)

    def forward_wrapper(self, module, on_cpu=False):
        def create_wrapper_method(method):
            def wrapper_method(self, *args, **kwargs):
                # convert to torch tensor
                args = [to_torch_tensor(arg) for arg in args]
                kwargs = {k: to_torch_tensor(v) for k, v in kwargs.items()}
                if on_cpu:
                    args = [
                        arg.detach().cpu() if arg is not None else None for arg in args
                    ]
                    kwargs = {
                        k: v.detach().cpu() if v is not None else None
                        for k, v in kwargs.items()
                    }
                # forward
                output = method(*args, **kwargs)
                # convert to numpy array
                if isinstance(output, tuple):
                    output = tuple(to_numpy_array(o) for o in output)
                elif isinstance(output, dict):
                    output = {k: to_numpy_array(v) for k, v in output.items()}
                else:
                    output = to_numpy_array(output)
                return output

            return wrapper_method

        class wrapper_module:
            __call__ = create_wrapper_method(module.__call__)
            if hasattr(module, "forward_lower"):
                forward_lower = create_wrapper_method(module.forward_lower)

        return wrapper_module()
