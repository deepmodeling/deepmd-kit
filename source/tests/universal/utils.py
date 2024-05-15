# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common test case."""

import unittest

from deepmd.backend.backend import (
    Backend,
)
from deepmd.dpmodel.common import (
    NativeOP,
)

INSTALLED_PT = Backend.get_backend("pytorch")().is_available()
if INSTALLED_PT:
    import torch


class CommonTestCase:
    """Common test case."""

    dp_module: NativeOP
    """DP module to test."""

    pt_module: "torch.nn.Module"
    """PT module to test."""

    @property
    def pt_script_module(self):
        return torch.jit.script(self.pt_module)

    @property
    def pt_deserialized_module(self):
        return self.pt_module.deserialize(self.pt_module.serialize())

    @property
    def dp_deserialized_module(self):
        return self.dp_module.deserialize(self.dp_module.serialize())

    @property
    def modules_to_test(self):
        modules = [
            self.dp_module,
            self.dp_deserialized_module,
        ]
        if INSTALLED_PT:
            modules.extend([self.pt_module, self.pt_deserialized_module])
        return modules

    @unittest.skipIf(not INSTALLED_PT, "PyTorch is not installed.")
    def test_pt_jit(self):
        self.pt_script_module


def forward_wrapper(module):
    if isinstance(module, NativeOP):
        return module
    elif INSTALLED_PT and isinstance(module, torch.nn.Module):
        from deepmd.pt.utils.utils import (
            to_numpy_array,
            to_torch_tensor,
        )

        def create_wrapper_method(method):
            def wrapper_method(self, *args, **kwargs):
                # convert to torch tensor
                args = [to_torch_tensor(arg) for arg in args]
                kwargs = {k: to_torch_tensor(v) for k, v in kwargs.items()}
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
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
