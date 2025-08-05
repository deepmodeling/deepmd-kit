# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
from types import (
    ModuleType,
)


def load_library(module_name: str) -> tuple[bool, ModuleType]:
    """Load OP library and return the module if success.

    Parameters
    ----------
    module_name : str
        Name of the module

    Returns
    -------
    bool
        Whether the library is loaded successfully
    ModuleType
        loaded custom operator module
    """
    if importlib.util.find_spec(module_name) is not None:
        module = importlib.import_module(module_name)
        return True, module

    return False, None


ENABLE_CUSTOMIZED_OP, paddle_ops_deepmd = load_library("deepmd_op_pd")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
    "paddle_ops_deepmd",
]
