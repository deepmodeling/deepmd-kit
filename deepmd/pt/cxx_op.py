# SPDX-License-Identifier: LGPL-3.0-or-later
import platform

import torch

from deepmd.env import (
    SHARED_LIB_DIR,
)


def load_library(module_name: str) -> bool:
    """Load OP library.

    Parameters
    ----------
    module_name : str
        Name of the module

    Returns
    -------
    bool
        Whether the library is loaded successfully
    """
    if platform.system() == "Windows":
        ext = ".dll"
        prefix = ""
    else:
        ext = ".so"
        prefix = "lib"

    module_file = (SHARED_LIB_DIR / (prefix + module_name)).with_suffix(ext).resolve()

    if module_file.is_file():
        torch.ops.load_library(module_file)
        return True
    raise RuntimeError("The PyTorch backend is not enabled.")


ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pt")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
]
