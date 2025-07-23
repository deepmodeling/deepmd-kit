# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from types import (
    ModuleType,
)

from paddle.utils.cpp_extension import (
    load,
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
    current_file = Path(__file__).resolve()
    base_dir = current_file.parents[2]

    module_file = base_dir / "source" / "op" / "pd" / "comm.cc"
    paddle_ops_deepmd = load(
        name="deepmd_op_pd",
        sources=[str(module_file)],
    )
    return True, paddle_ops_deepmd


ENABLE_CUSTOMIZED_OP, paddle_ops_deepmd = load_library("deepmd_op_pd")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
    "paddle_ops_deepmd",
]
