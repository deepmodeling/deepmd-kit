# SPDX-License-Identifier: LGPL-3.0-or-later


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
    # NOTE: Paddle do not support loading library from .so file yet.
    return False


ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pd")

__all__ = [
    "ENABLE_CUSTOMIZED_OP",
]
