# SPDX-License-Identifier: LGPL-3.0-or-later
from .deep_eval import (
    DeepEval,
)
from .deep_pot import (
    DeepPot,
)
from .model_devi import (
    calc_model_devi,
)

__all__ = [
    "DeepPot",
    "calc_model_devi",
    "DeepEval",
    "DeepPotential",
]


def DeepPotential(*args, **kwargs) -> "DeepEval":
    """Factory function that forwards to DeepEval (for compatbility).

    Parameters
    ----------
    *args
        positional arguments
    **kwargs
        keyword arguments

    Returns
    -------
    DeepEval
        potentials
    """
    return DeepEval(*args, **kwargs)
