# SPDX-License-Identifier: LGPL-3.0-or-later
"""Submodule containing all the implemented potentials."""

from typing import (
    TYPE_CHECKING,
)

from .data_modifier import (
    DipoleChargeModifier,
)
from .deep_dipole import (
    DeepDipole,
)
from .deep_dos import (
    DeepDOS,
)
from .deep_eval import (
    DeepEval,
)
from .deep_polar import (
    DeepGlobalPolar,
    DeepPolar,
)
from .deep_pot import (
    DeepPot,
)
from .deep_wfc import (
    DeepWFC,
)
from .ewald_recp import (
    EwaldRecp,
)
from .model_devi import (
    calc_model_devi,
)

if TYPE_CHECKING:
    from deepmd.infer.deep_eval import (
        DeepEval,
    )

__all__ = [
    "DeepPotential",
    "DeepDipole",
    "DeepEval",
    "DeepGlobalPolar",
    "DeepPolar",
    "DeepPot",
    "DeepDOS",
    "DeepWFC",
    "DipoleChargeModifier",
    "EwaldRecp",
    "calc_model_devi",
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
    from deepmd.infer.deep_eval import (
        DeepEval,
    )

    return DeepEval(*args, **kwargs)
