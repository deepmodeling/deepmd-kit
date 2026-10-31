# SPDX-License-Identifier: LGPL-3.0-or-later
"""Long-range modules of the atomic energy of a descriptor-fitting model.

A long-range module is a component of the atomic energy model beside its
energy fitting: it reads the descriptor features shared with that fitting and
the topologies it declares, and contributes an energy per atom. Every consumer
of the atomic model therefore evaluates the complete potential.
"""

from .base_long_range import (
    BaseLongRange,
    check_long_range_support,
    make_base_long_range,
)
from .kernel import (
    COULOMB_CONSTANT,
    BaseInteractionKernel,
    LESKernel,
    SOGKernel,
)
from .latent_charge import (
    LatentChargeLongRange,
)

__all__ = [
    "COULOMB_CONSTANT",
    "BaseInteractionKernel",
    "BaseLongRange",
    "LESKernel",
    "LatentChargeLongRange",
    "SOGKernel",
    "check_long_range_support",
    "make_base_long_range",
]
