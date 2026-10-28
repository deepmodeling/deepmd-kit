# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4/SeZM-family energy model.

A THIN subclass of the descriptor-agnostic :class:`EnergyModel` that owns
exactly the dpa4-family concerns: the family's registry wire types
(``dpa4_ener``/``sezm_ener`` fitting-type dispatch keys). The generic
``EnergyModel`` stays free of descriptor-specific registrations.
"""

from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)


@BaseModel.register("dpa4_ener")
@BaseModel.register("sezm_ener")
class DPA4EnergyModel(EnergyModel):
    r"""Energy model for the DPA4/SeZM descriptor family.

    Behaviorally identical to :class:`EnergyModel`; exists so the
    dpa4-family wire types dispatch to a class that owns them instead of
    polluting the generic energy model's registry.
    """
