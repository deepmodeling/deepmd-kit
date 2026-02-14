# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.atomic_model.energy_atomic_model import (
    DPEnergyAtomicModel as DPEnergyAtomicModelDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPEnergyAtomicModel(DPAtomicModel):
    """Energy atomic model for pt_expt backend.

    This is a thin wrapper around DPAtomicModel that validates
    the fitting is an EnergyFittingNet or InvarFitting.
    """

    pass


register_dpmodel_mapping(
    DPEnergyAtomicModelDP,
    lambda v: DPEnergyAtomicModel.deserialize(v.serialize()),
)
