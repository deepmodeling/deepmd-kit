# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.polarizability_fitting import PolarFitting as PolarFittingDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("polar")
@torch_module
class PolarFitting(PolarFittingDP):
    pass


register_dpmodel_mapping(
    PolarFittingDP,
    lambda v: PolarFitting.deserialize(v.serialize()),
)
