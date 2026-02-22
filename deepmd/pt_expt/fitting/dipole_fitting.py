# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.dipole_fitting import DipoleFitting as DipoleFittingDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("dipole")
@torch_module
class DipoleFitting(DipoleFittingDP):
    pass


register_dpmodel_mapping(
    DipoleFittingDP,
    lambda v: DipoleFitting.deserialize(v.serialize()),
)
