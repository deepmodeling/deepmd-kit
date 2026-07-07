# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.dpmodel.descriptor.dpa4_nn.activation import (
    SwiGLU,
)
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    GridProduct,
)
from deepmd.dpmodel.descriptor.dpa4_nn.radial import (
    BridgingSwitch,
    C3CutoffEnvelope,
    InnerClamp,
)
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator,
)

from ..common import (
    array_api_strict_module,
    register_dpmodel_mapping,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@array_api_strict_module
class DescrptDPA4(DescrptDPA4DP):
    pass


for _stateless_cls in (
    BridgingSwitch,
    C3CutoffEnvelope,
    GridProduct,
    InnerClamp,
    SwiGLU,
    WignerDCalculator,
):
    register_dpmodel_mapping(_stateless_cls, lambda v: v)
