# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
)

from deepmd.dpmodel.descriptor.repflows import (
    DescrptBlockRepflows as DescrptBlockRepflowsDP,
)
from deepmd.dpmodel.descriptor.repflows import RepFlowLayer as RepFlowLayerDP

from ..common import (
    tf2_module,
)
from ..utils import exclude_mask as _tf2_exclude_mask  # noqa: F401
from ..utils import network as _tf2_network  # noqa: F401


@tf2_module
class DescrptBlockRepflows(DescrptBlockRepflowsDP):
    pass


@tf2_module
class RepFlowLayer(RepFlowLayerDP):
    _tf2_data_list_attrs: ClassVar[set[str]] = {
        "n_residual",
        "e_residual",
        "a_residual",
    }
