# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
)

from deepmd.dpmodel.descriptor.repflows import (
    DescrptBlockRepflows as DescrptBlockRepflowsDP,
)
from deepmd.dpmodel.descriptor.repflows import RepFlowLayer as RepFlowLayerDP

from ..common import (
    array_api_strict_module,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401


@array_api_strict_module
class DescrptBlockRepflows(DescrptBlockRepflowsDP):
    pass


@array_api_strict_module
class RepFlowLayer(RepFlowLayerDP):
    _array_api_strict_data_list_attrs: ClassVar[set[str]] = {
        "n_residual",
        "e_residual",
        "a_residual",
    }
