# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
)

import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.descriptor.repflows import (
    DescrptBlockRepflows as DescrptBlockRepflowsDP,
)
from deepmd.dpmodel.descriptor.repflows import RepFlowLayer as RepFlowLayerDP
from deepmd.jax.common import (
    flax_module,
)


@flax_module
class DescrptBlockRepflows(DescrptBlockRepflowsDP):
    pass


@flax_module
class RepFlowLayer(RepFlowLayerDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {
        "n_residual",
        "e_residual",
        "a_residual",
    }
