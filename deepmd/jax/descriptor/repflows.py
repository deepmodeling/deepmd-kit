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
    # JAX/jax2tf export cannot represent the compact dynamic layout where
    # boolean indexing creates runtime-sized ``n_edge``/``n_angle`` arrays.
    # Use the fixed-capacity dynamic layout instead:
    #   edges  = nf * nloc * e_sel
    #   angles = nf * nloc * a_sel * a_sel
    # Invalid slots are still masked by switch weights, so DPA-3 outputs match
    # the compact dynamic implementation.
    _use_static_dynamic_sel = True


@flax_module
class RepFlowLayer(RepFlowLayerDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {
        "n_residual",
        "e_residual",
        "a_residual",
    }
    # Keep the layer-level graph operations in the same fixed-capacity layout
    # selected by the owning descriptor block.
    _use_static_dynamic_sel = True
