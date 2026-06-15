# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
)

import deepmd.jax.utils.exclude_mask as _jax_exclude_mask  # noqa: F401
import deepmd.jax.utils.network as _jax_network  # noqa: F401
from deepmd.dpmodel.descriptor.repformers import (
    Atten2EquiVarApply as Atten2EquiVarApplyDP,
)
from deepmd.dpmodel.descriptor.repformers import Atten2Map as Atten2MapDP
from deepmd.dpmodel.descriptor.repformers import (
    Atten2MultiHeadApply as Atten2MultiHeadApplyDP,
)
from deepmd.dpmodel.descriptor.repformers import (
    DescrptBlockRepformers as DescrptBlockRepformersDP,
)
from deepmd.dpmodel.descriptor.repformers import LocalAtten as LocalAttenDP
from deepmd.dpmodel.descriptor.repformers import RepformerLayer as RepformerLayerDP
from deepmd.jax.common import (
    flax_module,
)


@flax_module
class DescrptBlockRepformers(DescrptBlockRepformersDP):
    pass


@flax_module
class Atten2Map(Atten2MapDP):
    pass


@flax_module
class Atten2MultiHeadApply(Atten2MultiHeadApplyDP):
    pass


@flax_module
class Atten2EquiVarApply(Atten2EquiVarApplyDP):
    pass


@flax_module
class LocalAtten(LocalAttenDP):
    pass


@flax_module
class RepformerLayer(RepformerLayerDP):
    _jax_data_list_attrs: ClassVar[set[str]] = {
        "g1_residual",
        "g2_residual",
        "h2_residual",
    }
