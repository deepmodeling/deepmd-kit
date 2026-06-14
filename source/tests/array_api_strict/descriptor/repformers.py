# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    ClassVar,
)

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

from ..common import (
    array_api_strict_module,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401


@array_api_strict_module
class DescrptBlockRepformers(DescrptBlockRepformersDP):
    pass


@array_api_strict_module
class Atten2Map(Atten2MapDP):
    pass


@array_api_strict_module
class Atten2MultiHeadApply(Atten2MultiHeadApplyDP):
    pass


@array_api_strict_module
class Atten2EquiVarApply(Atten2EquiVarApplyDP):
    pass


@array_api_strict_module
class LocalAtten(LocalAttenDP):
    pass


@array_api_strict_module
class RepformerLayer(RepformerLayerDP):
    _array_api_strict_data_list_attrs: ClassVar[set[str]] = {
        "g1_residual",
        "g2_residual",
        "h2_residual",
    }
