# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP

from ..common import (
    array_api_strict_module,
)
from ..utils import network as _strict_network  # noqa: F401
from ..utils import type_embed as _strict_type_embed  # noqa: F401
from . import dpa1 as _strict_dpa1  # noqa: F401
from . import repformers as _strict_repformers  # noqa: F401
from . import se_t_tebd as _strict_se_t_tebd  # noqa: F401
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("dpa2")
@array_api_strict_module
class DescrptDPA2(DescrptDPA2DP):
    pass
