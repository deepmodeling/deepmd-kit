# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP

from ..common import (
    tf2_module,
)
from ..utils import network as _tf2_network  # noqa: F401
from ..utils import type_embed as _tf2_type_embed  # noqa: F401
from . import dpa1 as _tf2_dpa1  # noqa: F401
from . import repformers as _tf2_repformers  # noqa: F401
from . import se_t_tebd as _tf2_se_t_tebd  # noqa: F401
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("dpa2")
@tf2_module
class DescrptDPA2(DescrptDPA2DP):
    pass
