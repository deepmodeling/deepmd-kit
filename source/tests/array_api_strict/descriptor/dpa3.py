# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3DP

from ..common import (
    array_api_strict_module,
)
from ..utils import network as _strict_network  # noqa: F401
from ..utils import type_embed as _strict_type_embed  # noqa: F401
from . import repflows as _strict_repflows  # noqa: F401
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("dpa3")
@array_api_strict_module
class DescrptDPA3(DescrptDPA3DP):
    pass
