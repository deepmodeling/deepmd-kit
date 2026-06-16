# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.se_r import DescrptSeR as DescrptSeRDP

from ..common import (
    array_api_strict_module,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e2_r")
@BaseDescriptor.register("se_r")
@array_api_strict_module
class DescrptSeR(DescrptSeRDP):
    pass
