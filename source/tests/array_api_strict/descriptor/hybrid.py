# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP

from ..common import (
    array_api_strict_module,
)
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("hybrid")
@array_api_strict_module
class DescrptHybrid(DescrptHybridDP):
    pass
