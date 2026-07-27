# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.se_t import DescrptSeT as DescrptSeTDP

from ..common import (
    tf2_module,
)
from ..utils import exclude_mask as _tf2_exclude_mask  # noqa: F401
from ..utils import network as _tf2_network  # noqa: F401
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("se_e3")
@BaseDescriptor.register("se_at")
@BaseDescriptor.register("se_a_3be")
@tf2_module
class DescrptSeT(DescrptSeTDP):
    pass
