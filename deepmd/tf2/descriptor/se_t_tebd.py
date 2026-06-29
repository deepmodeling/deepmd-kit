# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd as DescrptBlockSeTTebdDP,
)
from deepmd.dpmodel.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdDP

from ..common import (
    tf2_module,
)
from ..utils import exclude_mask as _tf2_exclude_mask  # noqa: F401
from ..utils import network as _tf2_network  # noqa: F401
from ..utils import type_embed as _tf2_type_embed  # noqa: F401


@tf2_module
class DescrptBlockSeTTebd(DescrptBlockSeTTebdDP):
    pass


@tf2_module
class DescrptSeTTebd(DescrptSeTTebdDP):
    pass
