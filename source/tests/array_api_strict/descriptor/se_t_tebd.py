# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd as DescrptBlockSeTTebdDP,
)
from deepmd.dpmodel.descriptor.se_t_tebd import DescrptSeTTebd as DescrptSeTTebdDP

from ..common import (
    array_api_strict_module,
)
from ..utils import exclude_mask as _strict_exclude_mask  # noqa: F401
from ..utils import network as _strict_network  # noqa: F401
from ..utils import type_embed as _strict_type_embed  # noqa: F401


@array_api_strict_module
class DescrptBlockSeTTebd(DescrptBlockSeTTebdDP):
    pass


@array_api_strict_module
class DescrptSeTTebd(DescrptSeTTebdDP):
    pass
