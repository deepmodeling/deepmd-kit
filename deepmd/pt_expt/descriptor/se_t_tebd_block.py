# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd as DescrptBlockSeTTebdDP,
)
from deepmd.pt_expt.common import (
    dpmodel_setattr,
    register_dpmodel_mapping,
)


class DescrptBlockSeTTebd(DescrptBlockSeTTebdDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        DescrptBlockSeTTebdDP.__init__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        handled, value = dpmodel_setattr(self, name, value)
        if not handled:
            super().__setattr__(name, value)


register_dpmodel_mapping(
    DescrptBlockSeTTebdDP,
    lambda v: DescrptBlockSeTTebd.deserialize(v.serialize()),
)
