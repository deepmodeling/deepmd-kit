# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP
from deepmd.pt_expt.common import (
    dpmodel_setattr,
    register_dpmodel_mapping,
)


class AtomExcludeMask(AtomExcludeMaskDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        AtomExcludeMaskDP.__init__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        handled, value = dpmodel_setattr(self, name, value)
        if not handled:
            super().__setattr__(name, value)


register_dpmodel_mapping(
    AtomExcludeMaskDP,
    lambda v: AtomExcludeMask(v.ntypes, exclude_types=list(v.get_exclude_types())),
)


class PairExcludeMask(PairExcludeMaskDP, torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        PairExcludeMaskDP.__init__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        handled, value = dpmodel_setattr(self, name, value)
        if not handled:
            super().__setattr__(name, value)


register_dpmodel_mapping(
    PairExcludeMaskDP,
    lambda v: PairExcludeMask(v.ntypes, exclude_types=list(v.get_exclude_types())),
)
