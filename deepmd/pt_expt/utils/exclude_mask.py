# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP
from deepmd.pt_expt.utils import (
    env,
)


class AtomExcludeMask(AtomExcludeMaskDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "type_mask":
            value = None if value is None else torch.as_tensor(value, device=env.DEVICE)
        return super().__setattr__(name, value)


class PairExcludeMask(PairExcludeMaskDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "type_mask":
            value = None if value is None else torch.as_tensor(value, device=env.DEVICE)
        return super().__setattr__(name, value)
