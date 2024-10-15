# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP

from ..common import (
    to_array_api_strict_array,
)


class AtomExcludeMask(AtomExcludeMaskDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"type_mask"}:
            value = to_array_api_strict_array(value)
        return super().__setattr__(name, value)


class PairExcludeMask(PairExcludeMaskDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"type_mask"}:
            value = to_array_api_strict_array(value)
        return super().__setattr__(name, value)
