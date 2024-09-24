# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP
from deepmd.jax.common import (
    to_jax_array,
)


class PairExcludeMask(PairExcludeMaskDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"type_mask"}:
            value = to_jax_array(value)
        return super().__setattr__(name, value)
