# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP

from ..common import (
    tf2_module,
)


@tf2_module
class AtomExcludeMask(AtomExcludeMaskDP):
    pass


@tf2_module
class PairExcludeMask(PairExcludeMaskDP):
    pass
