# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP

from ..common import (
    array_api_strict_module,
)


@array_api_strict_module
class AtomExcludeMask(AtomExcludeMaskDP):
    pass


@array_api_strict_module
class PairExcludeMask(PairExcludeMaskDP):
    pass
