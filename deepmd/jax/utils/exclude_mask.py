# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP
from deepmd.jax.common import (
    flax_module,
    register_dpmodel_mapping,
)


@flax_module
class AtomExcludeMask(AtomExcludeMaskDP):
    pass


@flax_module
class PairExcludeMask(PairExcludeMaskDP):
    pass


register_dpmodel_mapping(
    AtomExcludeMaskDP,
    lambda v: AtomExcludeMask(v.ntypes, exclude_types=list(v.get_exclude_types())),
)

register_dpmodel_mapping(
    PairExcludeMaskDP,
    lambda v: PairExcludeMask(v.ntypes, exclude_types=list(v.get_exclude_types())),
)
