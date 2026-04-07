# SPDX-License-Identifier: LGPL-3.0-or-later


from deepmd.dpmodel.utils.exclude_mask import AtomExcludeMask as AtomExcludeMaskDP
from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask as PairExcludeMaskDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class AtomExcludeMask(AtomExcludeMaskDP):
    pass


register_dpmodel_mapping(
    AtomExcludeMaskDP,
    lambda v: AtomExcludeMask(v.ntypes, exclude_types=list(v.get_exclude_types())),
)


@torch_module
class PairExcludeMask(PairExcludeMaskDP):
    pass


register_dpmodel_mapping(
    PairExcludeMaskDP,
    lambda v: PairExcludeMask(v.ntypes, exclude_types=list(v.get_exclude_types())),
)
