# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.utils.env_mat import (
    EnvMat,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
)

from .exclude_mask import (
    AtomExcludeMask,
    PairExcludeMask,
)
from .network import (
    NetworkCollection,
)
from .type_embed import (
    TypeEmbedNet,
)

# Register EnvMat with identity converter - it doesn't need wrapping
# as it's a stateless utility class
register_dpmodel_mapping(EnvMat, lambda v: v)

__all__ = [
    "AtomExcludeMask",
    "NetworkCollection",
    "PairExcludeMask",
    "TypeEmbedNet",
]
