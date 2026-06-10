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

# Register fake tensor implementations for custom tabulate ops.
# comm.py (border_op fake/autograd) is NOT imported here — its
# ensure_comm_registered() is called lazily from the with_comm_dict
# export path in serialization.py to avoid eager libdeepmd_op_pt.so
# loading that breaks fake-op registration order in tests.
from deepmd.pt_expt.utils import tabulate_ops  # noqa: F401

__all__ = [
    "AtomExcludeMask",
    "NetworkCollection",
    "PairExcludeMask",
    "TypeEmbedNet",
]
