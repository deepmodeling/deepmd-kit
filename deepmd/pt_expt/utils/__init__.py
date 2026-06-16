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

# Note: tabulate_ops (fake-op registration for the compressed tabulate path)
# and comm.py (border_op fake/autograd) are intentionally NOT imported here.
# Their ensure_*_registered() helpers are called lazily from the paths that
# actually need them (compression entry / with_comm_dict export). Eager-loading
# them at package import time pulls custom-op registration onto the plain pt
# (torch.jit) inference path — `deepmd.pt.infer.deep_eval` imports the vesin
# neighbor list from this package — which crashes `dp test` when the C++ op
# library is absent (the pt descriptor fallback monkeypatches a plain Python
# function onto torch.ops.deepmd, so register_fake raises "operator does not
# exist"). See tests/pt_expt/utils/test_tabulate_ops_lazy.py.

__all__ = [
    "AtomExcludeMask",
    "NetworkCollection",
    "PairExcludeMask",
    "TypeEmbedNet",
]
