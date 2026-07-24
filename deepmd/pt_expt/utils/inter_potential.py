# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt wrapper for the analytical bridging pair potential."""

from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model.inter_potential import (
    InterPotential as InterPotentialDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class InterPotential(InterPotentialDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# InterPotential carries no trainable state (only the constant per-type
# atomic-number table, derived from the constructor arguments), so it
# implements no serialize()/deserialize(); rebuild it fresh from
# (type_map, mode).
register_dpmodel_mapping(
    InterPotentialDP,
    lambda v: InterPotential(type_map=v.type_map, mode=v.mode),
)
