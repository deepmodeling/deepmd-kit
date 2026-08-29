# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt wrapper for the analytical bridging pair potential."""

from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model.inner_potential import (
    InnerPotential as InnerPotentialDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)


@torch_module
class InnerPotential(InnerPotentialDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# InnerPotential carries no trainable state (only the constant per-type
# atomic-number table, derived from the constructor arguments), so it
# implements no serialize()/deserialize(); rebuild it fresh from
# (type_map, mode).
register_dpmodel_mapping(
    InnerPotentialDP,
    lambda v: InnerPotential(type_map=v.type_map, mode=v.mode),
)
