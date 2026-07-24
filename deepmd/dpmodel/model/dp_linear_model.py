# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel linear energy model: a make_model CM over the linear atomic-model
composition (twin of ``deepmd.pt_expt.model.dp_linear_model``).
"""

from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.dpmodel.model.make_model import (
    make_model,
)

DPLinearModel_ = make_model(LinearEnergyAtomicModel, T_Bases=(NativeOP, BaseModel))


@BaseModel.register("linear_ener")
@BaseModel.register(
    "linear"
)  # the atomic dict's registered type (CM.serialize is the flat atomic dict)
class LinearEnergyModel(DPModelCommon, DPLinearModel_):
    r"""Energy model over a linear combination of atomic models.

    The atomic energy is the weighted sum of the children's atomic
    energies; on the NeighborGraph route every child consumes the same
    graph, so the summed energy differentiates through one shared edge
    backward. Used e.g. for analytical bridging compositions
    (learned model + :class:`~deepmd.dpmodel.atomic_model.inter_potential.InterPotentialAtomicModel`).
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPLinearModel_.__init__(self, *args, **kwargs)
