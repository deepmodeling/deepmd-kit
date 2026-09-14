# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model wrapper for Uni-Mol pretraining on a DPA backbone."""

from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model import (
    DPUniMolDPAAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPUniMolDPAPretrainModel_ = make_model(
    DPUniMolDPAAtomicModel, T_Bases=(NativeOP, BaseModel)
)


@BaseModel.register("unimol_dpa_pretrain")
class UniMolDPAPretrainModel(DPModelCommon, DPUniMolDPAPretrainModel_):
    r"""Uni-Mol's pretraining objective on a DPA backbone.

    Predicts the element of every corrupted atom, denoises the coordinates from
    the backbone's equivariant state, and predicts the clean pairwise distances
    from pairs of atom representations. Nothing reduces to a frame total and
    nothing is differentiated with respect to the coordinates: this is
    representation learning, not a potential energy surface.

    Unlike the Uni-Mol backbone's own model, this one is periodic-capable in
    principle, because the backbone it reads is. Whether a periodic frame makes
    sense for this objective is a question about the data, not about the model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        DPUniMolDPAPretrainModel_.__init__(self, *args, **kwargs)
