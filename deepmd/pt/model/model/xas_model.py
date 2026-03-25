# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.pt.model.atomic_model import (
    DPXASAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .property_model import (
    PropertyModel,
)


@BaseModel.register("xas")
class XASModel(PropertyModel):
    """Model for XAS spectrum fitting.

    Identical to :class:`PropertyModel` but uses :class:`DPXASAtomicModel`
    as the underlying atomic model, which carries the per-(absorbing_type,
    edge) energy reference buffer ``xas_e_ref`` in the checkpoint.  This
    buffer is populated by :meth:`deepmd.pt.loss.xas.XASLoss.compute_output_stats`
    before training starts and restored at inference time so that absolute
    edge energies are available without any external reference files.
    """

    model_type = "xas"

    def __init__(
        self,
        descriptor: Any,
        fitting: Any,
        type_map: Any,
        **kwargs: Any,
    ) -> None:
        xas_atomic = DPXASAtomicModel(descriptor, fitting, type_map, **kwargs)
        super().__init__(
            descriptor, fitting, type_map, atomic_model_=xas_atomic, **kwargs
        )
