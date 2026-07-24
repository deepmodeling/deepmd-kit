# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model.base_model import (
    make_base_model,
)


class BaseModel(make_base_model()):
    """Base class for pt_expt models.

    Provides the plugin registry so that model classes can be registered
    with ``@BaseModel.register("ener")`` etc. Deserialization is pure
    registry dispatch: descriptor-family-specific wire formats (e.g. the
    pt SeZM checkpoint layouts) are owned by the family's registered model
    class (see ``deepmd.pt_expt.model.dpa4_model.DPA4EnergyModel``), never
    by this base.

    See Also
    --------
    deepmd.dpmodel.model.base_model.BaseBaseModel
        Backend-independent BaseModel class.
    """
