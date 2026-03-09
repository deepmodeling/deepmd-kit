# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.model.base_model import (
    make_base_model,
)


class BaseModel(make_base_model()):
    """Base class for pt_expt models.

    Provides the plugin registry so that model classes can be
    registered with ``@BaseModel.register("ener")`` etc.

    See Also
    --------
    deepmd.dpmodel.model.base_model.BaseBaseModel
        Backend-independent BaseModel class.
    """

    pass
