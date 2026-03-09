# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Any,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)


def make_hessian_model(T_Model: type) -> type:
    """Make a model that can compute Hessian.

    With the JAX-mirrored approach, hessian is computed in
    ``forward_common_atomic`` (in make_model.py) on extended coordinates.
    This wrapper only needs to override ``atomic_output_def()`` to set
    ``r_hessian=True``, and ``communicate_extended_output`` in dpmodel
    naturally maps it from nall to nloc.

    Parameters
    ----------
    T_Model
        The model. Should provide the ``atomic_output_def`` method.

    Returns
    -------
    The model that computes hessian.

    """

    class CM(T_Model):
        def __init__(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                *args,
                **kwargs,
            )
            self.hess_fitting_def = copy.deepcopy(super().atomic_output_def())

        def requires_hessian(
            self,
            keys: str | list[str],
        ) -> None:
            """Set which output variable(s) requires hessian."""
            if isinstance(keys, str):
                keys = [keys]
            for kk in self.hess_fitting_def.keys():
                if kk in keys:
                    self.hess_fitting_def[kk].r_hessian = True

        def atomic_output_def(self) -> FittingOutputDef:
            """Get the fitting output def."""
            return self.hess_fitting_def

    return CM
