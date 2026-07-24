# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.fitting.dipole_fitting import (
    DipoleFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    r"""Atomic dipole model reconstructed from descriptor rotation matrices.

    The fitting network predicts local coefficients and contracts them with
    the equivariant descriptor output:

    .. math::

       \mathbf M_i=F_\theta(\mathcal D_i),\qquad
       \boldsymbol\mu_i=\mathbf M_i\mathbf R_i,

    where :math:`\mathbf R_i\in\mathbb R^{m_1\times3}` is the descriptor
    rotation matrix.  This contraction produces the lab-frame vector.

    Frame dipoles are additive: :math:`\boldsymbol\mu=\sum_i\boldsymbol\mu_i`.
    """

    def __init__(
        self,
        descriptor: BaseDescriptor,
        fitting: BaseFitting,
        type_map: list[str],
        **kwargs: Any,
    ) -> None:
        if not isinstance(fitting, DipoleFitting):
            raise TypeError(
                "fitting must be an instance of DipoleFitting for DPDipoleAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, Array],
        atype: Array,
    ) -> dict[str, Array]:
        # dipole not applying bias
        return ret
