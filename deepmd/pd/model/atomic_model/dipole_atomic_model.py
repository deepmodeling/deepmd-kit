# SPDX-License-Identifier: LGPL-3.0-or-later

import paddle

from deepmd.pd.model.task.dipole import (
    DipoleFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, DipoleFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, paddle.Tensor],
        atype: paddle.Tensor,
    ):
        # dipole not applying bias
        return ret
