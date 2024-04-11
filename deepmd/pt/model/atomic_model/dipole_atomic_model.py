# SPDX-License-Identifier: LGPL-3.0-or-later
from .dp_atomic_model import (
    DPAtomicModel,
)


class DPDipoleAtomicModel(DPAtomicModel):
    def apply_out_stat(self, ret, atype):
        # dipole not applying bias
        return ret
