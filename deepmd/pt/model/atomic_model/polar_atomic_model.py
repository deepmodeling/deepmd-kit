# SPDX-License-Identifier: LGPL-3.0-or-later
from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPolarAtomicModel(DPAtomicModel):
    def apply_out_stat(self, ret, atype):
        # TODO: migrate bias
        return ret
