# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Type,
)

from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.neighbor_stat import (
    NeighborStat,
)
from deepmd.utils.update_sel import (
    BaseUpdateSel,
)


class UpdateSel(BaseUpdateSel):
    @property
    def neighbor_stat(self) -> Type[NeighborStat]:
        return NeighborStat

    def hook(self, min_nbor_dist, max_nbor_size):
        # moved from traier.py as duplicated
        # TODO: this is a simple fix but we should have a clear
        #       architecture to call neighbor stat
        tf.constant(
            min_nbor_dist,
            name="train_attr/min_nbor_dist",
            dtype=GLOBAL_ENER_FLOAT_PRECISION,
        )
        tf.constant(max_nbor_size, name="train_attr/max_nbor_size", dtype=tf.int32)
