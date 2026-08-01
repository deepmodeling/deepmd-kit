# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.utils.neighbor_stat import (
    NeighborStat,
)
from deepmd.utils.update_sel import (
    BaseUpdateSel,
)


class UpdateSel(BaseUpdateSel):
    r"""Neighbor-selection update computing :math:`n_{sel}` from statistics."""

    @property
    def neighbor_stat(self) -> type[NeighborStat]:
        return NeighborStat
