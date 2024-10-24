# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.pd.utils.neighbor_stat import (
    NeighborStat,
)
from deepmd.utils.update_sel import (
    BaseUpdateSel,
)


class UpdateSel(BaseUpdateSel):
    @property
    def neighbor_stat(self) -> type[NeighborStat]:
        return NeighborStat
