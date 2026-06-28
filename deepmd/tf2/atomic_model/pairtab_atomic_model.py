# SPDX-License-Identifier: LGPL-3.0-or-later
import deepmd.tf2.utils.exclude_mask as _tf2_exclude_mask  # noqa: F401
from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel as PairTabAtomicModelDP,
)
from deepmd.tf2.common import (
    tf2_module,
)
from deepmd.tf2.env import (
    stop_gradient,
    xp,
)


@tf2_module
class PairTabAtomicModel(PairTabAtomicModelDP):
    def forward_common_atomic(
        self,
        extended_coord: xp.ndarray,
        extended_atype: xp.ndarray,
        nlist: xp.ndarray,
        mapping: xp.ndarray | None = None,
        fparam: xp.ndarray | None = None,
        aparam: xp.ndarray | None = None,
        comm_dict: dict | None = None,
        charge_spin: xp.ndarray | None = None,
    ) -> dict[str, xp.ndarray]:
        del comm_dict  # tf2 path has no MPI ghost exchange
        return super().forward_common_atomic(
            extended_coord,
            extended_atype,
            stop_gradient(nlist),
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
        )
