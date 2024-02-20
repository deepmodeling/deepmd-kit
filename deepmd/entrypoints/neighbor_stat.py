# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
)

from deepmd.backend.backend import (
    Backend,
)
from deepmd.common import (
    expand_sys_str,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

log = logging.getLogger(__name__)


def neighbor_stat(
    *,
    system: str,
    rcut: float,
    type_map: List[str],
    mixed_type: bool = False,
    backend: str = "tensorflow",
    **kwargs,
):
    """Calculate neighbor statistics.

    Parameters
    ----------
    system : str
        system to stat
    rcut : float
        cutoff radius
    type_map : list[str]
        type map
    mixed_type : bool, optional, default=False
        treat all types as a single type
    backend : str, optional, default="tensorflow"
        backend to use
    **kwargs
        additional arguments

    Examples
    --------
    >>> neighbor_stat(
    ...     system=".",
    ...     rcut=6.0,
    ...     type_map=[
    ...         "C",
    ...         "H",
    ...         "O",
    ...         "N",
    ...         "P",
    ...         "S",
    ...         "Mg",
    ...         "Na",
    ...         "HW",
    ...         "OW",
    ...         "mNa",
    ...         "mCl",
    ...         "mC",
    ...         "mH",
    ...         "mMg",
    ...         "mN",
    ...         "mO",
    ...         "mP",
    ...     ],
    ... )
    min_nbor_dist: 0.6599510670195264
    max_nbor_size: [23, 26, 19, 16, 2, 2, 1, 1, 72, 37, 5, 0, 31, 29, 1, 21, 20, 5]
    """
    backends = Backend.get_backends_by_feature(Backend.Feature.NEIGHBOR_STAT)
    try:
        backend_obj = backends[backend]()
    except KeyError:
        raise ValueError(f"Invalid backend {backend}")
    NeighborStat = backend_obj.neighbor_stat
    all_sys = expand_sys_str(system)
    if not len(all_sys):
        raise RuntimeError("Did not find valid system")
    data = DeepmdDataSystem(
        systems=all_sys,
        batch_size=1,
        test_size=1,
        rcut=rcut,
        type_map=type_map,
    )
    data.get_batch()
    nei = NeighborStat(data.get_ntypes(), rcut, mixed_type=mixed_type)
    min_nbor_dist, max_nbor_size = nei.get_stat(data)
    log.info("min_nbor_dist: %f" % min_nbor_dist)
    log.info("max_nbor_size: %s" % str(max_nbor_size))
    return min_nbor_dist, max_nbor_size
