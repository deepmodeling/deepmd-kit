# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.neighbor_stat import (
    NeighborStat,
)

log = logging.getLogger(__name__)


class BaseUpdateSel(ABC):
    """Update the sel field in the descriptor."""

    def update_one_sel(
        self,
        train_data: DeepmdDataSystem,
        type_map: Optional[List[str]],
        rcut: float,
        sel: Union[int, List[int], str],
        mixed_type: bool = False,
    ) -> Tuple[float, List[int]]:
        min_nbor_dist, tmp_sel = self.get_nbor_stat(
            train_data,
            type_map,
            rcut,
            mixed_type=mixed_type,
        )
        if isinstance(sel, int):
            # convert to list and finnally convert back to int
            sel = [sel]
        if self.parse_auto_sel(sel):
            ratio = self.parse_auto_sel_ratio(sel)
            sel = [int(self.wrap_up_4(ii * ratio)) for ii in tmp_sel]
        else:
            # sel is set by user
            for ii, (tt, dd) in enumerate(zip(tmp_sel, sel)):
                if dd and tt > dd:
                    # we may skip warning for sel=0, where the user is likely
                    # to exclude such type in the descriptor
                    log.warning(
                        "sel of type %d is not enough! The expected value is "
                        "not less than %d, but you set it to %d. The accuracy"
                        " of your model may get worse." % (ii, tt, dd)
                    )
        return min_nbor_dist, sel

    def parse_auto_sel(self, sel):
        if not isinstance(sel, str):
            return False
        words = sel.split(":")
        if words[0] == "auto":
            return True
        else:
            return False

    def parse_auto_sel_ratio(self, sel):
        if not self.parse_auto_sel(sel):
            raise RuntimeError(f"invalid auto sel format {sel}")
        else:
            words = sel.split(":")
            if len(words) == 1:
                ratio = 1.1
            elif len(words) == 2:
                ratio = float(words[1])
            else:
                raise RuntimeError(f"invalid auto sel format {sel}")
            return ratio

    def wrap_up_4(self, xx):
        return 4 * ((int(xx) + 3) // 4)

    def get_nbor_stat(
        self,
        train_data: DeepmdDataSystem,
        type_map: Optional[List[str]],
        rcut: float,
        mixed_type: bool = False,
    ) -> Tuple[float, Union[int, List[int]]]:
        """Get the neighbor statistics of the data.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            The training data.
        type_map : Optional[List[str]]
            The type map.
        rcut : float
            The cutoff radius.
        mixed_type : bool, optional
            Whether to mix the types.

        Returns
        -------
        min_nbor_dist : float
            The minimum neighbor distance.
        max_nbor_size : List[int]
            The maximum neighbor size.
        """
        if type_map and len(type_map) == 0:
            type_map = None
        train_data.get_batch()
        data_ntypes = train_data.get_ntypes()
        if type_map is not None:
            map_ntypes = len(type_map)
        else:
            map_ntypes = data_ntypes
        ntypes = max([map_ntypes, data_ntypes])

        neistat = self.neighbor_stat(ntypes, rcut, mixed_type=mixed_type)

        min_nbor_dist, max_nbor_size = neistat.get_stat(train_data)

        return min_nbor_dist, max_nbor_size

    @property
    @abstractmethod
    def neighbor_stat(self) -> Type[NeighborStat]:
        pass

    def get_min_nbor_dist(
        self,
        train_data: DeepmdDataSystem,
    ):
        min_nbor_dist, _ = self.get_nbor_stat(
            train_data,
            None,  # type_map doesn't affect min_nbor_dist
            1e-6,  # we don't need the max_nbor_size
            mixed_type=True,  # mixed_types doesn't affect min_nbor_dist
        )
        return min_nbor_dist
