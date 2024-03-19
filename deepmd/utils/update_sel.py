# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    abstractmethod,
)
from typing import (
    Type,
)

from deepmd.utils.data_system import (
    get_data,
)
from deepmd.utils.neighbor_stat import (
    NeighborStat,
)

log = logging.getLogger(__name__)


class BaseUpdateSel:
    """Update the sel field in the descriptor."""

    def update_one_sel(
        self,
        jdata,
        descriptor,
        mixed_type: bool = False,
        rcut_key="rcut",
        sel_key="sel",
    ):
        rcut = descriptor[rcut_key]
        tmp_sel = self.get_sel(
            jdata,
            rcut,
            mixed_type=mixed_type,
        )
        sel = descriptor[sel_key]
        if isinstance(sel, int):
            # convert to list and finnally convert back to int
            sel = [sel]
        if self.parse_auto_sel(descriptor[sel_key]):
            ratio = self.parse_auto_sel_ratio(descriptor[sel_key])
            descriptor[sel_key] = sel = [
                int(self.wrap_up_4(ii * ratio)) for ii in tmp_sel
            ]
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
        if mixed_type:
            descriptor[sel_key] = sum(sel)
        return descriptor

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

    def get_sel(self, jdata, rcut, mixed_type: bool = False):
        _, max_nbor_size = self.get_nbor_stat(jdata, rcut, mixed_type=mixed_type)
        return max_nbor_size

    def get_rcut(self, jdata):
        if jdata["model"].get("type") == "pairwise_dprc":
            return max(
                jdata["model"]["qm_model"]["descriptor"]["rcut"],
                jdata["model"]["qmmm_model"]["descriptor"]["rcut"],
            )
        descrpt_data = jdata["model"]["descriptor"]
        rcut_list = []
        if descrpt_data["type"] == "hybrid":
            for ii in descrpt_data["list"]:
                rcut_list.append(ii["rcut"])
        else:
            rcut_list.append(descrpt_data["rcut"])
        return max(rcut_list)

    def get_type_map(self, jdata):
        return jdata["model"].get("type_map", None)

    def get_nbor_stat(self, jdata, rcut, mixed_type: bool = False):
        # it seems that DeepmdDataSystem does not need rcut
        # it's not clear why there is an argument...
        # max_rcut = get_rcut(jdata)
        max_rcut = rcut
        type_map = self.get_type_map(jdata)

        if type_map and len(type_map) == 0:
            type_map = None
        multi_task_mode = "data_dict" in jdata["training"]
        if not multi_task_mode:
            train_data = get_data(
                jdata["training"]["training_data"], max_rcut, type_map, None
            )
            train_data.get_batch()
        else:
            assert (
                type_map is not None
            ), "Data stat in multi-task mode must have available type_map! "
            train_data = None
            for systems in jdata["training"]["data_dict"]:
                tmp_data = get_data(
                    jdata["training"]["data_dict"][systems]["training_data"],
                    max_rcut,
                    type_map,
                    None,
                )
                tmp_data.get_batch()
                assert tmp_data.get_type_map(), f"In multi-task mode, 'type_map.raw' must be defined in data systems {systems}! "
                if train_data is None:
                    train_data = tmp_data
                else:
                    train_data.system_dirs += tmp_data.system_dirs
                    train_data.data_systems += tmp_data.data_systems
                    train_data.natoms += tmp_data.natoms
                    train_data.natoms_vec += tmp_data.natoms_vec
                    train_data.default_mesh += tmp_data.default_mesh
        data_ntypes = train_data.get_ntypes()
        if type_map is not None:
            map_ntypes = len(type_map)
        else:
            map_ntypes = data_ntypes
        ntypes = max([map_ntypes, data_ntypes])

        neistat = self.neighbor_stat(ntypes, rcut, mixed_type=mixed_type)

        min_nbor_dist, max_nbor_size = neistat.get_stat(train_data)
        self.hook(min_nbor_dist, max_nbor_size)

        return min_nbor_dist, max_nbor_size

    @property
    @abstractmethod
    def neighbor_stat(self) -> Type[NeighborStat]:
        pass

    @abstractmethod
    def hook(self, min_nbor_dist, max_nbor_size):
        pass

    def get_min_nbor_dist(self, jdata, rcut):
        min_nbor_dist, _ = self.get_nbor_stat(jdata, rcut)
        return min_nbor_dist
