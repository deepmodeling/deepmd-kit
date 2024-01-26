# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from .dp_atomic_model import (
    DPAtomicModel,
)
from .make_model import (
    make_model,
)

DPModel = make_model(DPAtomicModel)


class EnergyModel(DPModel):
    model_type = "ener"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord, atype, box, do_atomic_virial=do_atomic_virial
        )
        if self.fitting_net is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atomic_virial"] = model_ret["energy_derv_c"].squeeze(
                        -3
                    )
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-3)
            else:
                model_predict["force"] = model_ret["dforce"]
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.common_forward_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            do_atomic_virial=do_atomic_virial,
        )
        if self.fitting_net is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-3)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
        else:
            model_predict = model_ret
        return model_predict


# should be a stand-alone function!!!!
def process_nlist(
    nlist,
    extended_atype,
    mapping: Optional[torch.Tensor] = None,
):
    # process the nlist_type and nlist_loc
    nframes, nloc = nlist.shape[:2]
    nmask = nlist == -1
    nlist[nmask] = 0
    if mapping is not None:
        nlist_loc = torch.gather(
            mapping,
            dim=1,
            index=nlist.reshape(nframes, -1),
        ).reshape(nframes, nloc, -1)
        nlist_loc[nmask] = -1
    else:
        nlist_loc = None
    nlist_type = torch.gather(
        extended_atype,
        dim=1,
        index=nlist.reshape(nframes, -1),
    ).reshape(nframes, nloc, -1)
    nlist_type[nmask] = -1
    nlist[nmask] = -1
    return nlist_loc, nlist_type, nframes, nloc


def process_nlist_gathered(
    nlist,
    extended_atype,
    split_sel: List[int],
    mapping: Optional[torch.Tensor] = None,
):
    nlist_list = list(torch.split(nlist, split_sel, -1))
    nframes, nloc = nlist_list[0].shape[:2]
    nlist_type_list = []
    nlist_loc_list = []
    for nlist_item in nlist_list:
        nmask = nlist_item == -1
        nlist_item[nmask] = 0
        if mapping is not None:
            nlist_loc_item = torch.gather(
                mapping, dim=1, index=nlist_item.reshape(nframes, -1)
            ).reshape(nframes, nloc, -1)
            nlist_loc_item[nmask] = -1
            nlist_loc_list.append(nlist_loc_item)
        nlist_type_item = torch.gather(
            extended_atype, dim=1, index=nlist_item.reshape(nframes, -1)
        ).reshape(nframes, nloc, -1)
        nlist_type_item[nmask] = -1
        nlist_type_list.append(nlist_type_item)
        nlist_item[nmask] = -1

    if mapping is not None:
        nlist_loc = torch.cat(nlist_loc_list, -1)
    else:
        nlist_loc = None
    nlist_type = torch.cat(nlist_type_list, -1)
    return nlist_loc, nlist_type, nframes, nloc
