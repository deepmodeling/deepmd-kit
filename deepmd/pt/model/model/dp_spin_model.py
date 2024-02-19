# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPSpinWrapperAtomicModel,
)

from .make_model import (
    make_model,
)

DPSpinModel_ = make_model(DPSpinWrapperAtomicModel)


class SpinModel(DPSpinModel_):
    """A spin model wrapper, with spin input preprocess and output split."""

    model_type = "ener"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def preprocess_spin_input(self, coord, atype, spin):
        nframes, nloc = coord.shape[:-1]
        atype_spin = torch.concat([atype, atype + self.spin.ntypes_real], dim=-1)
        virtual_scale_mask = self.spin.get_virtual_scale_mask()
        virtual_coord = coord + spin * torch.gather(
            virtual_scale_mask, -1, index=atype.view(-1)
        ).reshape([nframes, nloc, 1])
        coord_spin = torch.concat([coord, virtual_coord], dim=-2)
        return coord_spin, atype_spin

    def preprocess_spin_output(self, atype, force):
        nframes, nloc_double = force.shape[:-1]
        nloc = nloc_double // 2
        virtual_scale_mask = self.spin.get_virtual_scale_mask()
        atmoic_mask = torch.gather(
            virtual_scale_mask, -1, index=atype.view(-1)
        ).reshape([nframes, nloc, 1])
        force_real, force_mag = torch.split(force, [nloc, nloc], dim=-2)
        force_mag = force_mag * atmoic_mask
        return force_real, force_mag, atmoic_mask > 0.0

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        spin: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        coord_updated, atype_updated = self.preprocess_spin_input(coord, atype, spin)
        model_ret = self.forward_common(
            coord_updated,
            atype_updated,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]

        if self.do_grad("energy"):
            force_all = model_ret["energy_derv_r"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        else:
            force_all = model_ret["dforce"]
        (
            model_predict["force_real"],
            model_predict["force_mag"],
            model_predict["atmoic_mask"],
        ) = self.preprocess_spin_output(atype, force_all)
        return model_predict

    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        ## TODO preprocess
        raise NotImplementedError("Not implemented forward_lower for spin")
