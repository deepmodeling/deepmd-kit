# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch


class SpinModel(torch.nn.Module):
    """A spin model wrapper, with spin input preprocess and output split."""

    __USE_SPIN_INPUT__ = True

    def __init__(
        self,
        backbone_model,
        spin,
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.spin = spin

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
        nframes, nloc_double = force.shape[:2]
        nloc = nloc_double // 2
        virtual_scale_mask = self.spin.get_virtual_scale_mask()
        atmoic_mask = torch.gather(
            virtual_scale_mask, -1, index=atype.view(-1)
        ).reshape([nframes, nloc, 1])
        force_real, force_mag = torch.split(force, [nloc, nloc], dim=1)
        force_mag = (force_mag.view([nframes, nloc, -1]) * atmoic_mask).view(
            force_mag.shape
        )
        return force_real, force_mag, atmoic_mask > 0.0

    def __getattr__(self, name):
        """Get attribute from the wrapped model."""
        if (
            name == "backbone_model"
        ):  # torch.nn.Module will exclude modules to self.__dict__["_modules"]
            return self.__dict__["_modules"]["backbone_model"]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.backbone_model, name)

    def forward_common(
        self,
        coord,
        atype,
        spin,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        coord_updated, atype_updated = self.preprocess_spin_input(coord, atype, spin)
        model_ret = self.backbone_model.forward_common(
            coord_updated,
            atype_updated,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.fitting_net is not None:
            var_name = self.fitting_net.var_name
            if self.do_grad(var_name):
                force_all = model_ret[f"{var_name}_derv_r"]
                (
                    model_ret[f"{var_name}_derv_r_real"],
                    model_ret[f"{var_name}_derv_r_mag"],
                    model_ret["atmoic_mask"],
                ) = self.preprocess_spin_output(atype, force_all)
            else:
                force_all = model_ret["dforce"]
                (
                    model_ret["dforce_real"],
                    model_ret["dforce_mag"],
                    model_ret["atmoic_mask"],
                ) = self.preprocess_spin_output(atype, force_all)
        return model_ret

    def forward_common_lower(
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
        raise NotImplementedError("Not implemented forward_common_lower for spin")


class SpinEnergyModel(SpinModel):
    """A spin model for energy."""

    model_type = "ener"

    def __init__(
        self,
        backbone_model,
        spin,
    ):
        super().__init__(backbone_model, spin)

    def forward(
        self,
        coord,
        atype,
        spin,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            spin,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["atmoic_mask"] = model_ret["atmoic_mask"]
        if self.do_grad("energy"):
            model_predict["force_real"] = model_ret["energy_derv_r_real"].squeeze(-2)
            model_predict["force_mag"] = model_ret["energy_derv_r_mag"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        else:
            model_predict["force_real"] = model_ret["dforce_real"]
            model_predict["force_mag"] = model_ret["dforce_mag"]
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
