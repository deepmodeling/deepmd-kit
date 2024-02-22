# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.pt.utils.utils import (
    dict_to_device,
)
from deepmd.utils.path import (
    DPPath,
)


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

    def process_spin_input(self, coord, atype, spin):
        """Generate virtual coordinates and types, concat into the input."""
        nframes, natom = coord.shape[:-1]
        atype_spin = torch.concat([atype, atype + self.spin.ntypes_real], dim=-1)
        virtual_scale_mask = self.spin.get_virtual_scale_mask()
        virtual_coord = coord + spin * virtual_scale_mask[atype].reshape(
            [nframes, natom, 1]
        )
        coord_spin = torch.concat([coord, virtual_coord], dim=-2)
        return coord_spin, atype_spin

    def process_spin_output(self, atype, force):
        """Split the output gradient of both real and virtual atoms, and scale the latter."""
        nframes, natom_double = force.shape[:2]
        natom = natom_double // 2
        virtual_scale_mask = self.spin.get_virtual_scale_mask()
        atmoic_mask = virtual_scale_mask[atype].reshape([nframes, natom, 1])
        force_real, force_mag = torch.split(force, [natom, natom], dim=1)
        force_mag = (force_mag.view([nframes, natom, -1]) * atmoic_mask).view(
            force_mag.shape
        )
        return force_real, force_mag, atmoic_mask > 0.0

    @staticmethod
    def extend_nlist(extended_atype, nlist):
        nframes, nloc, nnei = nlist.shape
        nall = extended_atype.shape[1]
        nlist_mask = nlist != -1
        nlist[nlist == -1] = 0
        nlist_shift = nlist + nall
        nlist[~nlist_mask] = -1
        nlist_shift[~nlist_mask] = -1
        self_spin = torch.arange(0, nloc, dtype=nlist.dtype, device=nlist.device) + nall
        self_spin = self_spin.view(1, -1, 1).expand(nframes, -1, -1)
        # self spin + real neighbor + virtual neighbor
        # nf x nloc x (1 + nnei + nnei)
        extended_nlist = torch.cat([self_spin, nlist, nlist_shift], dim=-1)
        # nf x (nloc + nloc) x (1 + nnei + nnei)
        extended_nlist = torch.cat(
            [extended_nlist, -1 * torch.ones_like(extended_nlist)], dim=-2
        )
        return extended_nlist

    @staticmethod
    def extend_mapping(mapping, nloc: int):
        return torch.cat([mapping, mapping + nloc], dim=-1)

    @staticmethod
    def switch_virtual_loc(extended_tensor, nloc: int):
        """
        Switch the virtual atoms of nloc ones from [nall: nall+nloc] to [nloc: nloc+nloc],
        to assure the atom types of first nloc * 2 atoms in nall * 2 to be right.
        """
        nframes, nall_double = extended_tensor.shape[:2]
        nall = nall_double // 2
        swithed_tensor = torch.zeros_like(extended_tensor)
        swithed_tensor[:, :nloc] = extended_tensor[:, :nloc]
        swithed_tensor[:, nloc : nloc + nloc] = extended_tensor[:, nall : nall + nloc]
        swithed_tensor[:, nloc + nloc : nloc + nall] = extended_tensor[:, nloc:nall]
        swithed_tensor[:, nloc + nall :] = extended_tensor[:, nloc + nall :]
        return swithed_tensor

    @staticmethod
    def switch_nlist(nlist_updated, nall: int):
        nframes, nloc_double = nlist_updated.shape[:2]
        nloc = nloc_double // 2
        first_part_index = (nloc <= nlist_updated) & (nlist_updated < nall)
        second_part_index = (nall <= nlist_updated) & (nlist_updated < (nall + nloc))
        nlist_updated[first_part_index] += nloc
        nlist_updated[second_part_index] -= nall - nloc
        return nlist_updated

    def extend_switch_input(
        self,
        extended_coord,
        extended_atype,
        extended_spin,
        nlist,
        mapping: Optional[torch.Tensor] = None,
    ):
        """
        Add `extended_spin` into `extended_coord` to generate virtual atoms, and extend `nlist` and `mapping`.
        Note that the final `extended_coord_updated` with shape [nframes, nall + nall, 3] has the following order:
        - [:, :nloc]: original nloc real atoms.
        - [:, nloc: nloc + nloc]: virtual atoms corresponding to nloc real atoms.
        - [:, nloc + nloc: nloc + nall]: ghost real atoms.
        - [:, nloc + nall: nall + nall]: virtual atoms corresponding to ghost real atoms.
        """
        nframes, nall = extended_coord.shape[:2]
        nloc = nlist.shape[1]
        # add spin but ignore the index switch
        extended_coord_updated, extended_atype_updated = self.process_spin_input(
            extended_coord, extended_atype, extended_spin
        )
        # extend the nlist and mapping but ignore the index switch
        nlist_updated = self.extend_nlist(extended_atype, nlist)
        mapping_updated = None
        if mapping is not None:
            mapping_updated = self.extend_mapping(mapping, nloc)
        # process the index switch
        extended_coord_updated = self.switch_virtual_loc(extended_coord_updated, nloc)
        extended_atype_updated = self.switch_virtual_loc(extended_atype_updated, nloc)
        mapping_updated = self.switch_virtual_loc(mapping_updated, nloc)
        nlist_updated = self.switch_nlist(nlist_updated, nall)
        return (
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping_updated,
        )

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

    def compute_or_load_stat(
        self,
        sampled,
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.
        When `sampled` is provided, all the statistics parameters will be calculated (or re-calculated for update),
        and saved in the `stat_file_path`(s).
        When `sampled` is not provided, it will check the existence of `stat_file_path`(s)
        and load the calculated statistics parameters.

        Parameters
        ----------
        sampled
            The sampled data frames from different data systems.
        stat_file_path
            The dictionary of paths to the statistics files.
        """
        spin_sampled = []
        for sys in sampled:
            dict_to_device(sys)
            coord_updated, atype_updated = self.process_spin_input(
                sys["coord"], sys["atype"], sys["spin"]
            )
            tmp_dict = {
                "coord": coord_updated,
                "atype": atype_updated,
            }
            if "natoms" in sys:
                natoms = sys["natoms"]
                tmp_dict["natoms"] = torch.cat(
                    [2 * natoms[:, :2], natoms[:, 2:], natoms[:, 2:]], dim=-1
                )
            for item_key in sys.keys():
                if item_key not in ["coord", "atype", "spin", "natoms"]:
                    tmp_dict[item_key] = sys[item_key]
            spin_sampled.append(tmp_dict)
        self.backbone_model.compute_or_load_stat(spin_sampled, stat_file_path)

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
        coord_updated, atype_updated = self.process_spin_input(coord, atype, spin)
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
            if self.do_grad_r(var_name):
                force_all = model_ret[f"{var_name}_derv_r"]
                (
                    model_ret[f"{var_name}_derv_r_real"],
                    model_ret[f"{var_name}_derv_r_mag"],
                    model_ret["atmoic_mask"],
                ) = self.process_spin_output(atype, force_all)
            else:
                force_all = model_ret["dforce"]
                (
                    model_ret["dforce_real"],
                    model_ret["dforce_mag"],
                    model_ret["atmoic_mask"],
                ) = self.process_spin_output(atype, force_all)
        return model_ret

    def forward_common_lower(
        self,
        extended_coord,
        extended_atype,
        extended_spin,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        (
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping_updated,
        ) = self.extend_switch_input(
            extended_coord, extended_atype, extended_spin, nlist, mapping=mapping
        )
        model_ret = self.backbone_model.forward_common_lower(
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping=mapping_updated,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.fitting_net is not None:
            var_name = self.fitting_net.var_name
            if self.do_grad_r(var_name):
                force_all = model_ret[f"{var_name}_derv_r"]
                (
                    model_ret[f"{var_name}_derv_r_real"],
                    model_ret[f"{var_name}_derv_r_mag"],
                    model_ret["atmoic_mask"],
                ) = self.process_spin_output(extended_atype, force_all)
            else:
                force_all = model_ret["dforce"]
                (
                    model_ret["dforce_real"],
                    model_ret["dforce_mag"],
                    model_ret["atmoic_mask"],
                ) = self.process_spin_output(extended_atype, force_all)
        return model_ret


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
        if self.do_grad_r("energy"):
            model_predict["force_real"] = model_ret["energy_derv_r_real"].squeeze(-2)
            model_predict["force_mag"] = model_ret["energy_derv_r_mag"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        else:
            assert model_ret["dforce_real"] is not None
            assert model_ret["dforce_mag"] is not None
            model_predict["force_real"] = model_ret["dforce_real"]
            model_predict["force_mag"] = model_ret["dforce_mag"]
        return model_predict

    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        extended_spin,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.fitting_net is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["extended_force_real"] = model_ret[
                    "energy_derv_r_real"
                ].squeeze(-2)
                model_predict["extended_force_mag"] = model_ret[
                    "energy_derv_r_mag"
                ].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-3)
            else:
                assert model_ret["dforce_real"] is not None
                assert model_ret["dforce_mag"] is not None
                model_predict["extended_force_real"] = model_ret["dforce_real"]
                model_predict["extended_force_mag"] = model_ret["dforce_mag"]
        else:
            model_predict = model_ret
        return model_predict
