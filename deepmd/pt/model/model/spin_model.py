# SPDX-License-Identifier: LGPL-3.0-or-later
import functools
from copy import (
    deepcopy,
)
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt.utils.utils import (
    to_torch_tensor,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.spin import (
    Spin,
)

from .make_model import (
    make_model,
)


class SpinModel(torch.nn.Module):
    """A spin model wrapper, with spin input preprocess and output split."""

    def __init__(
        self,
        backbone_model,
        spin: Spin,
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.spin = spin
        self.ntypes_real = self.spin.ntypes_real
        self.virtual_scale_mask = to_torch_tensor(self.spin.get_virtual_scale_mask())
        self.spin_mask = to_torch_tensor(self.spin.get_spin_mask())

    def process_spin_input(self, coord, atype, spin):
        """Generate virtual coordinates and types, concat into the input."""
        nframes, nloc = atype.shape
        coord = coord.reshape(nframes, nloc, 3)
        spin = spin.reshape(nframes, nloc, 3)
        atype_spin = torch.concat([atype, atype + self.ntypes_real], dim=-1)
        virtual_coord = coord + spin * (self.virtual_scale_mask.to(atype.device))[
            atype
        ].reshape([nframes, nloc, 1])
        coord_spin = torch.concat([coord, virtual_coord], dim=-2)
        return coord_spin, atype_spin

    def process_spin_input_lower(
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
        virtual_extended_coord = extended_coord + extended_spin * (
            self.virtual_scale_mask.to(extended_atype.device)
        )[extended_atype].reshape([nframes, nall, 1])
        virtual_extended_atype = extended_atype + self.ntypes_real
        extended_coord_updated = self.concat_switch_virtual(
            extended_coord, virtual_extended_coord, nloc
        )
        extended_atype_updated = self.concat_switch_virtual(
            extended_atype, virtual_extended_atype, nloc
        )
        if mapping is not None:
            virtual_mapping = mapping + nloc
            mapping_updated = self.concat_switch_virtual(mapping, virtual_mapping, nloc)
        else:
            mapping_updated = None
        # extend the nlist
        nlist_updated = self.extend_nlist(extended_atype, nlist)
        return (
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping_updated,
        )

    def process_spin_output(
        self, atype, out_tensor, add_mag: bool = True, virtual_scale: bool = True
    ):
        """
        Split the output both real and virtual atoms, and scale the latter.
        add_mag: whether to add magnetic tensor onto the real tensor.
            Default: True. e.g. Ture for forces and False for atomic virials on real atoms.
        virtual_scale: whether to scale the magnetic tensor with virtual scale factor.
            Default: True. e.g. Ture for forces and False for atomic virials on virtual atoms.
        """
        nframes, nloc_double = out_tensor.shape[:2]
        nloc = nloc_double // 2
        if virtual_scale:
            virtual_scale_mask = self.virtual_scale_mask.to(atype.device)
        else:
            virtual_scale_mask = self.spin_mask.to(atype.device)
        atomic_mask = virtual_scale_mask[atype].reshape([nframes, nloc, 1])
        out_real, out_mag = torch.split(out_tensor, [nloc, nloc], dim=1)
        if add_mag:
            out_real = out_real + out_mag
        out_mag = (out_mag.view([nframes, nloc, -1]) * atomic_mask).view(out_mag.shape)
        return out_real, out_mag, atomic_mask > 0.0

    def process_spin_output_lower(
        self,
        extended_atype,
        extended_out_tensor,
        nloc: int,
        add_mag: bool = True,
        virtual_scale: bool = True,
    ):
        """
        Split the extended output of both real and virtual atoms with switch, and scale the latter.
        add_mag: whether to add magnetic tensor onto the real tensor.
            Default: True. e.g. Ture for forces and False for atomic virials on real atoms.
        virtual_scale: whether to scale the magnetic tensor with virtual scale factor.
            Default: True. e.g. Ture for forces and False for atomic virials on virtual atoms.
        """
        nframes, nall_double = extended_out_tensor.shape[:2]
        nall = nall_double // 2
        if virtual_scale:
            virtual_scale_mask = self.virtual_scale_mask.to(extended_atype.device)
        else:
            virtual_scale_mask = self.spin_mask.to(extended_atype.device)
        atomic_mask = virtual_scale_mask[extended_atype].reshape([nframes, nall, 1])
        extended_out_real = torch.cat(
            [
                extended_out_tensor[:, :nloc],
                extended_out_tensor[:, nloc + nloc : nloc + nall],
            ],
            dim=1,
        )
        extended_out_mag = torch.cat(
            [
                extended_out_tensor[:, nloc : nloc + nloc],
                extended_out_tensor[:, nloc + nall :],
            ],
            dim=1,
        )
        if add_mag:
            extended_out_real = extended_out_real + extended_out_mag
        extended_out_mag = (
            extended_out_mag.view([nframes, nall, -1]) * atomic_mask
        ).view(extended_out_mag.shape)
        return extended_out_real, extended_out_mag, atomic_mask > 0.0

    @staticmethod
    def extend_nlist(extended_atype, nlist):
        nframes, nloc, nnei = nlist.shape
        nall = extended_atype.shape[1]
        nlist_mask = nlist != -1
        nlist[nlist == -1] = 0
        nlist_shift = nlist + nall
        nlist[~nlist_mask] = -1
        nlist_shift[~nlist_mask] = -1
        self_real = (
            torch.arange(0, nloc, dtype=nlist.dtype, device=nlist.device)
            .view(1, -1, 1)
            .expand(nframes, -1, -1)
        )
        self_spin = self_real + nall
        # real atom's neighbors: self spin + real neighbor + virtual neighbor
        # nf x nloc x (1 + nnei + nnei)
        real_nlist = torch.cat([self_spin, nlist, nlist_shift], dim=-1)
        # spin atom's neighbors: real + real neighbor + virtual neighbor
        # nf x nloc x (1 + nnei + nnei)
        spin_nlist = torch.cat([self_real, nlist, nlist_shift], dim=-1)
        # nf x (nloc + nloc) x (1 + nnei + nnei)
        extended_nlist = torch.cat([real_nlist, spin_nlist], dim=-2)
        # update the index for switch
        first_part_index = (nloc <= extended_nlist) & (extended_nlist < nall)
        second_part_index = (nall <= extended_nlist) & (extended_nlist < (nall + nloc))
        extended_nlist[first_part_index] += nloc
        extended_nlist[second_part_index] -= nall - nloc
        return extended_nlist

    @staticmethod
    def concat_switch_virtual(extended_tensor, extended_tensor_virtual, nloc: int):
        """
        Concat real and virtual extended tensors, and switch all the local ones to the first nloc * 2 atoms.
        - [:, :nloc]: original nloc real atoms.
        - [:, nloc: nloc + nloc]: virtual atoms corresponding to nloc real atoms.
        - [:, nloc + nloc: nloc + nall]: ghost real atoms.
        - [:, nloc + nall: nall + nall]: virtual atoms corresponding to ghost real atoms.
        """
        nframes, nall = extended_tensor.shape[:2]
        out_shape = list(extended_tensor.shape)
        out_shape[1] *= 2
        extended_tensor_updated = torch.zeros(
            out_shape,
            dtype=extended_tensor.dtype,
            device=extended_tensor.device,
        )
        extended_tensor_updated[:, :nloc] = extended_tensor[:, :nloc]
        extended_tensor_updated[:, nloc : nloc + nloc] = extended_tensor_virtual[
            :, :nloc
        ]
        extended_tensor_updated[:, nloc + nloc : nloc + nall] = extended_tensor[
            :, nloc:
        ]
        extended_tensor_updated[:, nloc + nall :] = extended_tensor_virtual[:, nloc:]
        return extended_tensor_updated.view(out_shape)

    @staticmethod
    def expand_aparam(aparam, nloc: int):
        """Expand the atom parameters for virtual atoms if necessary."""
        nframes, natom, numb_aparam = aparam.shape
        if natom == nloc:  # good
            pass
        elif natom < nloc:  # for spin with virtual atoms
            aparam = torch.concat(
                [
                    aparam,
                    torch.zeros(
                        [nframes, nloc - natom, numb_aparam],
                        device=aparam.device,
                        dtype=aparam.dtype,
                    ),
                ],
                dim=1,
            )
        else:
            raise ValueError(
                f"get an input aparam with {aparam.shape[1]} inputs, ",
                f"which is larger than {nloc} atoms.",
            )
        return aparam

    @torch.jit.export
    def get_type_map(self) -> List[str]:
        """Get the type map."""
        tmap = self.backbone_model.get_type_map()
        ntypes = len(tmap) // 2  # ignore the virtual type
        return tmap[:ntypes]

    @torch.jit.export
    def get_ntypes(self):
        """Returns the number of element types."""
        return len(self.get_type_map())

    @torch.jit.export
    def get_rcut(self):
        """Get the cut-off radius."""
        return self.backbone_model.get_rcut()

    @torch.jit.export
    def get_dim_fparam(self):
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.backbone_model.get_dim_fparam()

    @torch.jit.export
    def get_dim_aparam(self):
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.backbone_model.get_dim_aparam()

    @torch.jit.export
    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.
        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.backbone_model.get_sel_type()

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).
        If False, the shape is (nframes, nloc, ndim).
        """
        return self.backbone_model.is_aparam_nall()

    @torch.jit.export
    def model_output_type(self) -> List[str]:
        """Get the output type for the model."""
        return self.backbone_model.model_output_type()

    @torch.jit.export
    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        return self.backbone_model.get_model_def_script()

    @torch.jit.export
    def get_min_nbor_dist(self) -> Optional[float]:
        """Get the minimum neighbor distance."""
        return self.backbone_model.get_min_nbor_dist()

    @torch.jit.export
    def get_nnei(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        # for C++ interface
        if not self.backbone_model.mixed_types():
            return self.backbone_model.get_nnei() // 2  # ignore the virtual selected
        else:
            return self.backbone_model.get_nnei()

    @torch.jit.export
    def get_nsel(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        if not self.backbone_model.mixed_types():
            return self.backbone_model.get_nsel() // 2  # ignore the virtual selected
        else:
            return self.backbone_model.get_nsel()

    @torch.jit.export
    def has_spin(self) -> bool:
        """Returns whether it has spin input and output."""
        return True

    @torch.jit.export
    def has_message_passing(self) -> bool:
        """Returns whether the model has message passing."""
        return self.backbone_model.has_message_passing()

    def model_output_def(self):
        """Get the output def for the model."""
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        backbone_model_atomic_output_def = self.backbone_model.atomic_output_def()
        backbone_model_atomic_output_def[var_name].magnetic = True
        return ModelOutputDef(backbone_model_atomic_output_def)

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
        sampled_func,
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
        sampled_func
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The dictionary of paths to the statistics files.
        """

        @functools.lru_cache
        def spin_sampled_func():
            sampled = sampled_func()
            spin_sampled = []
            for sys in sampled:
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
            return spin_sampled

        self.backbone_model.compute_or_load_stat(spin_sampled_func, stat_file_path)

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
        nframes, nloc = atype.shape
        coord_updated, atype_updated = self.process_spin_input(coord, atype, spin)
        if aparam is not None:
            aparam = self.expand_aparam(aparam, nloc * 2)
        model_ret = self.backbone_model.forward_common(
            coord_updated,
            atype_updated,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        model_ret[f"{var_name}"] = torch.split(
            model_ret[f"{var_name}"], [nloc, nloc], dim=1
        )[0]
        if self.backbone_model.do_grad_r(var_name):
            (
                model_ret[f"{var_name}_derv_r"],
                model_ret[f"{var_name}_derv_r_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output(atype, model_ret[f"{var_name}_derv_r"])
        if self.backbone_model.do_grad_c(var_name) and do_atomic_virial:
            (
                model_ret[f"{var_name}_derv_c"],
                model_ret[f"{var_name}_derv_c_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output(
                atype,
                model_ret[f"{var_name}_derv_c"],
                add_mag=False,
                virtual_scale=False,
            )
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
        nframes, nloc = nlist.shape[:2]
        (
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping_updated,
        ) = self.process_spin_input_lower(
            extended_coord, extended_atype, extended_spin, nlist, mapping=mapping
        )
        if aparam is not None:
            aparam = self.expand_aparam(aparam, nloc * 2)
        model_ret = self.backbone_model.forward_common_lower(
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping=mapping_updated,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        model_ret[f"{var_name}"] = torch.split(
            model_ret[f"{var_name}"], [nloc, nloc], dim=1
        )[0]
        if self.backbone_model.do_grad_r(var_name):
            (
                model_ret[f"{var_name}_derv_r"],
                model_ret[f"{var_name}_derv_r_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output_lower(
                extended_atype, model_ret[f"{var_name}_derv_r"], nloc
            )
        if self.backbone_model.do_grad_c(var_name) and do_atomic_virial:
            (
                model_ret[f"{var_name}_derv_c"],
                model_ret[f"{var_name}_derv_c_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output_lower(
                extended_atype,
                model_ret[f"{var_name}_derv_c"],
                nloc,
                add_mag=False,
                virtual_scale=False,
            )
        return model_ret

    def serialize(self) -> dict:
        return {
            "backbone_model": self.backbone_model.serialize(),
            "spin": self.spin.serialize(),
        }

    @classmethod
    def deserialize(cls, data) -> "SpinModel":
        backbone_model_obj = make_model(DPAtomicModel).deserialize(
            data["backbone_model"]
        )
        spin = Spin.deserialize(data["spin"])
        return cls(
            backbone_model=backbone_model_obj,
            spin=spin,
        )


class SpinEnergyModel(SpinModel):
    """A spin model for energy."""

    model_type = "ener"

    def __init__(
        self,
        backbone_model,
        spin: Spin,
    ):
        super().__init__(backbone_model, spin)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": deepcopy(out_def_data["energy"]),
            "energy": deepcopy(out_def_data["energy_redu"]),
            "mask_mag": deepcopy(out_def_data["mask_mag"]),
        }
        if self.do_grad_r("energy"):
            output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
            output_def["force"].squeeze(-2)
            output_def["force_mag"] = deepcopy(out_def_data["energy_derv_r_mag"])
            output_def["force_mag"].squeeze(-2)
        return output_def

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
        model_predict["mask_mag"] = model_ret["mask_mag"]
        if self.backbone_model.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            model_predict["force_mag"] = model_ret["energy_derv_r_mag"].squeeze(-2)
        # not support virial by far
        return model_predict

    @torch.jit.export
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
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["extended_mask_mag"] = model_ret["mask_mag"]
        if self.backbone_model.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            model_predict["extended_force_mag"] = model_ret[
                "energy_derv_r_mag"
            ].squeeze(-2)
        # not support virial by far
        return model_predict
