# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import numpy as np

from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.make_model import (
    make_model,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.utils.spin import (
    Spin,
)


class SpinModel(NativeOP):
    """A spin model wrapper, with spin input preprocess and output split."""

    def __init__(
        self,
        backbone_model,
        spin: Spin,
    ) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.spin = spin
        self.ntypes_real = self.spin.ntypes_real
        self.virtual_scale_mask = self.spin.get_virtual_scale_mask()
        self.spin_mask = self.spin.get_spin_mask()

    def process_spin_input(self, coord, atype, spin):
        """Generate virtual coordinates and types, concat into the input."""
        nframes, nloc = coord.shape[:-1]
        atype_spin = np.concatenate([atype, atype + self.ntypes_real], axis=-1)
        virtual_coord = coord + spin * self.virtual_scale_mask[atype].reshape(
            [nframes, nloc, 1]
        )
        coord_spin = np.concatenate([coord, virtual_coord], axis=-2)
        return coord_spin, atype_spin

    def process_spin_input_lower(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        extended_spin: np.ndarray,
        nlist: np.ndarray,
        mapping: Optional[np.ndarray] = None,
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
        virtual_extended_coord = (
            extended_coord
            + extended_spin
            * self.virtual_scale_mask[extended_atype].reshape([nframes, nall, 1])
        )
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
        """Split the output both real and virtual atoms, and scale the latter."""
        nframes, nloc_double = out_tensor.shape[:2]
        nloc = nloc_double // 2
        if virtual_scale:
            virtual_scale_mask = self.virtual_scale_mask
        else:
            virtual_scale_mask = self.spin_mask
        atomic_mask = virtual_scale_mask[atype].reshape([nframes, nloc, 1])
        out_real, out_mag = np.split(out_tensor, [nloc], axis=1)
        if add_mag:
            out_real = out_real + out_mag
        out_mag = (out_mag.reshape([nframes, nloc, -1]) * atomic_mask).reshape(
            out_mag.shape
        )
        return out_real, out_mag, atomic_mask > 0.0

    def process_spin_output_lower(
        self,
        extended_atype,
        extended_out_tensor,
        nloc: int,
        add_mag: bool = True,
        virtual_scale: bool = True,
    ):
        """Split the extended output of both real and virtual atoms with switch, and scale the latter."""
        nframes, nall_double = extended_out_tensor.shape[:2]
        nall = nall_double // 2
        if virtual_scale:
            virtual_scale_mask = self.virtual_scale_mask
        else:
            virtual_scale_mask = self.spin_mask
        atomic_mask = virtual_scale_mask[extended_atype].reshape([nframes, nall, 1])
        extended_out_real = np.concatenate(
            [
                extended_out_tensor[:, :nloc],
                extended_out_tensor[:, nloc + nloc : nloc + nall],
            ],
            axis=1,
        )
        extended_out_mag = np.concatenate(
            [
                extended_out_tensor[:, nloc : nloc + nloc],
                extended_out_tensor[:, nloc + nall :],
            ],
            axis=1,
        )
        if add_mag:
            extended_out_real = extended_out_real + extended_out_mag
        extended_out_mag = (
            extended_out_mag.reshape([nframes, nall, -1]) * atomic_mask
        ).reshape(extended_out_mag.shape)
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
            np.arange(0, nloc, dtype=nlist.dtype)
            .reshape(1, -1, 1)
            .repeat(nframes, axis=0)
        )
        self_spin = self_real + nall
        # real atom's neighbors: self spin + real neighbor + virtual neighbor
        # nf x nloc x (1 + nnei + nnei)
        real_nlist = np.concatenate([self_spin, nlist, nlist_shift], axis=-1)
        # spin atom's neighbors: real + real neighbor + virtual neighbor
        # nf x nloc x (1 + nnei + nnei)
        spin_nlist = np.concatenate([self_real, nlist, nlist_shift], axis=-1)
        # nf x (nloc + nloc) x (1 + nnei + nnei)
        extended_nlist = np.concatenate([real_nlist, spin_nlist], axis=-2)
        # update the index for switch
        first_part_index = (nloc <= extended_nlist) & (extended_nlist < nall)
        second_part_index = (nall <= extended_nlist) & (extended_nlist < (nall + nloc))
        extended_nlist[first_part_index] += nloc
        extended_nlist[second_part_index] -= nall - nloc
        return extended_nlist

    @staticmethod
    def concat_switch_virtual(extended_tensor, extended_tensor_virtual, nloc: int):
        nframes, nall = extended_tensor.shape[:2]
        out_shape = list(extended_tensor.shape)
        out_shape[1] *= 2
        extended_tensor_updated = np.zeros(
            out_shape,
            dtype=extended_tensor.dtype,
        )
        extended_tensor_updated[:, :nloc] = extended_tensor[:, :nloc]
        extended_tensor_updated[:, nloc : nloc + nloc] = extended_tensor_virtual[
            :, :nloc
        ]
        extended_tensor_updated[:, nloc + nloc : nloc + nall] = extended_tensor[
            :, nloc:
        ]
        extended_tensor_updated[:, nloc + nall :] = extended_tensor_virtual[:, nloc:]
        return extended_tensor_updated.reshape(out_shape)

    @staticmethod
    def expand_aparam(aparam, nloc: int):
        """Expand the atom parameters for virtual atoms if necessary."""
        nframes, natom, numb_aparam = aparam.shape
        if natom == nloc:  # good
            pass
        elif natom < nloc:  # for spin with virtual atoms
            aparam = np.concatenate(
                [
                    aparam,
                    np.zeros(
                        [nframes, nloc - natom, numb_aparam],
                        dtype=aparam.dtype,
                    ),
                ],
                axis=1,
            )
        else:
            raise ValueError(
                f"get an input aparam with {aparam.shape[1]} inputs, ",
                f"which is larger than {nloc} atoms.",
            )
        return aparam

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        tmap = self.backbone_model.get_type_map()
        ntypes = len(tmap) // 2  # ignore the virtual type
        return tmap[:ntypes]

    def get_ntypes(self):
        """Returns the number of element types."""
        return len(self.get_type_map())

    def get_rcut(self):
        """Get the cut-off radius."""
        return self.backbone_model.get_rcut()

    def get_dim_fparam(self):
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.backbone_model.get_dim_fparam()

    def get_dim_aparam(self):
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.backbone_model.get_dim_aparam()

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.
        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.backbone_model.get_sel_type()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).
        If False, the shape is (nframes, nloc, ndim).
        """
        return self.backbone_model.is_aparam_nall()

    def model_output_type(self) -> list[str]:
        """Get the output type for the model."""
        return self.backbone_model.model_output_type()

    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        return self.backbone_model.get_model_def_script()

    def get_min_nbor_dist(self) -> Optional[float]:
        """Get the minimum neighbor distance."""
        return self.backbone_model.get_min_nbor_dist()

    def get_nnei(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        # for C++ interface
        if not self.backbone_model.mixed_types():
            return self.backbone_model.get_nnei() // 2  # ignore the virtual selected
        else:
            return self.backbone_model.get_nnei()

    def get_nsel(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        if not self.backbone_model.mixed_types():
            return self.backbone_model.get_nsel() // 2  # ignore the virtual selected
        else:
            return self.backbone_model.get_nsel()

    @staticmethod
    def has_spin() -> bool:
        """Returns whether it has spin input and output."""
        return True

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
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.backbone_model, name)

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

    def call(
        self,
        coord,
        atype,
        spin,
        box: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return model prediction.

        Parameters
        ----------
        coord
            The coordinates of the atoms.
            shape: nf x (nloc x 3)
        atype
            The type of atoms. shape: nf x nloc
        spin
            The spins of the atoms.
            shape: nf x (nloc x 3)
        box
            The simulation box. shape: nf x 9
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda
        do_atomic_virial
            If calculate the atomic virial.

        Returns
        -------
        ret_dict
            The result dict of type dict[str,np.ndarray].
            The keys are defined by the `ModelOutputDef`.

        """
        nframes, nloc = atype.shape[:2]
        coord = coord.reshape(nframes, nloc, 3)
        spin = spin.reshape(nframes, nloc, 3)
        coord_updated, atype_updated = self.process_spin_input(coord, atype, spin)
        if aparam is not None:
            aparam = self.expand_aparam(aparam, nloc * 2)
        model_predict = self.backbone_model.call(
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
        model_predict[f"{var_name}"] = np.split(
            model_predict[f"{var_name}"], [nloc], axis=1
        )[0]
        # for now omit the grad output
        return model_predict

    def call_lower(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        extended_spin: np.ndarray,
        nlist: np.ndarray,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        do_atomic_virial: bool = False,
    ):
        """Return model prediction. Lower interface that takes
        extended atomic coordinates, types and spins, nlist, and mapping
        as input, and returns the predictions on the extended region.
        The predictions are not reduced.

        Parameters
        ----------
        extended_coord
            coordinates in extended region. nf x (nall x 3).
        extended_atype
            atomic type in extended region. nf x nall.
        extended_spin
            spins in extended region. nf x (nall x 3).
        nlist
            neighbor list. nf x nloc x nsel.
        mapping
            maps the extended indices to local indices. nf x nall.
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda
        do_atomic_virial
            whether calculate atomic virial

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
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
        model_predict = self.backbone_model.call_lower(
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
        model_predict[f"{var_name}"] = np.split(
            model_predict[f"{var_name}"], [nloc], axis=1
        )[0]
        # for now omit the grad output
        return model_predict

    forward_lower = call_lower
