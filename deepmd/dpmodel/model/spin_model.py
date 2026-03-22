# SPDX-License-Identifier: LGPL-3.0-or-later
import functools
from collections.abc import (
    Callable,
)
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
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
    r"""A spin model wrapper, with spin input preprocess and output split.

    This model extends a backbone DP model to handle magnetic spin degrees of freedom.
    Virtual atoms are created at positions offset from real atoms by their spin vectors:

    .. math::
        \mathbf{r}_i^{\mathrm{virtual}} = \mathbf{r}_i^{\mathrm{real}} + s_i \cdot \boldsymbol{\sigma}_i,

    where :math:`s_i` is a scaling factor and :math:`\boldsymbol{\sigma}_i` is the spin vector.

    The model then computes interactions between real atoms, virtual atoms, and between
    real and virtual atoms, enabling the prediction of spin-dependent properties.

    The output forces on virtual atoms are converted to magnetic torques:

    .. math::
        \boldsymbol{\tau}_i = \mathbf{F}_i^{\mathrm{virtual}} \times \boldsymbol{\sigma}_i.
    """

    def __init__(
        self,
        backbone_model: DPAtomicModel,
        spin: Spin,
    ) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.spin = spin
        self.ntypes_real = self.spin.ntypes_real
        self.virtual_scale_mask = self.spin.get_virtual_scale_mask()
        self.spin_mask = self.spin.get_spin_mask()

    def _to_xp(self, arr: Any, xp: Any, ref_arr: Any) -> Any:
        """Convert a numpy array to the same namespace as ref_arr."""
        return xp.asarray(arr, device=array_api_compat.device(ref_arr))

    def process_spin_input(
        self, coord: Array, atype: Array, spin: Array
    ) -> tuple[Array, Array, Array]:
        """Generate virtual coordinates and types, concat into the input.

        Returns
        -------
        coord_spin : Array
            Concatenated coordinates with shape (nframes, 2*nloc, 3).
        atype_spin : Array
            Concatenated atom types with shape (nframes, 2*nloc).
        coord_corr : Array
            Coordinate correction for virial with shape (nframes, 2*nloc, 3).
        """
        xp = array_api_compat.array_namespace(coord)
        nframes, nloc = coord.shape[:-1]
        atype_spin = xp.concat([atype, atype + self.ntypes_real], axis=-1)
        vsm = self._to_xp(self.virtual_scale_mask, xp, coord)
        spin_dist = spin * xp.reshape(vsm[atype], (nframes, nloc, 1))
        virtual_coord = coord + spin_dist
        coord_spin = xp.concat([coord, virtual_coord], axis=-2)
        # for spin virial correction
        coord_corr = xp.concat(
            [
                xp.zeros(
                    coord.shape,
                    dtype=coord.dtype,
                    device=array_api_compat.device(coord),
                ),
                -spin_dist,
            ],
            axis=-2,
        )
        return coord_spin, atype_spin, coord_corr

    def process_spin_input_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        extended_spin: Array,
        nlist: Array,
        mapping: Array | None = None,
    ) -> tuple[Array, Array, Array, Array | None, Array]:
        """
        Add `extended_spin` into `extended_coord` to generate virtual atoms, and extend `nlist` and `mapping`.

        Returns
        -------
        extended_coord_updated : Array
            Updated coordinates with virtual atoms, shape (nframes, 2*nall, 3).
        extended_atype_updated : Array
            Updated atom types with virtual atoms, shape (nframes, 2*nall).
        nlist_updated : Array
            Updated neighbor list including virtual atoms.
        mapping_updated : Array or None
            Updated mapping indices, or None if input mapping is None.
        extended_coord_corr : Array
            Coordinate correction for virial with shape (nframes, 2*nall, 3).

        Notes
        -----
        The final `extended_coord_updated` with shape [nframes, nall + nall, 3] has the following order:
        - [:, :nloc]: original nloc real atoms.
        - [:, nloc: nloc + nloc]: virtual atoms corresponding to nloc real atoms.
        - [:, nloc + nloc: nloc + nall]: ghost real atoms.
        - [:, nloc + nall: nall + nall]: virtual atoms corresponding to ghost real atoms.
        """
        xp = array_api_compat.array_namespace(extended_coord)
        nframes, nall = extended_coord.shape[:2]
        nloc = nlist.shape[1]
        vsm = self._to_xp(self.virtual_scale_mask, xp, extended_coord)
        extended_spin_dist = extended_spin * xp.reshape(
            vsm[extended_atype], (nframes, nall, 1)
        )
        virtual_extended_coord = extended_coord + extended_spin_dist
        virtual_extended_atype = extended_atype + self.ntypes_real
        extended_coord_updated = self.concat_switch_virtual(
            extended_coord, virtual_extended_coord, nloc
        )
        # for spin virial correction
        extended_coord_corr = self.concat_switch_virtual(
            xp.zeros(
                extended_coord.shape,
                dtype=extended_coord.dtype,
                device=array_api_compat.device(extended_coord),
            ),
            -extended_spin_dist,
            nloc,
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
            extended_coord_corr,
        )

    def process_spin_output(
        self,
        atype: Array,
        out_tensor: Array,
        add_mag: bool = True,
        virtual_scale: bool = True,
    ) -> tuple[Array, Array]:
        """Split the output both real and virtual atoms, and scale the latter."""
        xp = array_api_compat.array_namespace(out_tensor)
        nframes, nloc_double = out_tensor.shape[:2]
        nloc = nloc_double // 2
        if virtual_scale:
            mask = self._to_xp(self.virtual_scale_mask, xp, out_tensor)
        else:
            mask = self._to_xp(self.spin_mask, xp, out_tensor)
        atomic_mask = xp.reshape(mask[atype], (nframes, nloc, 1))
        out_real, out_mag = out_tensor[:, :nloc], out_tensor[:, nloc:]
        if add_mag:
            out_real = out_real + out_mag
        out_mag = xp.reshape(
            xp.reshape(out_mag, (nframes, nloc, -1)) * atomic_mask,
            out_mag.shape,
        )
        return out_real, out_mag, atomic_mask > 0.0

    def process_spin_output_lower(
        self,
        extended_atype: Array,
        extended_out_tensor: Array,
        nloc: int,
        add_mag: bool = True,
        virtual_scale: bool = True,
    ) -> tuple[Array, Array]:
        """Split the extended output of both real and virtual atoms with switch, and scale the latter."""
        xp = array_api_compat.array_namespace(extended_out_tensor)
        nframes, nall_double = extended_out_tensor.shape[:2]
        nall = nall_double // 2
        if virtual_scale:
            mask = self._to_xp(self.virtual_scale_mask, xp, extended_out_tensor)
        else:
            mask = self._to_xp(self.spin_mask, xp, extended_out_tensor)
        atomic_mask = xp.reshape(mask[extended_atype], (nframes, nall, 1))
        extended_out_real = xp.concat(
            [
                extended_out_tensor[:, :nloc],
                extended_out_tensor[:, nloc + nloc : nloc + nall],
            ],
            axis=1,
        )
        extended_out_mag = xp.concat(
            [
                extended_out_tensor[:, nloc : nloc + nloc],
                extended_out_tensor[:, nloc + nall :],
            ],
            axis=1,
        )
        if add_mag:
            extended_out_real = extended_out_real + extended_out_mag
        extended_out_mag = xp.reshape(
            xp.reshape(extended_out_mag, (nframes, nall, -1)) * atomic_mask,
            extended_out_mag.shape,
        )
        return extended_out_real, extended_out_mag, atomic_mask > 0.0

    @staticmethod
    def extend_nlist(extended_atype: Array, nlist: Array) -> Array:
        xp = array_api_compat.array_namespace(nlist)
        nframes, nloc, _nnei = nlist.shape
        nall = extended_atype.shape[1]
        nlist_mask = nlist != -1
        # Use xp.where instead of in-place boolean indexing
        nlist_safe = xp.where(nlist_mask, nlist, xp.zeros_like(nlist))
        nlist_shift = xp.where(nlist_mask, nlist_safe + nall, -1 * xp.ones_like(nlist))
        # Restore nlist with -1 for masked entries (non-mutating)
        nlist = xp.where(nlist_mask, nlist, -1 * xp.ones_like(nlist))
        self_real = xp.reshape(
            xp.arange(
                0, nloc, dtype=nlist.dtype, device=array_api_compat.device(nlist)
            ),
            (1, nloc, 1),
        ) * xp.ones(
            (nframes, 1, 1), dtype=nlist.dtype, device=array_api_compat.device(nlist)
        )
        self_spin = self_real + nall
        # real atom's neighbors: self spin + real neighbor + virtual neighbor
        # nf x nloc x (1 + nnei + nnei)
        real_nlist = xp.concat([self_spin, nlist, nlist_shift], axis=-1)
        # spin atom's neighbors: real + real neighbor + virtual neighbor
        # nf x nloc x (1 + nnei + nnei)
        spin_nlist = xp.concat([self_real, nlist, nlist_shift], axis=-1)
        # nf x (nloc + nloc) x (1 + nnei + nnei)
        extended_nlist = xp.concat([real_nlist, spin_nlist], axis=-2)
        # update the index for switch using xp.where instead of in-place ops
        first_part_mask = (nloc <= extended_nlist) & (extended_nlist < nall)
        second_part_mask = (nall <= extended_nlist) & (extended_nlist < (nall + nloc))
        extended_nlist = xp.where(
            first_part_mask, extended_nlist + nloc, extended_nlist
        )
        extended_nlist = xp.where(
            second_part_mask, extended_nlist - (nall - nloc), extended_nlist
        )
        return extended_nlist

    @staticmethod
    def concat_switch_virtual(
        extended_tensor: Array, extended_tensor_virtual: Array, nloc: int
    ) -> Array:
        xp = array_api_compat.array_namespace(extended_tensor)
        return xp.concat(
            [
                extended_tensor[:, :nloc],
                extended_tensor_virtual[:, :nloc],
                extended_tensor[:, nloc:],
                extended_tensor_virtual[:, nloc:],
            ],
            axis=1,
        )

    @staticmethod
    def expand_aparam(aparam: Array, nloc: int) -> Array:
        """Expand the atom parameters for virtual atoms if necessary."""
        xp = array_api_compat.array_namespace(aparam)
        nframes, natom, numb_aparam = aparam.shape
        if natom == nloc:  # good
            pass
        elif natom < nloc:  # for spin with virtual atoms
            aparam = xp.concat(
                [
                    aparam,
                    xp.zeros(
                        [nframes, nloc - natom, numb_aparam],
                        dtype=aparam.dtype,
                        device=array_api_compat.device(aparam),
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

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], list[dict[str, Any]]],
        stat_file_path: Any | None = None,
        preset_observed_type: list[str] | None = None,
    ) -> None:
        """Compute or load the statistics parameters of the model.

        Parameters
        ----------
        sampled_func
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The dictionary of paths to the statistics files.
        preset_observed_type
            The preset observed types.
        """

        @functools.lru_cache
        def spin_sampled_func() -> list[dict[str, Any]]:
            sampled = sampled_func()
            spin_sampled = []
            for sys in sampled:
                coord_updated, atype_updated, _ = self.process_spin_input(
                    sys["coord"], sys["atype"], sys["spin"]
                )
                tmp_dict = {
                    "coord": coord_updated,
                    "atype": atype_updated,
                }
                if "aparam" in sys:
                    tmp_dict["aparam"] = self.expand_aparam(
                        sys["aparam"], atype_updated.shape[1]
                    )
                if "natoms" in sys:
                    natoms = sys["natoms"]
                    xp = array_api_compat.array_namespace(natoms)
                    tmp_dict["natoms"] = xp.concat(
                        [2 * natoms[:, :2], natoms[:, 2:], natoms[:, 2:]], axis=-1
                    )
                for item_key in sys:
                    if item_key not in [
                        "coord",
                        "atype",
                        "spin",
                        "natoms",
                        "aparam",
                    ]:
                        tmp_dict[item_key] = sys[item_key]
                spin_sampled.append(tmp_dict)
            return spin_sampled

        self.backbone_model.compute_or_load_stat(
            spin_sampled_func,
            stat_file_path,
            preset_observed_type=preset_observed_type,
        )

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        tmap = self.backbone_model.get_type_map()
        ntypes = len(tmap) // 2  # ignore the virtual type
        return tmap[:ntypes]

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return len(self.get_type_map())

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.backbone_model.get_rcut()

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.backbone_model.get_dim_fparam()

    def get_dim_aparam(self) -> int:
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

    def get_min_nbor_dist(self) -> float | None:
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

    def model_output_def(self) -> ModelOutputDef:
        """Get the output def for the model."""
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        backbone_model_atomic_output_def = self.backbone_model.atomic_output_def()
        backbone_model_atomic_output_def[var_name].magnetic = True
        return ModelOutputDef(backbone_model_atomic_output_def)

    def _get_spin_sampled_func(
        self, sampled_func: Callable[[], list[dict]]
    ) -> Callable[[], list[dict]]:
        """Get a spin-aware sampled function that transforms spin data for the backbone model.

        Parameters
        ----------
        sampled_func
            A callable that returns a list of data dicts containing 'coord', 'atype', 'spin', etc.

        Returns
        -------
        Callable
            A cached callable that returns spin-preprocessed data dicts.
        """

        @functools.lru_cache
        def spin_sampled_func() -> list[dict]:
            sampled = sampled_func()
            spin_sampled = []
            for sys in sampled:
                coord_updated, atype_updated, _ = self.process_spin_input(
                    sys["coord"], sys["atype"], sys["spin"]
                )
                tmp_dict = {
                    "coord": coord_updated,
                    "atype": atype_updated,
                }
                if "natoms" in sys:
                    natoms = sys["natoms"]
                    xp = array_api_compat.array_namespace(natoms)
                    tmp_dict["natoms"] = xp.concat(
                        [2 * natoms[:, :2], natoms[:, 2:], natoms[:, 2:]], axis=-1
                    )
                for item_key in sys.keys():
                    if item_key not in ["coord", "atype", "spin", "natoms"]:
                        tmp_dict[item_key] = sys[item_key]
                spin_sampled.append(tmp_dict)
            return spin_sampled

        return self.backbone_model.atomic_model._make_wrapped_sampler(spin_sampled_func)

    def change_out_bias(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        """Change the output bias of atomic model according to the input data and the pretrained model.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `np.ndarray`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        bias_adjust_mode : str
            The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
            'change-by-statistic' : perform predictions on labels of target dataset,
                    and do least square on the errors to obtain the target shift as bias.
            'set-by-statistic' : directly use the statistic output bias in the target dataset.
        """
        spin_sampled_func = self._get_spin_sampled_func(
            merged if callable(merged) else lambda: merged
        )
        self.backbone_model.change_out_bias(
            spin_sampled_func,
            bias_adjust_mode=bias_adjust_mode,
        )

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any = None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        type_map_with_spin = type_map + [item + "_spin" for item in type_map]
        self.backbone_model.change_type_map(
            type_map_with_spin, model_with_new_type_stat
        )

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the wrapped model."""
        if "backbone_model" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.backbone_model, name)

    def serialize(self) -> dict:
        return {
            "backbone_model": self.backbone_model.serialize(),
            "spin": self.spin.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SpinModel":
        backbone_model_obj = make_model(
            DPAtomicModel, T_Bases=(NativeOP, BaseModel)
        ).deserialize(data["backbone_model"])
        spin = Spin.deserialize(data["spin"])
        return cls(
            backbone_model=backbone_model_obj,
            spin=spin,
        )

    def call_common(
        self,
        coord: Array,
        atype: Array,
        spin: Array,
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
        """Return model prediction with raw internal keys.

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
        xp = array_api_compat.array_namespace(coord)
        nframes, nloc = atype.shape[:2]
        coord = xp.reshape(coord, (nframes, nloc, 3))
        spin = xp.reshape(spin, (nframes, nloc, 3))
        coord_updated, atype_updated, coord_corr_for_virial = self.process_spin_input(
            coord, atype, spin
        )
        if aparam is not None:
            aparam = self.expand_aparam(aparam, nloc * 2)
        model_ret = self.backbone_model.call_common(
            coord_updated,
            atype_updated,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            coord_corr_for_virial=coord_corr_for_virial,
        )
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        model_ret[f"{var_name}"] = model_ret[f"{var_name}"][:, :nloc]
        if (
            self.backbone_model.do_grad_r(var_name)
            and model_ret.get(f"{var_name}_derv_r") is not None
        ):
            (
                model_ret[f"{var_name}_derv_r"],
                model_ret[f"{var_name}_derv_r_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output(atype, model_ret[f"{var_name}_derv_r"])
        if (
            self.backbone_model.do_grad_c(var_name)
            and do_atomic_virial
            and model_ret.get(f"{var_name}_derv_c") is not None
        ):
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
        # Always compute mask_mag from atom types (even when forces are unavailable)
        if "mask_mag" not in model_ret:
            xp = array_api_compat.array_namespace(atype)
            nframes_m, nloc_m = atype.shape[:2]
            vsm = self._to_xp(self.virtual_scale_mask, xp, atype)
            atomic_mask = xp.reshape(vsm[atype], (nframes_m, nloc_m, 1))
            model_ret["mask_mag"] = atomic_mask > 0.0
        return model_ret

    def call(
        self,
        coord: Array,
        atype: Array,
        spin: Array,
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
        """Return model prediction with translated user-facing keys.

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
            The result dict with translated keys, e.g.
            ``atom_energy``, ``energy``, ``force``, ``force_mag``.

        """
        model_ret = self.call_common(
            coord,
            atype,
            spin,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
        if "mask_mag" in model_ret:
            model_predict["mask_mag"] = model_ret["mask_mag"]
        if (
            self.backbone_model.do_grad_r(var_name)
            and model_ret.get(f"{var_name}_derv_r") is not None
        ):
            model_predict["force"] = model_ret[f"{var_name}_derv_r"].squeeze(-2)
            model_predict["force_mag"] = model_ret[f"{var_name}_derv_r_mag"].squeeze(-2)
        if (
            self.backbone_model.do_grad_c(var_name)
            and model_ret.get(f"{var_name}_derv_c_redu") is not None
        ):
            model_predict["virial"] = model_ret[f"{var_name}_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret.get(f"{var_name}_derv_c") is not None:
                model_predict["atom_virial"] = model_ret[f"{var_name}_derv_c"].squeeze(
                    -2
                )
        return model_predict

    def call_common_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        extended_spin: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
        """Return model prediction with raw internal keys. Lower interface that takes
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
            extended_coord_corr,
        ) = self.process_spin_input_lower(
            extended_coord, extended_atype, extended_spin, nlist, mapping=mapping
        )
        if aparam is not None:
            aparam = self.expand_aparam(aparam, nloc * 2)
        model_ret = self.backbone_model.call_common_lower(
            extended_coord_updated,
            extended_atype_updated,
            nlist_updated,
            mapping=mapping_updated,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            extended_coord_corr=extended_coord_corr,
        )
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        model_ret[f"{var_name}"] = model_ret[f"{var_name}"][:, :nloc]
        if (
            self.backbone_model.do_grad_r(var_name)
            and model_ret.get(f"{var_name}_derv_r") is not None
        ):
            (
                model_ret[f"{var_name}_derv_r"],
                model_ret[f"{var_name}_derv_r_mag"],
                model_ret["mask_mag"],
            ) = self.process_spin_output_lower(
                extended_atype, model_ret[f"{var_name}_derv_r"], nloc
            )
        if (
            self.backbone_model.do_grad_c(var_name)
            and do_atomic_virial
            and model_ret.get(f"{var_name}_derv_c") is not None
        ):
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
        # Always compute mask_mag from atom types (even when forces are unavailable)
        if "mask_mag" not in model_ret:
            xp = array_api_compat.array_namespace(extended_atype)
            nall = extended_atype.shape[1]
            vsm = self._to_xp(self.virtual_scale_mask, xp, extended_atype)
            atomic_mask = xp.reshape(vsm[extended_atype], (nframes, nall, 1))
            model_ret["mask_mag"] = atomic_mask > 0.0
        return model_ret

    def call_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        extended_spin: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
        """Return model prediction with translated user-facing keys. Lower interface.

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
            The result dict with translated keys, e.g.
            ``atom_energy``, ``energy``, ``extended_force``, ``extended_force_mag``.

        """
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            extended_spin,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
        if "mask_mag" in model_ret:
            model_predict["extended_mask_mag"] = model_ret["mask_mag"]
        if (
            self.backbone_model.do_grad_r(var_name)
            and model_ret.get(f"{var_name}_derv_r") is not None
        ):
            model_predict["extended_force"] = model_ret[f"{var_name}_derv_r"].squeeze(
                -2
            )
            model_predict["extended_force_mag"] = model_ret[
                f"{var_name}_derv_r_mag"
            ].squeeze(-2)
        if (
            self.backbone_model.do_grad_c(var_name)
            and model_ret.get(f"{var_name}_derv_c_redu") is not None
        ):
            model_predict["virial"] = model_ret[f"{var_name}_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret.get(f"{var_name}_derv_c") is not None:
                model_predict["extended_virial"] = model_ret[
                    f"{var_name}_derv_c"
                ].squeeze(-2)
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
        """Get the translated output definition.

        Maps internal output names to user-facing names, e.g.
        ``energy`` -> ``atom_energy``, ``energy_redu`` -> ``energy``,
        ``energy_derv_r`` -> ``force``, ``energy_derv_r_mag`` -> ``force_mag``.
        """
        out_def_data = self.model_output_def().get_data()
        model_output_type = self.backbone_model.model_output_type()
        if "mask" in model_output_type:
            model_output_type.pop(model_output_type.index("mask"))
        var_name = model_output_type[0]
        output_def = {
            f"atom_{var_name}": out_def_data[var_name],
            var_name: out_def_data[f"{var_name}_redu"],
            "mask_mag": out_def_data["mask_mag"],
        }
        if self.backbone_model.do_grad_r(var_name):
            output_def["force"] = deepcopy(out_def_data[f"{var_name}_derv_r"])
            output_def["force"].squeeze(-2)
            output_def["force_mag"] = deepcopy(out_def_data[f"{var_name}_derv_r_mag"])
            output_def["force_mag"].squeeze(-2)
        if self.backbone_model.do_grad_c(var_name):
            output_def["virial"] = deepcopy(out_def_data[f"{var_name}_derv_c_redu"])
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = deepcopy(out_def_data[f"{var_name}_derv_c"])
            output_def["atom_virial"].squeeze(-2)
        return output_def
