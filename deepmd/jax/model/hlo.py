# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from copy import (
    deepcopy,
)
from functools import (
    partial,
)
from typing import (
    Any,
)

from deepmd.dpmodel.model.make_model import (
    model_call_from_call_lower,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.jax.env import (
    jax_export,
    jnp,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

OUTPUT_DEFS = {
    "energy": OutputVariableDef(
        "energy",
        shape=[1],
        reducible=True,
        r_differentiable=True,
        c_differentiable=True,
    ),
    "energy_hessian": OutputVariableDef(
        "energy",
        shape=[1],
        reducible=True,
        r_differentiable=True,
        c_differentiable=True,
        r_hessian=True,
    ),
    "dos": OutputVariableDef(
        "dos",
        shape=[-1],
        reducible=True,
        r_differentiable=False,
        c_differentiable=False,
    ),
    "mask": OutputVariableDef(
        "mask",
        shape=[1],
        reducible=False,
        r_differentiable=False,
        c_differentiable=False,
    ),
    "mask_mag": OutputVariableDef(
        "mask_mag",
        shape=[1],
        reducible=False,
        r_differentiable=False,
        c_differentiable=False,
    ),
}


class HLO(BaseModel):
    def __init__(
        self,
        stablehlo: bytearray,
        stablehlo_atomic_virial: bytearray,
        stablehlo_no_ghost: bytearray,
        stablehlo_atomic_virial_no_ghost: bytearray,
        model_def_script: str,
        type_map: list[str],
        rcut: float,
        dim_fparam: int,
        dim_aparam: int,
        sel_type: list[int],
        is_aparam_nall: bool,
        model_output_type: list[str],
        mixed_types: bool,
        min_nbor_dist: float | None,
        sel: list[int],
        # new in v3.1.1
        has_default_fparam: bool = False,
        default_fparam: list[float] | None = None,
        numb_dos: int = 0,
        # property models only
        var_name: str | None = None,
        task_dim: int | None = None,
        intensive: bool = False,
        is_spin: bool = False,
        dim_chg_spin: int = 0,
        default_chg_spin: list[float] | None = None,
        full_neighbor_list: bool = False,
    ) -> None:
        self._call_lower = jax_export.deserialize(stablehlo).call
        self._call_lower_atomic_virial = jax_export.deserialize(
            stablehlo_atomic_virial
        ).call
        self._call_lower_no_ghost = jax_export.deserialize(stablehlo_no_ghost).call
        self._call_lower_atomic_virial_no_ghost = jax_export.deserialize(
            stablehlo_atomic_virial_no_ghost
        ).call
        self.stablehlo = stablehlo
        self.type_map = type_map
        self.rcut = rcut
        self.dim_fparam = dim_fparam
        self.dim_aparam = dim_aparam
        self.sel_type = sel_type
        self._is_aparam_nall = is_aparam_nall
        self._model_output_type = model_output_type
        self._mixed_types = mixed_types
        self.min_nbor_dist = min_nbor_dist
        self.sel = sel
        self.model_def_script = model_def_script
        self._has_default_fparam = has_default_fparam
        self.default_fparam = default_fparam
        self.numb_dos = numb_dos
        self._var_name = var_name
        self._task_dim = task_dim
        self._intensive = intensive
        self._is_spin = is_spin
        self._dim_chg_spin = dim_chg_spin
        self._default_chg_spin = default_chg_spin
        self._full_neighbor_list = full_neighbor_list
        # Model-level pair_exclude_types, rebuilt from the training config so
        # the outer wrapper can fold it into the nlist at BUILD time (decision
        # #18/A4) — the serialized StableHLO lower consumes a pre-excluded
        # nlist and never re-applies it.
        import json

        from deepmd.dpmodel.utils.exclude_mask import (
            PairExcludeMask,
        )

        pet = []
        if model_def_script:
            try:
                pet = json.loads(model_def_script).get("pair_exclude_types", [])
            except (ValueError, AttributeError):
                pet = []
        self._pair_excl = (
            PairExcludeMask(len(type_map), [tuple(p) for p in pet]) if pet else None
        )

    def __call__(
        self,
        coord: jnp.ndarray,
        atype: jnp.ndarray,
        box: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        do_atomic_virial: bool = False,
        spin: jnp.ndarray | None = None,
        charge_spin: jnp.ndarray | None = None,
    ) -> Any:
        """Return model prediction.

        Parameters
        ----------
        coord
            The coordinates of the atoms.
            shape: nf x (nloc x 3)
        atype
            The type of atoms. shape: nf x nloc
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
        return self.call(
            coord,
            atype,
            box,
            fparam,
            aparam,
            do_atomic_virial,
            spin=spin,
            charge_spin=charge_spin,
        )

    def call(
        self,
        coord: jnp.ndarray,
        atype: jnp.ndarray,
        box: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        do_atomic_virial: bool = False,
        spin: jnp.ndarray | None = None,
        charge_spin: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Return model prediction.

        Parameters
        ----------
        coord
            The coordinates of the atoms.
            shape: nf x (nloc x 3)
        atype
            The type of atoms. shape: nf x nloc
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
        from deepmd.jax.utils.neighbor_list import (
            FullNeighborList,
        )

        return model_call_from_call_lower(
            call_lower=partial(self.call_lower, spin=spin),
            rcut=self.get_rcut(),
            sel=self.get_sel(),
            mixed_types=self.mixed_types(),
            model_output_def=self.model_output_def(),
            coord=coord,
            atype=atype,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
            # exclusion is a nlist-BUILD transform (decision #18/A4); the
            # serialized StableHLO lower consumes a pre-excluded nlist.
            pair_excl=self._pair_excl,
            neighbor_list=FullNeighborList() if self._full_neighbor_list else None,
        )

    def model_output_def(self) -> ModelOutputDef:
        return ModelOutputDef(
            FittingOutputDef(
                [self._output_var_def(tt) for tt in self.model_output_type()]
            )
        )

    def _output_var_def(self, name: str) -> OutputVariableDef:
        if name == "energy" and json.loads(self.model_def_script).get(
            "hessian_mode", False
        ):
            definition = deepcopy(OUTPUT_DEFS["energy_hessian"])
            definition.magnetic = self._is_spin
            return definition
        if name in OUTPUT_DEFS:
            definition = deepcopy(OUTPUT_DEFS[name])
            if name == "energy":
                definition.magnetic = self._is_spin
            return definition
        # property models carry a user-defined output name (``var_name``) that
        # is not in the fixed table; rebuild its def from the persisted metadata.
        if self._var_name is not None and name == self._var_name:
            return OutputVariableDef(
                self._var_name,
                shape=[self._task_dim],
                reducible=True,
                r_differentiable=False,
                c_differentiable=False,
                intensive=self._intensive,
            )
        raise KeyError(f"Unknown model output variable {name!r}")

    def call_lower(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        do_atomic_virial: bool = False,
        charge_spin: jnp.ndarray | None = None,
        spin: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        extended_coord = jnp.asarray(extended_coord, dtype=jnp.float64)
        extended_atype = jnp.asarray(extended_atype, dtype=jnp.int32)
        nlist = jnp.asarray(nlist, dtype=jnp.int64)
        if mapping is None:
            if extended_atype.shape[1] != nlist.shape[1]:
                raise ValueError("Ghost atoms require an extended-to-local mapping.")
            mapping = jnp.broadcast_to(
                jnp.arange(extended_atype.shape[1], dtype=jnp.int64),
                extended_atype.shape,
            )
        else:
            mapping = jnp.asarray(mapping, dtype=jnp.int64)
        fparam = jnp.asarray(fparam, dtype=jnp.float64) if fparam is not None else None
        aparam = jnp.asarray(aparam, dtype=jnp.float64) if aparam is not None else None
        if extended_coord.shape[1] > nlist.shape[1]:
            if do_atomic_virial:
                call_lower = self._call_lower_atomic_virial
            else:
                call_lower = self._call_lower
        else:
            if do_atomic_virial:
                call_lower = self._call_lower_atomic_virial_no_ghost
            else:
                call_lower = self._call_lower_no_ghost
        arguments = (
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam,
            aparam,
        )
        if self._is_spin or self._dim_chg_spin:
            nf, nall = extended_atype.shape
            if self._is_spin:
                if spin is None:
                    raise ValueError(
                        "A native-spin model requires per-atom spin vectors."
                    )
                spin = jnp.asarray(spin, dtype=extended_coord.dtype).reshape(nf, -1, 3)
                spin = jnp.pad(spin, ((0, 0), (0, nall - spin.shape[1]), (0, 0)))
            if self._dim_chg_spin and charge_spin is None:
                if self._default_chg_spin is None:
                    raise ValueError("The model requires charge_spin conditioning.")
                charge_spin = jnp.broadcast_to(
                    jnp.asarray(self._default_chg_spin, dtype=extended_coord.dtype),
                    (nf, self._dim_chg_spin),
                )
            if charge_spin is not None:
                charge_spin = jnp.asarray(charge_spin, dtype=jnp.float64)
            arguments += (spin, charge_spin)
        return call_lower(*arguments)

    def has_spin(self) -> bool:
        """Whether the artifact accepts per-atom native-spin vectors."""
        return self._is_spin

    def has_chg_spin_ebd(self) -> bool:
        """Whether the artifact accepts frame-level charge/spin conditioning."""
        return self._dim_chg_spin > 0

    def get_dim_chg_spin(self) -> int:
        return self._dim_chg_spin

    def get_default_chg_spin(self) -> list[float] | None:
        return self._default_chg_spin

    def uses_full_neighbor_list(self) -> bool:
        """Whether the artifact consumes every neighbor within the cutoff."""
        return self._full_neighbor_list

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    def get_numb_dos(self) -> int:
        """Get the number of DOS."""
        return self.numb_dos

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.dim_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.dim_aparam

    def get_var_name(self) -> str:
        """Get the name of the property (property models only)."""
        if self._var_name is None:
            raise NotImplementedError
        return self._var_name

    def get_task_dim(self) -> int:
        """Get the output dimension of the property (property models only)."""
        if self._task_dim is None:
            raise NotImplementedError
        return self._task_dim

    def get_intensive(self) -> bool:
        """Whether the property is intensive (property models only)."""
        return self._intensive

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.sel_type

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return self._is_aparam_nall

    def model_output_type(self) -> list[str]:
        """Get the output type for the model."""
        return self._model_output_type

    def serialize(self) -> dict:
        """Serialize the model.

        Returns
        -------
        dict
            The serialized data
        """
        raise NotImplementedError("Not implemented")

    @classmethod
    def deserialize(cls, data: dict) -> "BaseModel":
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        BaseModel
            The deserialized model
        """
        raise NotImplementedError("Not implemented")

    def get_observed_type_list(self) -> list[str]:
        """Get observed types (elements) of the model during data statistics."""
        raise NotImplementedError("Not implemented for HLO model")

    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        return self.model_def_script

    def get_min_nbor_dist(self) -> float | None:
        """Get the minimum distance between two atoms."""
        return self.min_nbor_dist

    def get_nnei(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return self.get_nsel()

    def get_sel(self) -> list[int]:
        return self.sel

    def get_nsel(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return sum(self.sel)

    def mixed_types(self) -> bool:
        return self._mixed_types

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        raise NotImplementedError("Not implemented")

    @classmethod
    def get_model(cls, model_params: dict) -> "BaseModel":
        """Get the model by the parameters.

        By default, all the parameters are directly passed to the constructor.
        If not, override this method.

        Parameters
        ----------
        model_params : dict
            The model parameters

        Returns
        -------
        BaseBaseModel
            The model
        """
        raise NotImplementedError("Not implemented")

    def has_default_fparam(self) -> bool:
        """Check whether the model has default frame parameters."""
        return self._has_default_fparam

    def get_default_fparam(self) -> list[float] | None:
        """Get the default frame parameters."""
        return self.default_fparam
