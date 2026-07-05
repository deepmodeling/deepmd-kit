# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import jax.experimental.jax2tf as jax2tf
import tensorflow as tf

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.jax.env import (
    jnp,
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
    "mask": OutputVariableDef(
        "mask",
        shape=[1],
        reducible=False,
        r_differentiable=False,
        c_differentiable=False,
    ),
}


def decode_list_of_bytes(list_of_bytes: list[bytes]) -> list[str]:
    """Decode a list of bytes to a list of strings."""
    return [x.decode() for x in list_of_bytes]


class TFModelWrapper(tf.Module):
    def __init__(
        self,
        model: str,
    ) -> None:
        self.model = tf.saved_model.load(model)
        self._call_lower = jax2tf.call_tf(self.model.call_lower)
        self._call_lower_atomic_virial = jax2tf.call_tf(
            self.model.call_lower_atomic_virial
        )
        self._call = jax2tf.call_tf(self.model.call)
        self._call_atomic_virial = jax2tf.call_tf(self.model.call_atomic_virial)
        self.type_map = decode_list_of_bytes(self.model.get_type_map().numpy().tolist())
        self.rcut = self.model.get_rcut().numpy().item()
        self.dim_fparam = self.model.get_dim_fparam().numpy().item()
        self.dim_aparam = self.model.get_dim_aparam().numpy().item()
        self.sel_type = self.model.get_sel_type().numpy().tolist()
        self._is_aparam_nall = self.model.is_aparam_nall().numpy().item()
        self._model_output_type = decode_list_of_bytes(
            self.model.model_output_type().numpy().tolist()
        )
        self._mixed_types = self.model.mixed_types().numpy().item()
        if hasattr(self.model, "get_min_nbor_dist"):
            self.min_nbor_dist = self.model.get_min_nbor_dist().numpy().item()
        else:
            self.min_nbor_dist = None
        self.sel = self.model.get_sel().numpy().tolist()
        self.model_def_script = self.model.get_model_def_script().numpy().decode()
        if hasattr(self.model, "has_default_fparam"):
            # No attrs before v3.1.2
            self._has_default_fparam = self.model.has_default_fparam().numpy().item()
        else:
            self._has_default_fparam = False
        if hasattr(self.model, "get_default_fparam"):
            self.default_fparam = self.model.get_default_fparam().numpy().tolist()
        else:
            self.default_fparam = None
        self._has_chg_spin_ebd = (
            self.model.has_chg_spin_ebd().numpy().item()
            if hasattr(self.model, "has_chg_spin_ebd")
            else False
        )
        self.dim_chg_spin = (
            self.model.get_dim_chg_spin().numpy().item()
            if hasattr(self.model, "get_dim_chg_spin")
            else 0
        )
        self._has_default_chg_spin = (
            self.model.has_default_chg_spin().numpy().item()
            if hasattr(self.model, "has_default_chg_spin")
            else False
        )
        self.default_chg_spin = (
            self.model.get_default_chg_spin().numpy().tolist()
            if hasattr(self.model, "get_default_chg_spin")
            else None
        )

    def __call__(
        self,
        coord: jnp.ndarray,
        atype: jnp.ndarray,
        box: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        do_atomic_virial: bool = False,
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
        charge_spin
            The charge and spin conditioning input. shape: nf x dim_chg_spin

        Returns
        -------
        ret_dict
            The result dict of type dict[str,jnp.ndarray].
            The keys are defined by the `ModelOutputDef`.

        """
        return self.call(
            coord,
            atype,
            box,
            fparam,
            aparam,
            do_atomic_virial=do_atomic_virial,
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
        charge_spin
            The charge and spin conditioning input. shape: nf x dim_chg_spin

        Returns
        -------
        ret_dict
            The result dict of type dict[str,jnp.ndarray].
            The keys are defined by the `ModelOutputDef`.

        """
        if do_atomic_virial:
            call = self._call_atomic_virial
        else:
            call = self._call
        # Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.
        if box is None:
            box = jnp.empty((coord.shape[0], 0, 0), dtype=jnp.float64)
        if fparam is None:
            fparam = jnp.empty(
                (coord.shape[0], self.get_dim_fparam()), dtype=jnp.float64
            )
        if aparam is None:
            aparam = jnp.empty(
                (coord.shape[0], coord.shape[1], self.get_dim_aparam()),
                dtype=jnp.float64,
            )
        args = (coord, atype, box, fparam, aparam)
        if self.get_dim_chg_spin() > 0:
            charge_spin = self._make_charge_spin_input(coord.shape[0], charge_spin)
            args = (*args, charge_spin)
        return call(*args)

    def model_output_def(self) -> ModelOutputDef:
        return ModelOutputDef(
            FittingOutputDef([OUTPUT_DEFS[tt] for tt in self.model_output_type()])
        )

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
    ) -> dict[str, jnp.ndarray]:
        if do_atomic_virial:
            call_lower = self._call_lower_atomic_virial
        else:
            call_lower = self._call_lower
        # Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.
        if fparam is None:
            fparam = jnp.empty(
                (extended_coord.shape[0], self.get_dim_fparam()), dtype=jnp.float64
            )
        if aparam is None:
            aparam = jnp.empty(
                (extended_coord.shape[0], nlist.shape[1], self.get_dim_aparam()),
                dtype=jnp.float64,
            )
        args = (extended_coord, extended_atype, nlist, mapping, fparam, aparam)
        if self.get_dim_chg_spin() > 0:
            charge_spin = self._make_charge_spin_input(
                extended_coord.shape[0], charge_spin
            )
            args = (*args, charge_spin)
        return call_lower(*args)

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.dim_fparam

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.dim_aparam

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
    def deserialize(cls, data: dict) -> "TFModelWrapper":
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
    def get_model(cls, model_params: dict) -> "TFModelWrapper":
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

    def has_chg_spin_ebd(self) -> bool:
        """Check if the model has charge spin embedding."""
        return self._has_chg_spin_ebd

    def get_dim_chg_spin(self) -> int:
        """Get the dimension of charge_spin input."""
        return self.dim_chg_spin

    def has_default_chg_spin(self) -> bool:
        """Check if the model has default charge_spin values."""
        return self._has_default_chg_spin

    def get_default_chg_spin(self) -> list[float] | None:
        """Get the default charge_spin values."""
        return self.default_chg_spin

    def _make_charge_spin_input(
        self, nframes: int, charge_spin: jnp.ndarray | None
    ) -> jnp.ndarray:
        dim_chg_spin = self.get_dim_chg_spin()
        if dim_chg_spin == 0:
            return jnp.empty((nframes, 0), dtype=jnp.float64)
        if charge_spin is None:
            if self.has_default_chg_spin():
                default_chg_spin = self.get_default_chg_spin()
                assert default_chg_spin is not None
                return jnp.tile(
                    jnp.asarray(default_chg_spin, dtype=jnp.float64).reshape(1, -1),
                    (nframes, 1),
                )
            raise ValueError(
                "charge_spin is required for this model but was not provided, "
                "and the model has no default_chg_spin."
            )
        charge_spin = jnp.asarray(charge_spin, dtype=jnp.float64)
        if charge_spin.ndim == 1:
            if charge_spin.size != dim_chg_spin:
                raise ValueError("charge_spin must contain [charge, spin].")
            charge_spin = charge_spin.reshape(1, dim_chg_spin)
        elif charge_spin.ndim != 2 or charge_spin.shape[-1] != dim_chg_spin:
            raise ValueError("charge_spin must have shape (nframes, 2).")
        if charge_spin.shape[0] == 1 and nframes != 1:
            return jnp.tile(charge_spin, (nframes, 1))
        if charge_spin.shape[0] != nframes:
            raise ValueError("charge_spin first dimension must match nframes.")
        return charge_spin
