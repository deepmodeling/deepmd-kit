# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
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
        model,
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

    def __call__(
        self,
        coord: jnp.ndarray,
        atype: jnp.ndarray,
        box: Optional[jnp.ndarray] = None,
        fparam: Optional[jnp.ndarray] = None,
        aparam: Optional[jnp.ndarray] = None,
        do_atomic_virial: bool = False,
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
            The result dict of type dict[str,jnp.ndarray].
            The keys are defined by the `ModelOutputDef`.

        """
        return self.call(coord, atype, box, fparam, aparam, do_atomic_virial)

    def call(
        self,
        coord: jnp.ndarray,
        atype: jnp.ndarray,
        box: Optional[jnp.ndarray] = None,
        fparam: Optional[jnp.ndarray] = None,
        aparam: Optional[jnp.ndarray] = None,
        do_atomic_virial: bool = False,
    ):
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
        return call(
            coord,
            atype,
            box,
            fparam,
            aparam,
        )

    def model_output_def(self):
        return ModelOutputDef(
            FittingOutputDef([OUTPUT_DEFS[tt] for tt in self.model_output_type()])
        )

    def call_lower(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: Optional[jnp.ndarray] = None,
        fparam: Optional[jnp.ndarray] = None,
        aparam: Optional[jnp.ndarray] = None,
        do_atomic_virial: bool = False,
    ):
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
        return call_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam,
            aparam,
        )

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def get_rcut(self):
        """Get the cut-off radius."""
        return self.rcut

    def get_dim_fparam(self):
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.dim_fparam

    def get_dim_aparam(self):
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

    def get_min_nbor_dist(self) -> Optional[float]:
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
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
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
