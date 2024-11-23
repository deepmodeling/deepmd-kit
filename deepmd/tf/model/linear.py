# SPDX-License-Identifier: LGPL-3.0-or-later
import operator
from enum import (
    Enum,
)
from functools import (
    lru_cache,
    reduce,
)
from typing import (
    Optional,
    Union,
)

from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    MODEL_VERSION,
    tf,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

from .model import (
    Model,
)


class LinearModel(Model):
    """Linear model make linear combinations of several existing models.

    Parameters
    ----------
    models : list[dict]
        A list of models to be combined.
    weights : list[float] or str
        If the type is list[float], a list of weights for each model.
        If "mean", the weights are set to be 1 / len(models).
        If "sum", the weights are set to be 1.
    """

    def __init__(self, models: list[dict], weights: list[float], **kwargs) -> None:
        super().__init__(**kwargs)
        self.models = [Model(**model) for model in models]
        if isinstance(weights, list):
            if len(weights) != len(models):
                raise ValueError(
                    "The length of weights is not equal to the number of models"
                )
            self.weights = weights
        elif weights == "mean":
            self.weights = [1 / len(models) for _ in range(len(models))]
        elif weights == "sum":
            self.weights = [1 for _ in range(len(models))]
        else:
            raise ValueError(f"Invalid weights {weights}")

    def get_fitting(self) -> Union[Fitting, dict]:
        """Get the fitting(s)."""
        return {
            f"model{ii}": model.get_fitting() for ii, model in enumerate(self.models)
        }

    def get_loss(self, loss: dict, lr) -> Optional[Union[Loss, dict]]:
        """Get the loss function(s)."""
        # the first model that is not None, or None if all models are None
        for model in self.models:
            loss = model.get_loss(loss, lr)
            if loss is not None:
                return loss
        return None

    def get_rcut(self):
        return max([model.get_rcut() for model in self.models])

    @lru_cache(maxsize=1)
    def get_ntypes(self) -> int:
        # check if all models have the same ntypes
        for model in self.models:
            if model.get_ntypes() != self.models[0].get_ntypes():
                raise ValueError("Models have different ntypes")
        return self.models[0].get_ntypes()

    def data_stat(self, data) -> None:
        for model in self.models:
            model.data_stat(data)

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        model_type: str = "original_model",
        suffix: str = "",
    ) -> None:
        """Init the embedding net variables with the given frozen model.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        model_type : str
            the type of the model
        suffix : str
            suffix to name scope
        """
        for ii, model in enumerate(self.models):
            model.init_variables(
                graph, graph_def, model_type, suffix=f"_model{ii}{suffix}"
            )

    def enable_compression(self, suffix: str = "") -> None:
        """Enable compression.

        Parameters
        ----------
        suffix : str
            suffix to name scope
        """
        for ii, model in enumerate(self.models):
            model.enable_compression(suffix=f"_model{ii}{suffix}")

    def get_type_map(self) -> list:
        """Get the type map."""
        return self.models[0].get_type_map()

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
            data used to do neighbor statistics
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
        local_jdata_cpy = local_jdata.copy()
        new_list = []
        min_nbor_dist = None
        for sub_jdata in local_jdata["models"]:
            new_sub_jdata, min_nbor_dist_ = Model.update_sel(
                train_data, type_map, sub_jdata
            )
            if min_nbor_dist_ is not None:
                min_nbor_dist = min_nbor_dist_
            new_list.append(new_sub_jdata)
        local_jdata_cpy["models"] = new_list
        return local_jdata_cpy, min_nbor_dist

    @property
    def input_requirement(self) -> list[DataRequirementItem]:
        """Return data requirements needed for the model input."""
        return reduce(
            operator.iadd, [model.input_requirement for model in self.models], []
        )


@Model.register("linear_ener")
class LinearEnergyModel(LinearModel):
    """Linear energy model make linear combinations of several existing energy models."""

    model_type = "ener"

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: dict,
        frz_model: Optional[str] = None,
        ckpt_meta: Optional[str] = None,
        suffix: str = "",
        reuse: Optional[Union[bool, Enum]] = None,
    ) -> dict:
        """Build the model.

        Parameters
        ----------
        coord_ : tf.Tensor
            The coordinates of atoms
        atype_ : tf.Tensor
            The atom types of atoms
        natoms : tf.Tensor
            The number of atoms
        box : tf.Tensor
            The box vectors
        mesh : tf.Tensor
            The mesh vectors
        input_dict : dict
            The input dict
        frz_model : str, optional
            The path to the frozen model
        ckpt_meta : str, optional
            The path prefix of the checkpoint and meta files
        suffix : str, optional
            The suffix of the scope
        reuse : bool or tf.AUTO_REUSE, optional
            Whether to reuse the variables

        Returns
        -------
        dict
            The output dict
        """
        with tf.variable_scope("model_attr" + suffix, reuse=reuse):
            t_tmap = tf.constant(
                " ".join(self.get_type_map()), name="tmap", dtype=tf.string
            )
            t_mt = tf.constant(self.model_type, name="model_type", dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION, name="model_version", dtype=tf.string)
        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            # non zero not supported
            t_dfparam = tf.constant(0, name="dfparam", dtype=tf.int32)
            t_daparam = tf.constant(0, name="daparam", dtype=tf.int32)
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            t_ntypes = tf.constant(self.get_ntypes(), name="ntypes", dtype=tf.int32)
            t_rcut = tf.constant(
                self.get_rcut(), name="rcut", dtype=GLOBAL_TF_FLOAT_PRECISION
            )

        subdicts = []
        for ii, model in enumerate(self.models):
            subdict = model.build(
                coord_,
                atype_,
                natoms,
                box,
                mesh,
                input_dict,
                frz_model=frz_model,
                ckpt_meta=ckpt_meta,
                suffix=f"_model{ii}{suffix}",
                reuse=reuse,
            )
            subdicts.append(subdict)
        t_weight = tf.constant(self.weights, dtype=GLOBAL_TF_FLOAT_PRECISION)

        model_dict = {}
        # energy shape is (n_batch,), other shapes are (n_batch, -1)
        energy = tf.reduce_sum(
            tf.stack([mm["energy"] for mm in subdicts], axis=0) * t_weight[:, None],
            axis=0,
        )
        force = tf.reduce_sum(
            tf.stack([mm["force"] for mm in subdicts], axis=0)
            * t_weight[:, None, None],
            axis=0,
        )
        virial = tf.reduce_sum(
            tf.stack([mm["virial"] for mm in subdicts], axis=0)
            * t_weight[:, None, None],
            axis=0,
        )
        atom_ener = tf.reduce_sum(
            tf.stack([mm["atom_ener"] for mm in subdicts], axis=0)
            * t_weight[:, None, None],
            axis=0,
        )
        atom_virial = tf.reduce_sum(
            tf.stack([mm["atom_virial"] for mm in subdicts], axis=0)
            * t_weight[:, None, None],
            axis=0,
        )

        model_dict["energy"] = tf.identity(energy, name="o_energy" + suffix)
        model_dict["force"] = tf.identity(force, name="o_force" + suffix)
        model_dict["virial"] = tf.identity(virial, name="o_virial" + suffix)
        model_dict["atom_ener"] = tf.identity(atom_ener, name="o_atom_energy" + suffix)
        model_dict["atom_virial"] = tf.identity(
            atom_virial, name="o_atom_virial" + suffix
        )

        model_dict["coord"] = coord_
        model_dict["atype"] = atype_
        return model_dict
