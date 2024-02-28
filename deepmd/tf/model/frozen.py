# SPDX-License-Identifier: LGPL-3.0-or-later
from enum import (
    Enum,
)
from typing import (
    Optional,
    Union,
)

from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    MODEL_VERSION,
    tf,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.infer import (
    DeepPotential,
)
from deepmd.tf.loss.loss import (
    Loss,
)

from .model import (
    Model,
)


@Model.register("frozen")
class FrozenModel(Model):
    """Load model from a frozen model, which cannot be trained.

    Parameters
    ----------
    model_file : str
        The path to the frozen model
    """

    def __init__(self, model_file: str, **kwargs):
        super().__init__(**kwargs)
        self.model_file = model_file
        self.model = DeepPotential(model_file)
        if isinstance(self.model, DeepPot):
            self.model_type = "ener"
        else:
            raise NotImplementedError(
                "This model type has not been implemented. " "Contribution is welcome!"
            )

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
        # reset the model to import to the correct graph
        extra_feed_dict = {}
        if input_dict is not None:
            if "fparam" in input_dict:
                extra_feed_dict["fparam"] = input_dict["fparam"]
            if "aparam" in input_dict:
                extra_feed_dict["aparam"] = input_dict["aparam"]
        input_map = self.get_feed_dict(
            coord_, atype_, natoms, box, mesh, **extra_feed_dict
        )
        self.model = DeepPotential(
            self.model_file,
            default_tf_graph=True,
            load_prefix="load" + suffix,
            input_map=input_map,
        )

        with tf.variable_scope("model_attr" + suffix, reuse=reuse):
            t_tmap = tf.constant(
                " ".join(self.get_type_map()), name="tmap", dtype=tf.string
            )
            t_mt = tf.constant(self.model_type, name="model_type", dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION, name="model_version", dtype=tf.string)
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            t_ntypes = tf.constant(self.get_ntypes(), name="ntypes", dtype=tf.int32)
            t_rcut = tf.constant(
                self.get_rcut(), name="rcut", dtype=GLOBAL_TF_FLOAT_PRECISION
            )
        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            t_dfparam = tf.constant(
                self.model.get_dim_fparam(), name="dfparam", dtype=tf.int32
            )
            t_daparam = tf.constant(
                self.model.get_dim_aparam(), name="daparam", dtype=tf.int32
            )
        if self.model_type == "ener":
            return {
                # must visit the backend class
                "energy": tf.identity(
                    self.model.deep_eval.output_tensors["energy_redu"],
                    name="o_energy" + suffix,
                ),
                "force": tf.identity(
                    self.model.deep_eval.output_tensors["energy_derv_r"],
                    name="o_force" + suffix,
                ),
                "virial": tf.identity(
                    self.model.deep_eval.output_tensors["energy_derv_c_redu"],
                    name="o_virial" + suffix,
                ),
                "atom_ener": tf.identity(
                    self.model.deep_eval.output_tensors["energy"],
                    name="o_atom_energy" + suffix,
                ),
                "atom_virial": tf.identity(
                    self.model.deep_eval.output_tensors["energy_derv_c"],
                    name="o_atom_virial" + suffix,
                ),
                "coord": coord_,
                "atype": atype_,
            }
        else:
            raise NotImplementedError(
                f"Model type {self.model_type} has not been implemented. "
                "Contribution is welcome!"
            )

    def get_fitting(self) -> Union[Fitting, dict]:
        """Get the fitting(s)."""
        return {}

    def get_loss(self, loss: dict, lr) -> Optional[Union[Loss, dict]]:
        """Get the loss function(s)."""
        # loss should be never used for a frozen model
        return

    def get_rcut(self):
        return self.model.get_rcut()

    def get_ntypes(self) -> int:
        return self.model.get_ntypes()

    def data_stat(self, data):
        pass

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
        pass

    def enable_compression(self, suffix: str = "") -> None:
        """Enable compression.

        Parameters
        ----------
        suffix : str
            suffix to name scope
        """
        pass

    def get_type_map(self) -> list:
        """Get the type map."""
        return self.model.get_type_map()

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        # we don't know how to compress it, so no neighbor statistics here
        return local_jdata
