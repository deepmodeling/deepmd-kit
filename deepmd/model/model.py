from abc import (
    ABC,
    abstractmethod,
)
from enum import (
    Enum,
)
from typing import (
    List,
    Optional,
    Union,
)

from deepmd.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.utils.graph import (
    load_graph_def,
)


class Model(ABC):
    @abstractmethod
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
    ):
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
            The path to the checkpoint and meta file
        suffix : str, optional
            The suffix of the scope
        reuse : bool or tf.AUTO_REUSE, optional
            Whether to reuse the variables

        Returns
        -------
        dict
            The output dict
        """

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
        raise RuntimeError(
            "The 'dp train init-frz-model' command do not support this model!"
        )

    def build_descrpt(
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
    ):
        """Build the descriptor part of the model.

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
            The path to the checkpoint and meta file
        suffix : str, optional
            The suffix of the scope
        reuse : bool or tf.AUTO_REUSE, optional
            Whether to reuse the variables

        Returns
        -------
        tf.Tensor
            The descriptor tensor
        """
        if frz_model is None and ckpt_meta is None:
            dout = self.descrpt.build(
                coord_,
                atype_,
                natoms,
                box,
                mesh,
                input_dict,
                suffix=suffix,
                reuse=reuse,
            )
            dout = tf.identity(dout, name="o_descriptor")
        else:
            tf.constant(
                self.rcut, name="descrpt_attr/rcut", dtype=GLOBAL_TF_FLOAT_PRECISION
            )
            tf.constant(self.ntypes, name="descrpt_attr/ntypes", dtype=tf.int32)
            feed_dict = self.descrpt.get_feed_dict(coord_, atype_, natoms, box, mesh)
            return_elements = [*self.descrpt.get_tensor_names(), "o_descriptor:0"]
            if frz_model is not None:
                imported_tensors = self._import_graph_def_from_frz_model(
                    frz_model, feed_dict, return_elements
                )
            elif ckpt_meta is not None:
                imported_tensors = self._import_graph_def_from_ckpt_meta(
                    ckpt_meta, feed_dict, return_elements
                )
            else:
                raise RuntimeError("should not reach here")  # pragma: no cover
            dout = imported_tensors[-1]
            self.descrpt.pass_tensors_from_frz_model(*imported_tensors[:-1])
        return dout

    def _import_graph_def_from_frz_model(
        self, frz_model: str, feed_dict: dict, return_elements: List[str]
    ):
        return_nodes = [x[:-2] for x in return_elements]
        graph, graph_def = load_graph_def(frz_model)
        sub_graph_def = tf.graph_util.extract_sub_graph(graph_def, return_nodes)
        return tf.import_graph_def(
            sub_graph_def, input_map=feed_dict, return_elements=return_elements, name=""
        )

    def _import_graph_def_from_ckpt_meta(
        self, ckpt_meta: str, feed_dict: dict, return_elements: List[str]
    ):
        return_nodes = [x[:-2] for x in return_elements]
        with tf.Graph().as_default() as graph:
            tf.train.import_meta_graph(f"{ckpt_meta}.meta", clear_devices=True)
            graph_def = graph.as_graph_def()
        sub_graph_def = tf.graph_util.extract_sub_graph(graph_def, return_nodes)
        return tf.import_graph_def(
            sub_graph_def, input_map=feed_dict, return_elements=return_elements, name=""
        )
