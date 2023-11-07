from functools import lru_cache
from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Union

# from deepmd.descriptor.descriptor import (
#     Descriptor,
# )
import numpy as np

from deepmd.common import data_requirement
from deepmd.common import expand_sys_str
from deepmd.common import j_loader
from deepmd.common import j_must_have
from deepmd.env import MODEL_VERSION
from deepmd.env import default_tf_session_config
from deepmd.env import paddle
from deepmd.env import tf
from deepmd.model import EnerModel
from deepmd.utils.batch_size import AutoBatchSize
from deepmd.utils.sess import run_sess

if TYPE_CHECKING:
    from pathlib import Path


class DeepEval:
    """Common methods for DeepPot, DeepWFC, DeepPolar, ...

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation
    auto_batch_size : bool or int or AutomaticBatchSize, default: False
        If True, automatic batch size will be used. If int, it will be used
        as the initial batch size.
    """

    load_prefix: str  # set by subclass

    def __init__(
        self,
        model_file: "Path",
        load_prefix: str = "load",
        default_tf_graph: bool = False,
        auto_batch_size: Union[bool, int, AutoBatchSize] = False,
    ):
        jdata = j_loader("input.json")
        model_param = j_must_have(jdata, "model")

        descrpt_param = j_must_have(model_param, "descriptor")
        from deepmd.descriptor import DescrptSeA

        descrpt_param.pop("type", None)
        descrpt_param.pop("_comment", None)
        self.spin = None
        descrpt_param["spin"] = self.spin
        self.descrpt = DescrptSeA(**descrpt_param)

        self.multi_task_mode = "fitting_net_dict" in model_param
        fitting_param = (
            j_must_have(model_param, "fitting_net")
            if not self.multi_task_mode
            else j_must_have(model_param, "fitting_net_dict")
        )
        from deepmd.fit import EnerFitting

        # fitting_param.pop("type", None)
        fitting_param.pop("_comment", None)
        fitting_param["descrpt"] = self.descrpt
        self.fitting = EnerFitting(**fitting_param)

        self.typeebd = None

        self.model = EnerModel(
            self.descrpt,
            self.fitting,
            self.typeebd,
            model_param.get("type_map"),
            model_param.get("data_stat_nbatch", 10),
            model_param.get("data_stat_protect", 1e-2),
            model_param.get("use_srtab"),
            model_param.get("smin_alpha"),
            model_param.get("sw_rmin"),
            model_param.get("sw_rmax"),
            self.spin,
        )
        load_state_dict = paddle.load(str(model_file))
        for k, v in load_state_dict.items():
            if k in self.model.state_dict():
                if load_state_dict[k].dtype != self.model.state_dict()[k].dtype:
                    # print(f"convert dtype from {load_state_dict[k].dtype} to {self.model.state_dict()[k].dtype}")
                    load_state_dict[k] = load_state_dict[k].astype(
                        self.model.state_dict()[k].dtype
                    )
                if list(load_state_dict[k].shape) != list(
                    self.model.state_dict()[k].shape
                ):
                    # print(f"convert shape from {load_state_dict[k].shape} to {self.model.state_dict()[k].shape}")
                    load_state_dict[k] = load_state_dict[k].reshape(
                        self.model.state_dict()[k].shape
                    )
        self.model.set_state_dict(load_state_dict)
        self.load_prefix = load_prefix

        # graph_compatable should be called after graph and prefix are set
        # if not self._graph_compatable():
        #     raise RuntimeError(
        #         f"model in graph (version {self.model_version}) is incompatible"
        #         f"with the model (version {MODEL_VERSION}) supported by the current code."
        #         "See https://deepmd.rtfd.io/compatability/ for details."
        #     )

        # set default to False, as subclasses may not support
        if isinstance(auto_batch_size, bool):
            if auto_batch_size:
                self.auto_batch_size = AutoBatchSize()
            else:
                self.auto_batch_size = None
        elif isinstance(auto_batch_size, int):
            self.auto_batch_size = AutoBatchSize(auto_batch_size)
        elif isinstance(auto_batch_size, AutoBatchSize):
            self.auto_batch_size = auto_batch_size
        else:
            raise TypeError("auto_batch_size should be bool, int, or AutoBatchSize")

    @property
    @lru_cache(maxsize=None)
    def model_type(self) -> str:
        return "ener"
        """Get type of model.

        :type:str
        """
        # t_mt = self._get_tensor("model_attr/model_type:0")
        # [mt] = run_sess(self.sess, [t_mt], feed_dict={})
        # return mt.decode("utf-8")
        self._model_type = self.model.t_mt

    @property
    @lru_cache(maxsize=None)
    def model_version(self) -> str:
        """Get version of model.

        Returns
        -------
        str
            version of model
        """
        return "0.1.0"
        try:
            t_mt = self._get_tensor("model_attr/model_version:0")
        except KeyError:
            # For deepmd-kit version 0.x - 1.x, set model version to 0.0
            return "0.0"
        else:
            [mt] = run_sess(self.sess, [t_mt], feed_dict={})
            return mt.decode("utf-8")

    @property
    @lru_cache(maxsize=None)
    def sess(self) -> tf.Session:
        """Get TF session."""
        # start a tf session associated to the graph
        return tf.Session(graph=self.graph, config=default_tf_session_config)

    def _graph_compatable(self) -> bool:
        return True
        """Check the model compatability.

        Returns
        -------
        bool
            If the model stored in the graph file is compatable with the current code
        """
        model_version_major = int(self.model_version.split(".")[0])
        model_version_minor = int(self.model_version.split(".")[1])
        MODEL_VERSION_MAJOR = int(MODEL_VERSION.split(".")[0])
        MODEL_VERSION_MINOR = int(MODEL_VERSION.split(".")[1])
        if (model_version_major != MODEL_VERSION_MAJOR) or (
            model_version_minor > MODEL_VERSION_MINOR
        ):
            return False
        else:
            return True

    def _get_value(
        self, tensor_name: str, attr_name: Optional[str] = None
    ) -> tf.Tensor:
        """Get TF graph tensor and assign it to class namespace.

        Parameters
        ----------
        tensor_name : str
            name of tensor to get
        attr_name : Optional[str], optional
            if specified, class attribute with this name will be created and tensor will
            be assigned to it, by default None

        Returns
        -------
        tf.Tensor
            loaded tensor
        """
        # do not use os.path.join as it doesn't work on Windows
        value = None
        for name, tensor in self.model.named_buffers():
            if tensor_name in name:
                value = tensor.numpy()[0] if tensor.shape == [1] else tensor.numpy()
        if attr_name:
            setattr(self, attr_name, tensor)
            return tensor
        else:
            return tensor

    @staticmethod
    def _load_graph(
        frozen_graph_filename: "Path",
        prefix: str = "load",
        default_tf_graph: bool = False,
    ):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(str(frozen_graph_filename), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            if default_tf_graph:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name=prefix,
                    producer_op_list=None,
                )
                graph = tf.get_default_graph()
            else:
                # Then, we can use again a convenient built-in function to import
                # a graph_def into the  current default Graph
                with tf.Graph().as_default() as graph:
                    tf.import_graph_def(
                        graph_def,
                        input_map=None,
                        return_elements=None,
                        name=prefix,
                        producer_op_list=None,
                    )
                #     with tf.Session() as sess:
                #         constant_ops = [op for op in graph.get_operations() if op.type == "Const"]
                #         for constant_op in constant_ops:
                #             param = sess.run(constant_op.outputs[0])
                #             # print(type(param))
                #             if hasattr(param, 'shape'):
                #                 # print(param.shape)
                #                 if param.shape == (2,):
                #                     print(constant_op.outputs[0], param)
                # exit()

            return graph

    @staticmethod
    def sort_input(
        coord: np.ndarray,
        atom_type: np.ndarray,
        sel_atoms: Optional[List[int]] = None,
        mixed_type: bool = False,
    ):
        """Sort atoms in the system according their types.

        Parameters
        ----------
        coord
            The coordinates of atoms.
            Should be of shape [nframes, natoms, 3]
        atom_type
            The type of atoms
            Should be of shape [natoms]
        sel_atoms
            The selected atoms by type
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        coord_out
            The coordinates after sorting
        atom_type_out
            The atom types after sorting
        idx_map
            The index mapping from the input to the output.
            For example coord_out = coord[:,idx_map,:]
        sel_atom_type
            Only output if sel_atoms is not None
            The sorted selected atom types
        sel_idx_map
            Only output if sel_atoms is not None
            The index mapping from the selected atoms to sorted selected atoms.
        """
        if mixed_type:
            # mixed_type need not to resort
            natoms = atom_type[0].size
            idx_map = np.arange(natoms)
            return coord, atom_type, idx_map
        if sel_atoms is not None:
            selection = [False] * np.size(atom_type)
            for ii in sel_atoms:
                selection += atom_type == ii
            sel_atom_type = atom_type[selection]
        natoms = atom_type.size
        idx = np.arange(natoms)
        idx_map = np.lexsort((idx, atom_type))
        nframes = coord.shape[0]
        coord = coord.reshape([nframes, -1, 3])
        coord = np.reshape(coord[:, idx_map, :], [nframes, -1])
        atom_type = atom_type[idx_map]
        if sel_atoms is not None:
            sel_natoms = np.size(sel_atom_type)
            sel_idx = np.arange(sel_natoms)
            sel_idx_map = np.lexsort((sel_idx, sel_atom_type))
            sel_atom_type = sel_atom_type[sel_idx_map]
            return coord, atom_type, idx_map, sel_atom_type, sel_idx_map
        else:
            return coord, atom_type, idx_map

    @staticmethod
    def reverse_map(vec: np.ndarray, imap: List[int]) -> np.ndarray:
        """Reverse mapping of a vector according to the index map.

        Parameters
        ----------
        vec
            Input vector. Be of shape [nframes, natoms, -1]
        imap
            Index map. Be of shape [natoms]

        Returns
        -------
        vec_out
            Reverse mapped vector.
        """
        ret = np.zeros(vec.shape)
        # for idx,ii in enumerate(imap) :
        #     ret[:,ii,:] = vec[:,idx,:]
        ret[:, imap, :] = vec
        return ret

    def make_natoms_vec(
        self, atom_types: np.ndarray, mixed_type: bool = False
    ) -> np.ndarray:
        """Make the natom vector used by deepmd-kit.

        Parameters
        ----------
        atom_types
            The type of atoms
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.

        Returns
        -------
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms

        """
        natoms_vec = np.zeros(self.ntypes + 2).astype(int)
        if mixed_type:
            natoms = atom_types[0].size
        else:
            natoms = atom_types.size
        natoms_vec[0] = natoms
        natoms_vec[1] = natoms
        if mixed_type:
            natoms_vec[2] = natoms
            return natoms_vec
        for ii in range(self.ntypes):
            natoms_vec[ii + 2] = np.count_nonzero(atom_types == ii)
        return natoms_vec

    def eval_typeebd(self) -> np.ndarray:
        """Evaluate output of type embedding network by using this model.

        Returns
        -------
        np.ndarray
            The output of type embedding network. The shape is [ntypes, o_size],
            where ntypes is the number of types, and o_size is the number of nodes
            in the output layer.

        Raises
        ------
        KeyError
            If the model does not enable type embedding.

        See Also
        --------
        deepmd.utils.type_embed.TypeEmbedNet : The type embedding network.

        Examples
        --------
        Get the output of type embedding network of `graph.pb`:

        >>> from deepmd.infer import DeepPotential
        >>> dp = DeepPotential('graph.pb')
        >>> dp.eval_typeebd()
        """
        t_typeebd = self._get_tensor("t_typeebd:0")
        [typeebd] = run_sess(self.sess, [t_typeebd], feed_dict={})
        return typeebd
