from functools import lru_cache
from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Union

# from deepmd.descriptor.descriptor import (
#     Descriptor,
# )
import numpy as np

import deepmd
from deepmd.common import data_requirement
from deepmd.common import expand_sys_str
from deepmd.common import j_loader
from deepmd.common import j_must_have
from deepmd.descriptor import DescrptSeA
from deepmd.env import MODEL_VERSION
from deepmd.env import default_tf_session_config
from deepmd.env import paddle
from deepmd.env import tf
from deepmd.fit import ener
from deepmd.model import EnerModel
from deepmd.utils.argcheck import type_embedding_args
from deepmd.utils.batch_size import AutoBatchSize
from deepmd.utils.sess import run_sess
from deepmd.utils.spin import Spin

if TYPE_CHECKING:
    from pathlib import Path


def remove_comment_in_json(jdata):
    """Remove the comment in json file.
    Parameters
    ----------
    jdata : dict
        The data loaded from json file.
    Returns
    -------
    dict
        The new data without comments.
    """
    if not isinstance(jdata, dict):
        return
    find = False
    for k, v in jdata.items():
        if "_comment" == k:
            find = True
        remove_comment_in_json(v)
    if find:
        del jdata["_comment"]


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
        remove_comment_in_json(jdata)
        model_param = j_must_have(jdata, "model")
        self.multi_task_mode = "fitting_net_dict" in model_param
        descrpt_param = j_must_have(model_param, "descriptor")
        fitting_param = (
            j_must_have(model_param, "fitting_net")
            if not self.multi_task_mode
            else j_must_have(model_param, "fitting_net_dict")
        )
        typeebd_param = model_param.get("type_embedding", None)
        spin_param = model_param.get("spin", None)
        self.model_param = model_param
        self.descrpt_param = descrpt_param
        # spin
        if spin_param is not None:
            spin = Spin(
                use_spin=spin_param["use_spin"],
                virtual_len=spin_param["virtual_len"],
                spin_norm=spin_param["spin_norm"],
            )
        else:
            spin = None

        # # nvnmd
        # self.nvnmd_param = jdata.get("nvnmd", {})
        # nvnmd_cfg.init_from_jdata(self.nvnmd_param)
        # if nvnmd_cfg.enable:
        #     nvnmd_cfg.init_from_deepmd_input(model_param)
        #     nvnmd_cfg.disp_message()
        #     nvnmd_cfg.save()

        # descriptor
        try:
            descrpt_type = descrpt_param["type"]
            self.descrpt_type = descrpt_type
        except KeyError:
            raise KeyError("the type of descriptor should be set by `type`")

        explicit_ntypes_descrpt = ["se_atten"]
        hybrid_with_tebd = False
        if descrpt_param["type"] in explicit_ntypes_descrpt:
            descrpt_param["ntypes"] = len(model_param["type_map"])
        elif descrpt_param["type"] == "hybrid":
            for descrpt_item in descrpt_param["list"]:
                if descrpt_item["type"] in explicit_ntypes_descrpt:
                    descrpt_item["ntypes"] = len(model_param["type_map"])
                    hybrid_with_tebd = True
        if self.multi_task_mode:
            descrpt_param["multi_task"] = True
        if descrpt_param["type"] in ["se_e2_a", "se_a", "se_e2_r", "se_r", "hybrid"]:
            descrpt_param["spin"] = spin
        descrpt_param.pop("type")
        descrpt = deepmd.descriptor.se_a.DescrptSeA(**descrpt_param)

        # fitting net
        if not self.multi_task_mode:
            fitting_type = fitting_param.get("type", "ener")
            self.fitting_type = fitting_type
            fitting_param["descrpt"] = descrpt
            if fitting_type == "ener":
                fitting_param["spin"] = spin
                fitting_param.pop("type", None)
            fitting = ener.EnerFitting(**fitting_param)
        else:
            self.fitting_dict = {}
            self.fitting_type_dict = {}
            self.nfitting = len(fitting_param)
            for item in fitting_param:
                item_fitting_param = fitting_param[item]
                item_fitting_type = item_fitting_param.get("type", "ener")
                self.fitting_type_dict[item] = item_fitting_type
                item_fitting_param["descrpt"] = descrpt
                if item_fitting_type == "ener":
                    item_fitting_param["spin"] = spin
                # self.fitting_dict[item] = Fitting(**item_fitting_param)

        # type embedding
        padding = False
        if descrpt_type == "se_atten" or hybrid_with_tebd:
            padding = True
        if typeebd_param is not None:
            raise NotImplementedError()
            typeebd = TypeEmbedNet(
                neuron=typeebd_param["neuron"],
                resnet_dt=typeebd_param["resnet_dt"],
                activation_function=typeebd_param["activation_function"],
                precision=typeebd_param["precision"],
                trainable=typeebd_param["trainable"],
                seed=typeebd_param["seed"],
                padding=padding,
            )
        elif descrpt_type == "se_atten" or hybrid_with_tebd:
            raise NotImplementedError()
            default_args = type_embedding_args()
            default_args_dict = {i.name: i.default for i in default_args}
            typeebd = TypeEmbedNet(
                neuron=default_args_dict["neuron"],
                resnet_dt=default_args_dict["resnet_dt"],
                activation_function=None,
                precision=default_args_dict["precision"],
                trainable=default_args_dict["trainable"],
                seed=default_args_dict["seed"],
                padding=padding,
            )
        else:
            typeebd = None

        # init model
        # infer model type by fitting_type
        if not self.multi_task_mode:
            if self.fitting_type == "ener":
                self.model = EnerModel(
                    descrpt,
                    fitting,
                    typeebd,
                    model_param.get("type_map"),
                    model_param.get("data_stat_nbatch", 10),
                    model_param.get("data_stat_protect", 1e-2),
                    model_param.get("use_srtab"),
                    model_param.get("smin_alpha"),
                    model_param.get("sw_rmin"),
                    model_param.get("sw_rmax"),
                    spin,
                )
            # elif fitting_type == 'wfc':
            #     self.model = WFCModel(model_param, descrpt, fitting)
            elif self.fitting_type == "dos":
                raise NotImplementedError()
                self.model = DOSModel(
                    descrpt,
                    fitting,
                    typeebd,
                    model_param.get("type_map"),
                    model_param.get("data_stat_nbatch", 10),
                    model_param.get("data_stat_protect", 1e-2),
                )

            elif self.fitting_type == "dipole":
                raise NotImplementedError()
                self.model = DipoleModel(
                    descrpt,
                    fitting,
                    typeebd,
                    model_param.get("type_map"),
                    model_param.get("data_stat_nbatch", 10),
                    model_param.get("data_stat_protect", 1e-2),
                )
            elif self.fitting_type == "polar":
                raise NotImplementedError()
                self.model = PolarModel(
                    descrpt,
                    fitting,
                    typeebd,
                    model_param.get("type_map"),
                    model_param.get("data_stat_nbatch", 10),
                    model_param.get("data_stat_protect", 1e-2),
                )
            # elif self.fitting_type == 'global_polar':
            #     self.model = GlobalPolarModel(
            #         descrpt,
            #         fitting,
            #         model_param.get('type_map'),
            #         model_param.get('data_stat_nbatch', 10),
            #         model_param.get('data_stat_protect', 1e-2)
            #     )
            else:
                raise RuntimeError("get unknown fitting type when building model")
        else:  # multi-task mode
            raise NotImplementedError()
            self.model = MultiModel(
                descrpt,
                self.fitting_dict,
                self.fitting_type_dict,
                typeebd,
                model_param.get("type_map"),
                model_param.get("data_stat_nbatch", 10),
                model_param.get("data_stat_protect", 1e-2),
                model_param.get("use_srtab"),
                model_param.get("smin_alpha"),
                model_param.get("sw_rmin"),
                model_param.get("sw_rmax"),
            )

        # # if descrpt_param["type"] in ["se_e2_a", "se_a", "se_e2_r", "se_r", "hybrid"]:
        # descrpt_param["spin"] = None
        # descrpt_param["type_one_side"] = False

        # descrpt_param.pop("type", None)
        # descrpt_param.pop("_comment", None)
        # spin = None
        # # descrpt_param["spin"] = spin
        # descrpt = DescrptSeA(**descrpt_param)

        # self.multi_task_mode = "fitting_net_dict" in model_param
        # fitting_param = (
        #     j_must_have(model_param, "fitting_net")
        #     if not self.multi_task_mode
        #     else j_must_have(model_param, "fitting_net_dict")
        # )
        # from deepmd.fit import EnerFitting

        # # fitting_param.pop("type", None)
        # fitting_param.pop("_comment", None)
        # fitting_param["descrpt"] = descrpt
        # fitting = EnerFitting(**fitting_param)

        # typeebd = None

        # self.model = EnerModel(
        #     descrpt,
        #     fitting,
        #     typeebd,
        #     model_param.get("type_map"),
        #     model_param.get("data_stat_nbatch", 10),
        #     model_param.get("data_stat_protect", 1e-2),
        #     model_param.get("use_srtab"),
        #     model_param.get("smin_alpha"),
        #     model_param.get("sw_rmin"),
        #     model_param.get("sw_rmax"),
        #     spin,
        # )
        model_file_str = str(model_file)
        if model_file_str.endswith((".pdmodel", ".pdiparams")):
            st_model_prefix = model_file_str.rsplit(".", 1)[0]
            self.st_model = paddle.jit.load(st_model_prefix)
            print(f"==>> Load static model successfully from: {str(st_model_prefix)}")
        else:
            load_state_dict = paddle.load(str(model_file))
            for k, v in load_state_dict.items():
                if k in self.model.state_dict():
                    if load_state_dict[k].dtype != self.model.state_dict()[k].dtype:
                        print(
                            f"convert {k}'s dtype from {load_state_dict[k].dtype} to {self.model.state_dict()[k].dtype}"
                        )
                        load_state_dict[k] = load_state_dict[k].astype(
                            self.model.state_dict()[k].dtype
                        )
                    if list(load_state_dict[k].shape) != list(
                        self.model.state_dict()[k].shape
                    ):
                        print(
                            f"convert {k}'s shape from {load_state_dict[k].shape} to {self.model.state_dict()[k].shape}"
                        )
                        load_state_dict[k] = load_state_dict[k].reshape(
                            self.model.state_dict()[k].shape
                        )
            self.model.set_state_dict(load_state_dict)
            print(f"==>> Load dynamic model successfully from: {str(model_file)}")
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
