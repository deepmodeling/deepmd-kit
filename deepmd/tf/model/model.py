# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from abc import (
    ABC,
    abstractmethod,
)
from enum import (
    Enum,
)
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

from deepmd.common import (
    j_get_type,
)
from deepmd.tf.descriptor.descriptor import (
    Descriptor,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.fit.dipole import (
    DipoleFittingSeA,
)
from deepmd.tf.fit.dos import (
    DOSFitting,
)
from deepmd.tf.fit.ener import (
    EnerFitting,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.fit.polar import (
    PolarFittingSeA,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.utils.argcheck import (
    type_embedding_args,
)
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.graph import (
    load_graph_def,
)
from deepmd.tf.utils.pair_tab import (
    PairTab,
)
from deepmd.tf.utils.spin import (
    Spin,
)
from deepmd.tf.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class Model(ABC, make_plugin_registry("model")):
    """Abstract base model.

    Parameters
    ----------
    type_embedding
        Type embedding net
    type_map
        Mapping atom type to the name (str) of the type.
        For example `type_map[1]` gives the name of the type 1.
    data_stat_nbatch
        Number of frames used for data statistic
    data_bias_nsample
        The number of training samples in a system to compute and change the energy bias.
    data_stat_protect
        Protect parameter for atomic energy regression
    use_srtab
        The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.
    smin_alpha
        The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.
    sw_rmin
        The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    sw_rmin
        The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    srtab_add_bias : bool
        Whether add energy bias from the statistics of the data to short-range tabulated atomic energy. It only takes effect when `use_srtab` is provided.
    spin
        spin
    compress
        Compression information for internal use
    """

    def __new__(cls, *args, **kwargs):
        if cls is Model:
            # init model
            cls = cls.get_class_by_type(kwargs.get("type", "standard"))
            return cls.__new__(cls, *args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        type_embedding: Optional[Union[dict, TypeEmbedNet]] = None,
        type_map: Optional[List[str]] = None,
        data_stat_nbatch: int = 10,
        data_bias_nsample: int = 10,
        data_stat_protect: float = 1e-2,
        use_srtab: Optional[str] = None,
        smin_alpha: Optional[float] = None,
        sw_rmin: Optional[float] = None,
        sw_rmax: Optional[float] = None,
        srtab_add_bias: bool = True,
        spin: Optional[Spin] = None,
        compress: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # spin
        if isinstance(spin, Spin):
            self.spin = spin
        elif spin is not None:
            self.spin = Spin(**spin)
        else:
            self.spin = None
        self.compress = compress
        # other inputs
        if type_map is None:
            self.type_map = []
        else:
            self.type_map = type_map
        self.data_stat_nbatch = data_stat_nbatch
        self.data_bias_nsample = data_bias_nsample
        self.data_stat_protect = data_stat_protect
        self.srtab_name = use_srtab
        if self.srtab_name is not None:
            self.srtab = PairTab(self.srtab_name)
            self.smin_alpha = smin_alpha
            self.sw_rmin = sw_rmin
            self.sw_rmax = sw_rmax
            self.srtab_add_bias = srtab_add_bias
        else:
            self.srtab = None

    def get_type_map(self) -> list:
        """Get the type map."""
        return self.type_map

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
            The path prefix of the checkpoint and meta files
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
            dout = tf.identity(dout, name="o_descriptor" + suffix)
        else:
            tf.constant(
                self.rcut,
                name="descrpt_attr%s/rcut" % suffix,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
            )
            tf.constant(
                self.ntypes, name="descrpt_attr%s/ntypes" % suffix, dtype=tf.int32
            )
            if "global_feed_dict" in input_dict:
                feed_dict = input_dict["global_feed_dict"]
            else:
                extra_feed_dict = {}
                if "fparam" in input_dict:
                    extra_feed_dict["fparam"] = input_dict["fparam"]
                if "aparam" in input_dict:
                    extra_feed_dict["aparam"] = input_dict["aparam"]
                feed_dict = self.get_feed_dict(
                    coord_, atype_, natoms, box, mesh, **extra_feed_dict
                )
            return_elements = [
                *self.descrpt.get_tensor_names(suffix=suffix),
                "o_descriptor%s:0" % suffix,
            ]
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

    def build_type_embedding(
        self,
        ntypes: int,
        frz_model: Optional[str] = None,
        ckpt_meta: Optional[str] = None,
        suffix: str = "",
        reuse: Optional[Union[bool, Enum]] = None,
    ) -> tf.Tensor:
        """Build the type embedding part of the model.

        Parameters
        ----------
        ntypes : int
            The number of types
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
        tf.Tensor
            The type embedding tensor
        """
        assert self.typeebd is not None
        if frz_model is None and ckpt_meta is None:
            dout = self.typeebd.build(
                ntypes,
                reuse=reuse,
                suffix=suffix,
            )
        else:
            # nothing input
            feed_dict = {}
            return_elements = [
                f"t_typeebd{suffix}:0",
            ]
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

    def enable_mixed_precision(self, mixed_prec: dict):
        """Enable mixed precision for the model.

        Parameters
        ----------
        mixed_prec : dict
            The mixed precision config
        """
        raise RuntimeError("Not supported")

    def change_energy_bias(
        self,
        data: DeepmdDataSystem,
        frozen_model: str,
        origin_type_map: list,
        full_type_map: str,
        bias_shift: str = "delta",
    ) -> None:
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        data : DeepmdDataSystem
            The training data.
        frozen_model : str
            The path file of frozen model.
        origin_type_map : list
            The original type_map in dataset, they are targets to change the energy bias.
        full_type_map : str
            The full type_map in pretrained model
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        """
        raise RuntimeError("Not supported")

    def enable_compression(self, suffix: str = ""):
        """Enable compression.

        Parameters
        ----------
        suffix : str
            suffix to name scope
        """
        raise RuntimeError("Not supported")

    def get_numb_fparam(self) -> Union[int, dict]:
        """Get the number of frame parameters."""
        return 0

    def get_numb_aparam(self) -> Union[int, dict]:
        """Get the number of atomic parameters."""
        return 0

    def get_numb_dos(self) -> Union[int, dict]:
        """Get the number of gridpoints in energy space."""
        return 0

    @abstractmethod
    def get_fitting(self) -> Union[Fitting, dict]:
        """Get the fitting(s)."""

    @abstractmethod
    def get_loss(self, loss: dict, lr) -> Optional[Union[Loss, dict]]:
        """Get the loss function(s)."""

    @abstractmethod
    def get_rcut(self) -> float:
        """Get cutoff radius of the model."""

    @abstractmethod
    def get_ntypes(self) -> int:
        """Get the number of types."""

    @abstractmethod
    def data_stat(self, data: dict):
        """Data staticis."""

    def get_feed_dict(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box: tf.Tensor,
        mesh: tf.Tensor,
        **kwargs,
    ) -> Dict[str, tf.Tensor]:
        """Generate the feed_dict for current descriptor.

        Parameters
        ----------
        coord_ : tf.Tensor
            The coordinate of atoms
        atype_ : tf.Tensor
            The type of atoms
        natoms : tf.Tensor
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box : tf.Tensor
            The box. Can be generated by deepmd.tf.model.make_stat_input
        mesh : tf.Tensor
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        **kwargs : dict
            The additional arguments

        Returns
        -------
        feed_dict : dict[str, tf.Tensor]
            The output feed_dict of current descriptor
        """
        feed_dict = {
            "t_coord:0": coord_,
            "t_type:0": atype_,
            "t_natoms:0": natoms,
            "t_box:0": box,
            "t_mesh:0": mesh,
        }
        if kwargs.get("fparam") is not None:
            feed_dict["t_fparam:0"] = kwargs["fparam"]
        if kwargs.get("aparam") is not None:
            feed_dict["t_aparam:0"] = kwargs["aparam"]
        return feed_dict

    @classmethod
    @abstractmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict) -> dict:
        """Update the selection and perform neighbor statistics.

        Notes
        -----
        Do not modify the input data without copying it.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        """
        cls = cls.get_class_by_type(local_jdata.get("type", "standard"))
        return cls.update_sel(global_jdata, local_jdata)

    @classmethod
    def deserialize(cls, data: dict, suffix: str = "") -> "Model":
        """Deserialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Parameters
        ----------
        data : dict
            The serialized data
        suffix : str, optional
            Name suffix to identify this model

        Returns
        -------
        Model
            The deserialized Model
        """
        if cls is Model:
            return Model.get_class_by_type(data.get("type", "standard")).deserialize(
                data
            )
        raise NotImplementedError("Not implemented in class %s" % cls.__name__)

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Returns
        -------
        dict
            The serialized data
        suffix : str, optional
            Name suffix to identify this descriptor
        """
        raise NotImplementedError("Not implemented in class %s" % self.__name__)


@Model.register("standard")
class StandardModel(Model):
    """Standard model, which must contain a descriptor and a fitting.

    Parameters
    ----------
    descriptor : Union[dict, Descriptor]
        The descriptor
    fitting_net : Union[dict, Fitting]
        The fitting network
    type_embedding : dict, optional
        The type embedding
    type_map : list of dict, optional
        The type map
    """

    def __new__(cls, *args, **kwargs):
        from .dos import (
            DOSModel,
        )
        from .ener import (
            EnerModel,
        )
        from .tensor import (
            DipoleModel,
            PolarModel,
        )

        if cls is StandardModel:
            if isinstance(kwargs["fitting_net"], dict):
                fitting_type = Fitting.get_class_by_type(
                    j_get_type(kwargs["fitting_net"], cls.__name__)
                )
            elif isinstance(kwargs["fitting_net"], Fitting):
                fitting_type = type(kwargs["fitting_net"])
            else:
                raise RuntimeError("get unknown fitting type when building model")
            # init model
            # infer model type by fitting_type
            if issubclass(fitting_type, EnerFitting):
                cls = EnerModel
            elif issubclass(fitting_type, DOSFitting):
                cls = DOSModel
            elif issubclass(fitting_type, DipoleFittingSeA):
                cls = DipoleModel
            elif issubclass(fitting_type, PolarFittingSeA):
                cls = PolarModel
            else:
                raise RuntimeError("get unknown fitting type when building model")
            return cls.__new__(cls)
        return super().__new__(cls)

    def __init__(
        self,
        descriptor: Union[dict, Descriptor],
        fitting_net: Union[dict, Fitting],
        type_embedding: Optional[Union[dict, TypeEmbedNet]] = None,
        type_map: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            descriptor=descriptor, fitting=fitting_net, type_map=type_map, **kwargs
        )
        if isinstance(descriptor, Descriptor):
            self.descrpt = descriptor
        else:
            self.descrpt = Descriptor(
                **descriptor, ntypes=len(self.get_type_map()), spin=self.spin
            )

        if isinstance(fitting_net, Fitting):
            self.fitting = fitting_net
        else:
            if fitting_net["type"] in ["dipole", "polar"]:
                fitting_net["embedding_width"] = self.descrpt.get_dim_rot_mat_1()
            self.fitting = Fitting(
                **fitting_net,
                descrpt=self.descrpt,
                spin=self.spin,
                ntypes=self.descrpt.get_ntypes(),
                dim_descrpt=self.descrpt.get_dim_out(),
            )
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()

        # type embedding
        if type_embedding is not None and isinstance(type_embedding, TypeEmbedNet):
            self.typeebd = type_embedding
        elif type_embedding is not None:
            self.typeebd = TypeEmbedNet(
                **type_embedding,
                padding=self.descrpt.explicit_ntypes,
            )
        elif self.descrpt.explicit_ntypes:
            default_args = type_embedding_args()
            default_args_dict = {i.name: i.default for i in default_args}
            default_args_dict["activation_function"] = None
            self.typeebd = TypeEmbedNet(
                **default_args_dict,
                padding=True,
            )
        else:
            self.typeebd = None

    def enable_mixed_precision(self, mixed_prec: dict):
        """Enable mixed precision for the model.

        Parameters
        ----------
        mixed_prec : dict
            The mixed precision config
        """
        self.descrpt.enable_mixed_precision(mixed_prec)
        self.fitting.enable_mixed_precision(mixed_prec)

    def enable_compression(self, suffix: str = ""):
        """Enable compression.

        Parameters
        ----------
        suffix : str
            suffix to name scope
        """
        graph, graph_def = load_graph_def(self.compress["model_file"])
        self.descrpt.enable_compression(
            self.compress["min_nbor_dist"],
            graph,
            graph_def,
            self.compress["table_config"][0],
            self.compress["table_config"][1],
            self.compress["table_config"][2],
            self.compress["table_config"][3],
            suffix=suffix,
        )
        # for fparam or aparam settings in 'ener' type fitting net
        self.fitting.init_variables(graph, graph_def, suffix=suffix)
        if (
            self.typeebd is not None
            and self.typeebd.type_embedding_net_variables is None
        ):
            self.typeebd.init_variables(graph, graph_def, suffix=suffix)

    def get_fitting(self) -> Union[Fitting, dict]:
        """Get the fitting(s)."""
        return self.fitting

    def get_loss(self, loss: dict, lr) -> Union[Loss, dict]:
        """Get the loss function(s)."""
        return self.fitting.get_loss(loss, lr)

    def get_rcut(self) -> float:
        """Get cutoff radius of the model."""
        return self.rcut

    def get_ntypes(self) -> int:
        """Get the number of types."""
        return self.ntypes

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
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["descriptor"] = Descriptor.update_sel(
            global_jdata, local_jdata["descriptor"]
        )
        return local_jdata_cpy

    @classmethod
    def deserialize(cls, data: dict, suffix: str = "") -> "Descriptor":
        """Deserialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Parameters
        ----------
        data : dict
            The serialized data
        suffix : str, optional
            Name suffix to identify this descriptor

        Returns
        -------
        Descriptor
            The deserialized descriptor
        """
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        descriptor = Descriptor.deserialize(data.pop("descriptor"), suffix=suffix)
        fitting = Fitting.deserialize(data.pop("fitting"), suffix=suffix)
        data.pop("atom_exclude_types")
        data.pop("pair_exclude_types")
        return cls(
            descriptor=descriptor,
            fitting_net=fitting,
            **data,
        )

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        There is no suffix in a native DP model, but it is important
        for the TF backend.

        Returns
        -------
        dict
            The serialized data
        suffix : str, optional
            Name suffix to identify this descriptor
        """
        if self.typeebd is not None:
            raise NotImplementedError("type embedding is not supported")
        if self.spin is not None:
            raise NotImplementedError("spin is not supported")
        return {
            "@class": "Model",
            "type": "standard",
            "@version": 1,
            "type_map": self.type_map,
            "descriptor": self.descrpt.serialize(suffix=suffix),
            "fitting": self.fitting.serialize(suffix=suffix),
            # not supported yet
            "atom_exclude_types": [],
            "pair_exclude_types": [],
        }
