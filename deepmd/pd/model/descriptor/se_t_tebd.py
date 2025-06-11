# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import paddle

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pd.model.descriptor import (
    DescriptorBlock,
)
from deepmd.pd.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pd.model.network.mlp import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.pd.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pd.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pd.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.pd.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_pair_exclude_types,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .descriptor import (
    extend_descrpt_stat,
)


@BaseDescriptor.register("se_e3_tebd")
class DescrptSeTTebd(BaseDescriptor, paddle.nn.Layer):
    r"""Construct an embedding net that takes angles between two neighboring atoms and type embeddings as input.

    Parameters
    ----------
    rcut
            The cut-off radius
    rcut_smth
            From where the environment matrix should be smoothed
    sel : Union[list[int], int]
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
    ntypes : int
            Number of element types
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net
    tebd_dim : int
            Dimension of the type embedding
    tebd_input_mode : str
            The input mode of the type embedding. Supported modes are ["concat", "strip"].
            - "concat": Concatenate the type embedding with the smoothed angular information as the union input for the embedding network.
            - "strip": Use a separated embedding network for the type embedding and combine the output with the angular embedding network output.
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    set_davg_zero
            Set the shift of embedding net input to zero.
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    env_protection: float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
    exclude_types : list[tuple[int, int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    concat_output_tebd: bool
            Whether to concat type embedding at the output of the descriptor.
    use_econf_tebd: bool, Optional
            Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, Optional
            Whether to use bias in the type embedding layer.
    smooth: bool
            Whether to use smooth process in calculation.

    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[list[int], int],
        ntypes: int,
        neuron: list = [2, 4, 8],
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        resnet_dt: bool = False,
        set_davg_zero: bool = True,
        activation_function: str = "tanh",
        env_protection: float = 0.0,
        exclude_types: list[tuple[int, int]] = [],
        precision: str = "float64",
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        type_map: Optional[list[str]] = None,
        concat_output_tebd: bool = True,
        use_econf_tebd: bool = False,
        use_tebd_bias=False,
        smooth: bool = True,
    ) -> None:
        super().__init__()
        self.se_ttebd = DescrptBlockSeTTebd(
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            tebd_dim=tebd_dim,
            tebd_input_mode=tebd_input_mode,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            exclude_types=exclude_types,
            env_protection=env_protection,
            smooth=smooth,
            seed=child_seed(seed, 1),
        )
        self.prec = PRECISION_DICT[precision]
        self.use_econf_tebd = use_econf_tebd
        self.type_map = type_map
        self.smooth = smooth
        self.type_embedding = TypeEmbedNet(
            ntypes,
            tebd_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            use_econf_tebd=use_econf_tebd,
            type_map=type_map,
            use_tebd_bias=use_tebd_bias,
        )
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.concat_output_tebd = concat_output_tebd
        self.trainable = trainable
        # set trainable
        for param in self.parameters():
            param.stop_gradient = not trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.se_ttebd.get_rcut()

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.se_ttebd.get_rcut_smth()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.se_ttebd.get_nsel()

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.se_ttebd.get_sel()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.se_ttebd.get_ntypes()

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        ret = self.se_ttebd.get_dim_out()
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        return self.se_ttebd.dim_emb

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return self.se_ttebd.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return self.se_ttebd.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return self.se_ttebd.need_sorted_nlist_for_lower()

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.se_ttebd.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        # For DPA1 descriptors, the user-defined share-level
        # shared_level: 0
        # share all parameters in both type_embedding and se_ttebd
        if shared_level == 0:
            self._sub_layers["type_embedding"] = base_class._sub_layers[
                "type_embedding"
            ]
            self.se_ttebd.share_params(base_class.se_ttebd, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding
        elif shared_level == 1:
            self._sub_layers["type_embedding"] = base_class._sub_layers[
                "type_embedding"
            ]
        # Other shared levels
        else:
            raise NotImplementedError

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ):
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        return self.se_ttebd.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: paddle.Tensor,
        stddev: paddle.Tensor,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.se_ttebd.mean = mean
        self.se_ttebd.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Get mean and stddev for descriptor."""
        return self.se_ttebd.mean, self.se_ttebd.stddev

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        obj = self.se_ttebd
        obj.ntypes = len(type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        obj.reinit_exclude(map_pair_exclude_types(obj.exclude_types, remap_index))
        if has_new_type:
            # the avg and std of new types need to be updated
            extend_descrpt_stat(
                obj,
                type_map,
                des_with_stat=model_with_new_type_stat.se_ttebd
                if model_with_new_type_stat is not None
                else None,
            )
        obj["davg"] = obj["davg"][remap_index]
        obj["dstd"] = obj["dstd"][remap_index]

    def serialize(self) -> dict:
        obj = self.se_ttebd
        data = {
            "@class": "Descriptor",
            "type": "se_e3_tebd",
            "@version": 1,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "ntypes": obj.ntypes,
            "neuron": obj.neuron,
            "tebd_dim": obj.tebd_dim,
            "tebd_input_mode": obj.tebd_input_mode,
            "set_davg_zero": obj.set_davg_zero,
            "activation_function": obj.activation_function,
            "resnet_dt": obj.resnet_dt,
            "concat_output_tebd": self.concat_output_tebd,
            "use_econf_tebd": self.use_econf_tebd,
            "type_map": self.type_map,
            # make deterministic
            "precision": RESERVED_PRECISION_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "env_mat": DPEnvMat(obj.rcut, obj.rcut_smth).serialize(),
            "type_embedding": self.type_embedding.embedding.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "smooth": self.smooth,
            "@variables": {
                "davg": obj["davg"].numpy(),
                "dstd": obj["dstd"].numpy(),
            },
            "trainable": self.trainable,
        }
        if obj.tebd_input_mode in ["strip"]:
            data.update({"embeddings_strip": obj.filter_layers_strip.serialize()})
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeTTebd":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        type_embedding = data.pop("type_embedding")
        env_mat = data.pop("env_mat")
        tebd_input_mode = data["tebd_input_mode"]
        if tebd_input_mode in ["strip"]:
            embeddings_strip = data.pop("embeddings_strip")
        else:
            embeddings_strip = None
        obj = cls(**data)

        def t_cvt(xx):
            return paddle.to_tensor(xx, dtype=obj.se_ttebd.prec).to(device=env.DEVICE)

        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )
        obj.se_ttebd["davg"] = t_cvt(variables["davg"])
        obj.se_ttebd["dstd"] = t_cvt(variables["dstd"])
        obj.se_ttebd.filter_layers = NetworkCollection.deserialize(embeddings)
        if tebd_input_mode in ["strip"]:
            obj.se_ttebd.filter_layers_strip = NetworkCollection.deserialize(
                embeddings_strip
            )
        return obj

    def forward(
        self,
        extended_coord: paddle.Tensor,
        extended_atype: paddle.Tensor,
        nlist: paddle.Tensor,
        mapping: Optional[paddle.Tensor] = None,
        comm_dict: Optional[dict[str, paddle.Tensor]] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, not required by this descriptor.
        comm_dict
            The data needed for communication for parallel inference.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        # cast the input to internal precsion
        extended_coord = extended_coord.to(dtype=self.prec)
        del mapping
        nframes, nloc, nnei = nlist.shape
        nall = extended_coord.reshape([nframes, -1]).shape[1] // 3
        g1_ext = self.type_embedding(extended_atype)
        g1_inp = g1_ext[:, :nloc, :]
        if self.tebd_input_mode in ["strip"]:
            type_embedding = self.type_embedding.get_full_embedding(g1_ext.place)
        else:
            type_embedding = None
        g1, _, _, _, sw = self.se_ttebd(
            nlist,
            extended_coord,
            extended_atype,
            g1_ext,
            mapping=None,
            type_embedding=type_embedding,
        )
        if self.concat_output_tebd:
            g1 = paddle.concat([g1, g1_inp], axis=-1)

        return (
            g1.to(dtype=env.GLOBAL_PD_FLOAT_PRECISION),
            None,
            None,
            None,
            sw.to(dtype=env.GLOBAL_PD_FLOAT_PRECISION),
        )

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
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data, type_map, local_jdata_cpy["rcut"], local_jdata_cpy["sel"], True
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist


@DescriptorBlock.register("se_ttebd")
class DescrptBlockSeTTebd(DescriptorBlock):
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[list[int], int],
        ntypes: int,
        neuron: list = [25, 50, 100],
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        set_davg_zero: bool = True,
        activation_function="tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        smooth: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        self.rcut_smth = float(rcut_smth)
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.resnet_dt = resnet_dt
        self.env_protection = env_protection
        self.seed = seed
        self.smooth = smooth

        if isinstance(sel, int):
            sel = [sel]

        self.ntypes = ntypes
        self.sel = sel
        self.sec = self.sel
        self.split_sel = self.sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = paddle.zeros(wanted_shape, dtype=env.GLOBAL_PD_FLOAT_PRECISION).to(
            device=env.DEVICE
        )
        stddev = paddle.ones(wanted_shape, dtype=env.GLOBAL_PD_FLOAT_PRECISION).to(
            device=env.DEVICE
        )
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.tebd_dim_input = self.tebd_dim * 2
        if self.tebd_input_mode in ["concat"]:
            self.embd_input_dim = 1 + self.tebd_dim_input
        else:
            self.embd_input_dim = 1

        self.filter_layers = None
        self.filter_layers_strip = None
        filter_layers = NetworkCollection(
            ndim=0, ntypes=self.ntypes, network_type="embedding_network"
        )
        filter_layers[0] = EmbeddingNet(
            self.embd_input_dim,
            self.filter_neuron,
            activation_function=self.activation_function,
            precision=self.precision,
            resnet_dt=self.resnet_dt,
            seed=child_seed(self.seed, 1),
        )
        self.filter_layers = filter_layers
        if self.tebd_input_mode in ["strip"]:
            filter_layers_strip = NetworkCollection(
                ndim=0, ntypes=self.ntypes, network_type="embedding_network"
            )
            filter_layers_strip[0] = EmbeddingNet(
                self.tebd_dim_input,
                self.filter_neuron,
                activation_function=self.activation_function,
                precision=self.precision,
                resnet_dt=self.resnet_dt,
                seed=child_seed(self.seed, 2),
            )
            self.filter_layers_strip = filter_layers_strip
        self.stats = None

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_emb(self) -> int:
        """Returns the output dimension of embedding."""
        return self.filter_neuron[-1]

    def __setitem__(self, key, value) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1]

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.tebd_dim

    @property
    def dim_emb(self):
        """Returns the output dimension of embedding."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `paddle.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        if not self.set_davg_zero:
            paddle.assign(
                paddle.to_tensor(mean, dtype=self.mean.dtype).to(device=env.DEVICE),
                self.mean,
            )  # pylint: disable=no-explicit-dtype
        paddle.assign(
            paddle.to_tensor(stddev, dtype=self.stddev.dtype).to(device=env.DEVICE),
            self.stddev,
        )  # pylint: disable=no-explicit-dtype

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def forward(
        self,
        nlist: paddle.Tensor,
        extended_coord: paddle.Tensor,
        extended_atype: paddle.Tensor,
        extended_atype_embd: Optional[paddle.Tensor] = None,
        mapping: Optional[paddle.Tensor] = None,
        type_embedding: Optional[paddle.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        extended_coord
            The extended coordinates of atoms. shape: nf x (nallx3)
        extended_atype
            The extended aotm types. shape: nf x nall x nt
        extended_atype_embd
            The extended type embedding of atoms. shape: nf x nall
        mapping
            The index mapping, not required by this descriptor.
        type_embedding
            Full type embeddings. shape: (ntypes+1) x nt
            Required for stripped type embeddings.

        Returns
        -------
        result
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        g2
            The rotationally invariant pair-partical representation.
            shape: nf x nloc x nnei x ng
        h2
            The rotationally equivariant pair-partical representation.
            shape: nf x nloc x nnei x 3
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        sw
            The smooth switch function. shape: nf x nloc x nnei

        """
        del mapping
        assert extended_atype_embd is not None
        nframes, nloc, nnei = nlist.shape
        atype = extended_atype[:, :nloc]
        nb = nframes
        nall = extended_coord.reshape([nb, -1, 3]).shape[1]
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )
        # nb x nloc x nnei
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = paddle.where(exclude_mask != 0, nlist, paddle.full_like(nlist, -1))
        nlist_mask = nlist != -1
        nlist = paddle.where(nlist == -1, paddle.zeros_like(nlist), nlist)
        sw = paddle.squeeze(sw, -1)
        # nf x nall x nt
        nt = extended_atype_embd.shape[-1]
        # beyond the cutoff sw should be 0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)
        # (nb x nloc) x nnei
        exclude_mask = exclude_mask.reshape([nb * nloc, nnei])
        assert self.filter_layers is not None
        # nfnl x nnei x 4
        dmatrix = dmatrix.reshape([-1, self.nnei, 4])
        nfnl = dmatrix.shape[0]
        # nfnl x nnei x 4
        rr = dmatrix
        rr = rr * exclude_mask[:, :, None].astype(rr.dtype)

        # nfnl x nt_i x 3
        rr_i = rr[:, :, 1:]
        # nfnl x nt_j x 3
        rr_j = rr[:, :, 1:]
        # nfnl x nt_i x nt_j
        # env_ij = paddle.einsum("ijm,ikm->ijk", rr_i, rr_j)
        env_ij = (
            # ij1m x i1km -> ijkm -> ijk
            rr_i.unsqueeze(2) * rr_j.unsqueeze(1)
        ).sum(-1)
        # nfnl x nt_i x nt_j x 1
        ss = env_ij.unsqueeze(-1)
        if self.tebd_input_mode in ["concat"]:
            atype_tebd_ext = extended_atype_embd
            # nb x (nloc x nnei) x nt
            index = nlist.reshape([nb, nloc * nnei]).unsqueeze(-1).expand([-1, -1, nt])
            # nb x (nloc x nnei) x nt
            # atype_tebd_nlist = paddle.take_along_axis(atype_tebd_ext, axis=1, index=index)
            atype_tebd_nlist = paddle.take_along_axis(
                atype_tebd_ext, axis=1, indices=index
            )
            # nb x nloc x nnei x nt
            atype_tebd_nlist = atype_tebd_nlist.reshape([nb, nloc, nnei, nt])
            # nfnl x nnei x tebd_dim
            nlist_tebd = atype_tebd_nlist.reshape([nfnl, nnei, self.tebd_dim])
            # nfnl x nt_i x nt_j x tebd_dim
            nlist_tebd_i = nlist_tebd.unsqueeze(2).expand([-1, -1, self.nnei, -1])
            nlist_tebd_j = nlist_tebd.unsqueeze(1).expand([-1, self.nnei, -1, -1])
            # nfnl x nt_i x nt_j x (1 + tebd_dim * 2)
            ss = paddle.concat([ss, nlist_tebd_i, nlist_tebd_j], axis=-1)
            # nfnl x nt_i x nt_j x ng
            gg = self.filter_layers.networks[0](ss)
        elif self.tebd_input_mode in ["strip"]:
            # nfnl x nt_i x nt_j x ng
            gg_s = self.filter_layers.networks[0](ss)
            assert self.filter_layers_strip is not None
            assert type_embedding is not None
            ng = self.filter_neuron[-1]
            ntypes_with_padding = type_embedding.shape[0]
            # nf x (nl x nnei)
            nlist_index = nlist.reshape([nb, nloc * nnei])
            # nf x (nl x nnei)
            nei_type = paddle.take_along_axis(
                extended_atype, indices=nlist_index, axis=1
            )
            # nfnl x nnei
            nei_type = nei_type.reshape([nfnl, nnei])
            # nfnl x nnei x nnei
            nei_type_i = nei_type.unsqueeze(2).expand([-1, -1, nnei])
            nei_type_j = nei_type.unsqueeze(1).expand([-1, nnei, -1])
            idx_i = nei_type_i * ntypes_with_padding
            idx_j = nei_type_j
            # (nf x nl x nt_i x nt_j) x ng
            idx = (
                (idx_i + idx_j)
                .reshape([-1, 1])
                .expand([-1, ng])
                .astype(paddle.int64)
                .to(paddle.int64)
            )
            # ntypes * (ntypes) * nt
            type_embedding_i = paddle.tile(
                type_embedding.reshape([ntypes_with_padding, 1, nt]),
                [1, ntypes_with_padding, 1],
            )
            # (ntypes) * ntypes * nt
            type_embedding_j = paddle.tile(
                type_embedding.reshape([1, ntypes_with_padding, nt]),
                [ntypes_with_padding, 1, 1],
            )
            # (ntypes * ntypes) * (nt+nt)
            two_side_type_embedding = paddle.concat(
                [type_embedding_i, type_embedding_j], -1
            ).reshape([-1, nt * 2])
            tt_full = self.filter_layers_strip.networks[0](two_side_type_embedding)
            # (nfnl x nt_i x nt_j) x ng
            gg_t = paddle.take_along_axis(tt_full, indices=idx, axis=0)
            # (nfnl x nt_i x nt_j) x ng
            gg_t = gg_t.reshape([nfnl, nnei, nnei, ng])
            if self.smooth:
                gg_t = (
                    gg_t
                    * sw.reshape([nfnl, self.nnei, 1, 1])
                    * sw.reshape([nfnl, 1, self.nnei, 1])
                )
            # nfnl x nt_i x nt_j x ng
            gg = gg_s * gg_t + gg_s
        else:
            raise NotImplementedError

        # nfnl x ng
        # res_ij = paddle.einsum("ijk,ijkm->im", env_ij, gg)
        res_ij = (
            # ijk1 x ijkm -> ijkm -> im
            env_ij.unsqueeze(-1) * gg
        ).sum([1, 2])
        res_ij = res_ij * (1.0 / float(self.nnei) / float(self.nnei))
        # nf x nl x ng
        result = res_ij.reshape([nframes, nloc, self.filter_neuron[-1]])
        return (
            result,
            None,
            None,
            None,
            sw,
        )

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return False
