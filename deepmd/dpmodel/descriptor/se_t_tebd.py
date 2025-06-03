# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    NoReturn,
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_take_along_axis,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EmbeddingNet,
    EnvMat,
    NetworkCollection,
    PairExcludeMask,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.dpmodel.utils.update_sel import (
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
    DescriptorBlock,
    extend_descrpt_stat,
)


@BaseDescriptor.register("se_e3_tebd")
class DescrptSeTTebd(NativeOP, BaseDescriptor):
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
            seed=child_seed(seed, 0),
        )
        self.use_econf_tebd = use_econf_tebd
        self.type_map = type_map
        self.smooth = smooth
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            seed=child_seed(seed, 1),
        )
        self.tebd_dim = tebd_dim
        self.concat_output_tebd = concat_output_tebd
        self.trainable = trainable
        self.precision = precision

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

    def share_params(self, base_class, shared_level, resume=False) -> NoReturn:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
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
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
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
        mean: np.ndarray,
        stddev: np.ndarray,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.se_ttebd.mean = mean
        self.se_ttebd.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[np.ndarray, np.ndarray]:
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

    @cast_precision
    def call(
        self,
        coord_ext,
        atype_ext,
        nlist,
        mapping: Optional[np.ndarray] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nallx3)
        atype_ext
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping from extended to local region. not used by this descriptor.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            this descriptor returns None
        h2
            The rotationally equivariant pair-partical representation.
            this descriptor returns None
        sw
            The smooth switch function.
        """
        xp = array_api_compat.array_namespace(nlist, coord_ext, atype_ext)
        del mapping
        nf, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nf, -1)).shape[1] // 3
        type_embedding = self.type_embedding.call()
        # nf x nall x tebd_dim
        atype_embd_ext = xp.reshape(
            xp.take(type_embedding, xp.reshape(atype_ext, [-1]), axis=0),
            (nf, nall, self.tebd_dim),
        )
        # nfnl x tebd_dim
        atype_embd = atype_embd_ext[:, :nloc, :]
        grrg, g2, h2, rot_mat, sw = self.se_ttebd(
            nlist,
            coord_ext,
            atype_ext,
            atype_embd_ext,
            mapping=None,
            type_embedding=type_embedding,
        )
        # nf x nloc x (ng + tebd_dim)
        if self.concat_output_tebd:
            grrg = xp.concat(
                [grrg, xp.reshape(atype_embd, (nf, nloc, self.tebd_dim))], axis=-1
            )
        return grrg, rot_mat, None, None, sw

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
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
            "precision": np.dtype(PRECISION_DICT[obj.precision]).name,
            "embeddings": obj.embeddings.serialize(),
            "env_mat": obj.env_mat.serialize(),
            "type_embedding": self.type_embedding.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "smooth": self.smooth,
            "@variables": {
                "davg": to_numpy_array(obj["davg"]),
                "dstd": to_numpy_array(obj["dstd"]),
            },
            "trainable": self.trainable,
        }
        if obj.tebd_input_mode in ["strip"]:
            data.update({"embeddings_strip": obj.embeddings_strip.serialize()})
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeTTebd":
        """Deserialize from dict."""
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

        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)
        obj.se_ttebd["davg"] = variables["davg"]
        obj.se_ttebd["dstd"] = variables["dstd"]
        obj.se_ttebd.embeddings = NetworkCollection.deserialize(embeddings)
        if tebd_input_mode in ["strip"]:
            obj.se_ttebd.embeddings_strip = NetworkCollection.deserialize(
                embeddings_strip
            )

        return obj

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
class DescrptBlockSeTTebd(NativeOP, DescriptorBlock):
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
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
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

        self.tebd_dim_input = self.tebd_dim * 2
        if self.tebd_input_mode in ["concat"]:
            self.embd_input_dim = 1 + self.tebd_dim_input
        else:
            self.embd_input_dim = 1

        embeddings = NetworkCollection(
            ndim=0,
            ntypes=self.ntypes,
            network_type="embedding_network",
        )
        embeddings[0] = EmbeddingNet(
            self.embd_input_dim,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
            seed=child_seed(seed, 0),
        )
        self.embeddings = embeddings
        if self.tebd_input_mode in ["strip"]:
            embeddings_strip = NetworkCollection(
                ndim=0,
                ntypes=self.ntypes,
                network_type="embedding_network",
            )
            embeddings_strip[0] = EmbeddingNet(
                self.tebd_dim_input,
                self.neuron,
                self.activation_function,
                self.resnet_dt,
                self.precision,
                seed=child_seed(seed, 1),
            )
            self.embeddings_strip = embeddings_strip
        else:
            self.embeddings_strip = None

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.env_mat = EnvMat(self.rcut, self.rcut_smth, protection=self.env_protection)
        self.mean = np.zeros(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.orig_sel = self.sel

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
        xp = array_api_compat.array_namespace(self.stddev)
        if not self.set_davg_zero:
            self.mean = xp.asarray(mean, dtype=self.mean.dtype, copy=True)
        self.stddev = xp.asarray(stddev, dtype=self.stddev.dtype, copy=True)

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

    def cal_g(
        self,
        ss,
        embedding_idx,
    ):
        # nfnl x nt_i x nt_j x ng
        gg = self.embeddings[embedding_idx].call(ss)
        return gg

    def cal_g_strip(
        self,
        ss,
        embedding_idx,
    ):
        assert self.embeddings_strip is not None
        # nfnl x nt_i x nt_j x ng
        gg = self.embeddings_strip[embedding_idx].call(ss)
        return gg

    def call(
        self,
        nlist: np.ndarray,
        coord_ext: np.ndarray,
        atype_ext: np.ndarray,
        atype_embd_ext: Optional[np.ndarray] = None,
        mapping: Optional[np.ndarray] = None,
        type_embedding: Optional[np.ndarray] = None,
    ):
        xp = array_api_compat.array_namespace(nlist, coord_ext, atype_ext)
        # nf x nloc x nnei x 4
        dmatrix, diff, sw = self.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.mean[...],
            self.stddev[...],
        )
        nf, nloc, nnei, _ = dmatrix.shape
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        # nfnl x nnei
        exclude_mask = xp.reshape(exclude_mask, (nf * nloc, nnei))
        # nfnl x nnei
        nlist = xp.reshape(nlist, (nf * nloc, nnei))
        exclude_mask = xp.astype(exclude_mask, xp.bool)
        nlist = xp.where(exclude_mask, nlist, xp.full_like(nlist, -1))
        # nfnl x nnei
        nlist_mask = nlist != -1
        # nfnl x nnei x 1
        sw = xp.where(
            nlist_mask[:, :, None],
            xp.reshape(sw, (nf * nloc, nnei, 1)),
            xp.zeros((nf * nloc, nnei, 1), dtype=sw.dtype),
        )

        # nfnl x nnei x 4
        dmatrix = xp.reshape(dmatrix, (nf * nloc, nnei, 4))
        # nfnl x nnei x 4
        rr = dmatrix
        rr = rr * xp.astype(exclude_mask[:, :, None], rr.dtype)
        # nfnl x nt_i x 3
        rr_i = rr[:, :, 1:]
        # nfnl x nt_j x 3
        rr_j = rr[:, :, 1:]
        # nfnl x nt_i x nt_j
        # env_ij = np.einsum("ijm,ikm->ijk", rr_i, rr_j)
        env_ij = xp.sum(rr_i[:, :, None, :] * rr_j[:, None, :, :], axis=-1)
        # nfnl x nt_i x nt_j x 1
        ss = env_ij[..., None]
        nlist_masked = xp.where(nlist_mask, nlist, xp.zeros_like(nlist))
        ng = self.neuron[-1]
        nt = self.tebd_dim

        if self.tebd_input_mode in ["concat"]:
            index = xp.tile(
                xp.reshape(nlist_masked, (nf, -1, 1)), (1, 1, self.tebd_dim)
            )
            # nfnl x nnei x tebd_dim
            atype_embd_nlist = xp_take_along_axis(atype_embd_ext, index, axis=1)
            atype_embd_nlist = xp.reshape(
                atype_embd_nlist, (nf * nloc, nnei, self.tebd_dim)
            )
            # nfnl x nt_i x nt_j x tebd_dim
            nlist_tebd_i = xp.tile(
                atype_embd_nlist[:, :, None, :], (1, 1, self.nnei, 1)
            )
            nlist_tebd_j = xp.tile(
                atype_embd_nlist[:, None, :, :], (1, self.nnei, 1, 1)
            )
            # nfnl x nt_i x nt_j x (1 + tebd_dim * 2)
            ss = xp.concat([ss, nlist_tebd_i, nlist_tebd_j], axis=-1)
            # nfnl x nt_i x nt_j x ng
            gg = self.cal_g(ss, 0)
        elif self.tebd_input_mode in ["strip"]:
            # nfnl x nt_i x nt_j x ng
            gg_s = self.cal_g(ss, 0)
            assert self.embeddings_strip is not None
            assert type_embedding is not None
            ntypes_with_padding = type_embedding.shape[0]
            # nf x (nl x nnei)
            nlist_index = xp.reshape(nlist_masked, (nf, nloc * nnei))
            # nf x (nl x nnei)
            nei_type = xp_take_along_axis(atype_ext, nlist_index, axis=1)
            # nfnl x nnei
            nei_type = xp.reshape(nei_type, (nf * nloc, nnei))

            # nfnl x nnei x nnei
            nei_type_i = xp.tile(nei_type[:, :, np.newaxis], (1, 1, nnei))
            nei_type_j = xp.tile(nei_type[:, np.newaxis, :], (1, nnei, 1))

            idx_i = nei_type_i * ntypes_with_padding
            idx_j = nei_type_j

            # (nf x nl x nt_i x nt_j) x ng
            idx = xp.tile(xp.reshape((idx_i + idx_j), (-1, 1)), (1, ng))

            # ntypes * (ntypes) * nt
            type_embedding_i = xp.tile(
                xp.reshape(type_embedding, (ntypes_with_padding, 1, nt)),
                (1, ntypes_with_padding, 1),
            )

            # (ntypes) * ntypes * nt
            type_embedding_j = xp.tile(
                xp.reshape(type_embedding, (1, ntypes_with_padding, nt)),
                (ntypes_with_padding, 1, 1),
            )

            # (ntypes * ntypes) * (nt+nt)
            two_side_type_embedding = xp.reshape(
                xp.concat([type_embedding_i, type_embedding_j], axis=-1), (-1, nt * 2)
            )
            tt_full = self.cal_g_strip(two_side_type_embedding, 0)

            # (nfnl x nt_i x nt_j) x ng
            gg_t = xp_take_along_axis(tt_full, idx, axis=0)

            # (nfnl x nt_i x nt_j) x ng
            gg_t = xp.reshape(gg_t, (nf * nloc, nnei, nnei, ng))
            if self.smooth:
                gg_t = (
                    gg_t
                    * xp.reshape(sw, (nf * nloc, self.nnei, 1, 1))
                    * xp.reshape(sw, (nf * nloc, 1, self.nnei, 1))
                )
            # nfnl x nt_i x nt_j x ng
            gg = gg_s * gg_t + gg_s
        else:
            raise NotImplementedError

        # nfnl x ng
        # res_ij = np.einsum("ijk,ijkm->im", env_ij, gg)
        res_ij = xp.sum(env_ij[:, :, :, None] * gg[:, :, :, :], axis=(1, 2))
        res_ij = res_ij * (1.0 / float(self.nnei) / float(self.nnei))
        # nf x nl x ng
        result = xp.reshape(res_ij, (nf, nloc, self.filter_neuron[-1]))
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

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        obj = self
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
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[obj.precision]).name,
            "embeddings": obj.embeddings.serialize(),
            "env_mat": obj.env_mat.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "smooth": obj.smooth,
            "@variables": {
                "davg": to_numpy_array(obj["davg"]),
                "dstd": to_numpy_array(obj["dstd"]),
            },
        }
        if obj.tebd_input_mode in ["strip"]:
            data.update({"embeddings_strip": obj.embeddings_strip.serialize()})
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeTTebd":
        """Deserialize from dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        tebd_input_mode = data["tebd_input_mode"]
        if tebd_input_mode in ["strip"]:
            embeddings_strip = data.pop("embeddings_strip")
        else:
            embeddings_strip = None
        se_ttebd = cls(**data)

        se_ttebd["davg"] = variables["davg"]
        se_ttebd["dstd"] = variables["dstd"]
        se_ttebd.embeddings = NetworkCollection.deserialize(embeddings)
        if tebd_input_mode in ["strip"]:
            se_ttebd.embeddings_strip = NetworkCollection.deserialize(embeddings_strip)

        return se_ttebd
