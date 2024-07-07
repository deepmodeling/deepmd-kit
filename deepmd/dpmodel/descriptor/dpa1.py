# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.utils import (
    EmbeddingNet,
    EnvMat,
    NetworkCollection,
    PairExcludeMask,
)
from deepmd.dpmodel.utils.network import (
    LayerNorm,
    NativeLayer,
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
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
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


def np_softmax(x, axis=-1):
    x = np.nan_to_num(x)  # to avoid value warning
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def np_normalize(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


@BaseDescriptor.register("se_atten")
@BaseDescriptor.register("dpa1")
class DescrptDPA1(NativeOP, BaseDescriptor):
    r"""Attention-based descriptor which is proposed in the pretrainable DPA-1[1] model.

    This descriptor, :math:`\mathcal{D}^i \in \mathbb{R}^{M \times M_{<}}`, is given by

    .. math::
        \mathcal{D}^i = \frac{1}{N_c^2}(\hat{\mathcal{G}}^i)^T \mathcal{R}^i (\mathcal{R}^i)^T \hat{\mathcal{G}}^i_<,

    where :math:`\hat{\mathcal{G}}^i` represents the embedding matrix:math:`\mathcal{G}^i`
    after additional self-attention mechanism and :math:`\mathcal{R}^i` is defined by the full case in the se_e2_a descriptor.
    Note that we obtain :math:`\mathcal{G}^i` using the type embedding method by default in this descriptor.

    To perform the self-attention mechanism, the queries :math:`\mathcal{Q}^{i,l} \in \mathbb{R}^{N_c\times d_k}`,
    keys :math:`\mathcal{K}^{i,l} \in \mathbb{R}^{N_c\times d_k}`,
    and values :math:`\mathcal{V}^{i,l} \in \mathbb{R}^{N_c\times d_v}` are first obtained:

    .. math::
        \left(\mathcal{Q}^{i,l}\right)_{j}=Q_{l}\left(\left(\mathcal{G}^{i,l-1}\right)_{j}\right),

    .. math::
        \left(\mathcal{K}^{i,l}\right)_{j}=K_{l}\left(\left(\mathcal{G}^{i,l-1}\right)_{j}\right),

    .. math::
        \left(\mathcal{V}^{i,l}\right)_{j}=V_{l}\left(\left(\mathcal{G}^{i,l-1}\right)_{j}\right),

    where :math:`Q_{l}`, :math:`K_{l}`, :math:`V_{l}` represent three trainable linear transformations
    that output the queries and keys of dimension :math:`d_k` and values of dimension :math:`d_v`, and :math:`l`
    is the index of the attention layer.
    The input embedding matrix to the attention layers,  denoted by :math:`\mathcal{G}^{i,0}`,
    is chosen as the two-body embedding matrix.

    Then the scaled dot-product attention method is adopted:

    .. math::
        A(\mathcal{Q}^{i,l}, \mathcal{K}^{i,l}, \mathcal{V}^{i,l}, \mathcal{R}^{i,l})=\varphi\left(\mathcal{Q}^{i,l}, \mathcal{K}^{i,l},\mathcal{R}^{i,l}\right)\mathcal{V}^{i,l},

    where :math:`\varphi\left(\mathcal{Q}^{i,l}, \mathcal{K}^{i,l},\mathcal{R}^{i,l}\right) \in \mathbb{R}^{N_c\times N_c}` is attention weights.
    In the original attention method,
    one typically has :math:`\varphi\left(\mathcal{Q}^{i,l}, \mathcal{K}^{i,l}\right)=\mathrm{softmax}\left(\frac{\mathcal{Q}^{i,l} (\mathcal{K}^{i,l})^{T}}{\sqrt{d_{k}}}\right)`,
    with :math:`\sqrt{d_{k}}` being the normalization temperature.
    This is slightly modified to incorporate the angular information:

    .. math::
        \varphi\left(\mathcal{Q}^{i,l}, \mathcal{K}^{i,l},\mathcal{R}^{i,l}\right) = \mathrm{softmax}\left(\frac{\mathcal{Q}^{i,l} (\mathcal{K}^{i,l})^{T}}{\sqrt{d_{k}}}\right) \odot \hat{\mathcal{R}}^{i}(\hat{\mathcal{R}}^{i})^{T},

    where :math:`\hat{\mathcal{R}}^{i} \in \mathbb{R}^{N_c\times 3}` denotes normalized relative coordinates,
     :math:`\hat{\mathcal{R}}^{i}_{j} = \frac{\boldsymbol{r}_{ij}}{\lVert \boldsymbol{r}_{ij} \lVert}`
     and :math:`\odot` means element-wise multiplication.

    Then layer normalization is added in a residual way to finally obtain the self-attention local embedding matrix
     :math:`\hat{\mathcal{G}}^{i} = \mathcal{G}^{i,L_a}` after :math:`L_a` attention layers:[^1]

    .. math::
        \mathcal{G}^{i,l} = \mathcal{G}^{i,l-1} + \mathrm{LayerNorm}(A(\mathcal{Q}^{i,l}, \mathcal{K}^{i,l}, \mathcal{V}^{i,l}, \mathcal{R}^{i,l})).

    Parameters
    ----------
    rcut: float
            The cut-off radius :math:`r_c`
    rcut_smth: float
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
    ntypes : int
            Number of element types
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron: int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    tebd_dim: int
            Dimension of the type embedding
    tebd_input_mode: str
            The input mode of the type embedding. Supported modes are ["concat", "strip"].
            - "concat": Concatenate the type embedding with the smoothed radial information as the union input for the embedding network.
            - "strip": Use a separated embedding network for the type embedding and combine the output with the radial embedding network output.
    resnet_dt: bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable: bool
            If the weights of this descriptors are trainable.
    trainable_ln: bool
            Whether to use trainable shift and scale weights in layer normalization.
    ln_eps: float, Optional
            The epsilon value for layer normalization.
    type_one_side: bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
    attn: int
            Hidden dimension of the attention vectors
    attn_layer: int
            Number of attention layers
    attn_dotr: bool
            If dot the angular gate to the attention weights
    attn_mask: bool
            (Only support False to keep consistent with other backend references.)
            (Not used in this version. True option is not implemented.)
            If mask the diagonal of attention weights
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection: float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
    set_davg_zero: bool
            Set the shift of embedding net input to zero.
    activation_function: str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision: str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    scaling_factor: float
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
    normalize: bool
            Whether to normalize the hidden vectors in attention weights calculation.
    temperature: float
            If not None, the scaling of attention weights is `temperature` itself.
    smooth_type_embedding: bool
            Whether to use smooth process in attention weights calculation.
    concat_output_tebd: bool
            Whether to concat type embedding at the output of the descriptor.
    stripped_type_embedding: bool, Optional
            (Deprecated, kept only for compatibility.)
            Whether to strip the type embedding into a separate embedding network.
            Setting this parameter to `True` is equivalent to setting `tebd_input_mode` to 'strip'.
            Setting it to `False` is equivalent to setting `tebd_input_mode` to 'concat'.
            The default value is `None`, which means the `tebd_input_mode` setting will be used instead.
    use_econf_tebd: bool, Optional
            Whether to use electronic configuration type embedding.
    type_map: List[str], Optional
            A list of strings. Give the name to each type of atoms.
    spin
            (Only support None to keep consistent with other backend references.)
            (Not used in this version. Not-none option is not implemented.)
            The old implementation of deepspin.

    Limitations
    -----------
    The currently implementation will not support the following deprecated features
    1. spin is not None
    2. attn_mask == True

    References
    ----------
    .. [1] Duo Zhang, Hangrui Bi, Fu-Zhi Dai, Wanrun Jiang, Linfeng Zhang, and Han Wang. 2022.
       DPA-1: Pretraining of Attention-based Deep Potential Model for Molecular Simulation.
       arXiv preprint arXiv:2208.08236.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[List[int], int],
        ntypes: int,
        neuron: List[int] = [25, 50, 100],
        axis_neuron: int = 8,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = False,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        scaling_factor=1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        smooth_type_embedding: bool = True,
        concat_output_tebd: bool = True,
        spin: Optional[Any] = None,
        stripped_type_embedding: Optional[bool] = None,
        use_econf_tebd: bool = False,
        type_map: Optional[List[str]] = None,
        # consistent with argcheck, not used though
        seed: Optional[Union[int, List[int]]] = None,
    ) -> None:
        ## seed, uniform_seed, not included.
        # Ensure compatibility with the deprecated stripped_type_embedding option.
        if stripped_type_embedding is not None:
            # Use the user-set stripped_type_embedding parameter first
            tebd_input_mode = "strip" if stripped_type_embedding else "concat"
        if spin is not None:
            raise NotImplementedError("old implementation of spin is not supported.")
        if attn_mask:
            raise NotImplementedError(
                "old implementation of attn_mask is not supported."
            )
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-5

        self.se_atten = DescrptBlockSeAtten(
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            axis_neuron=axis_neuron,
            tebd_dim=tebd_dim,
            tebd_input_mode=tebd_input_mode,
            set_davg_zero=set_davg_zero,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=False,
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            smooth=smooth_type_embedding,
            type_one_side=type_one_side,
            exclude_types=exclude_types,
            env_protection=env_protection,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
            seed=child_seed(seed, 0),
        )
        self.use_econf_tebd = use_econf_tebd
        self.type_map = type_map
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=use_econf_tebd,
            type_map=type_map,
            seed=child_seed(seed, 1),
        )
        self.tebd_dim = tebd_dim
        self.concat_output_tebd = concat_output_tebd
        self.trainable = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.se_atten.get_rcut()

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.se_atten.get_rcut_smth()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.se_atten.get_nsel()

    def get_sel(self) -> List[int]:
        """Returns the number of selected atoms for each type."""
        return self.se_atten.get_sel()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.se_atten.get_ntypes()

    def get_type_map(self) -> List[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        ret = self.se_atten.get_dim_out()
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        return self.se_atten.dim_emb

    def mixed_types(self) -> bool:
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return self.se_atten.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return self.se_atten.has_message_passing()

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.se_atten.get_env_protection()

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

    @property
    def dim_out(self):
        return self.get_dim_out()

    @property
    def dim_emb(self):
        return self.get_dim_emb()

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        raise NotImplementedError

    def set_stat_mean_and_stddev(
        self,
        mean: np.ndarray,
        stddev: np.ndarray,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.se_atten.mean = mean
        self.se_atten.stddev = stddev

    def get_stat_mean_and_stddev(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and stddev for descriptor."""
        return self.se_atten.mean, self.se_atten.stddev

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert (
            self.type_map is not None
        ), "'type_map' must be defined when performing type changing!"
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        obj = self.se_atten
        obj.ntypes = len(type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        obj.reinit_exclude(map_pair_exclude_types(obj.exclude_types, remap_index))
        if has_new_type:
            # the avg and std of new types need to be updated
            extend_descrpt_stat(
                obj,
                type_map,
                des_with_stat=model_with_new_type_stat.se_atten
                if model_with_new_type_stat is not None
                else None,
            )
        obj["davg"] = obj["davg"][remap_index]
        obj["dstd"] = obj["dstd"][remap_index]

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
            The index mapping from extended to lcoal region. not used by this descriptor.

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
        del mapping
        nf, nloc, nnei = nlist.shape
        nall = coord_ext.reshape(nf, -1).shape[1] // 3
        # nf x nall x tebd_dim
        atype_embd_ext = self.type_embedding.call()[atype_ext]
        # nfnl x tebd_dim
        atype_embd = atype_embd_ext[:, :nloc, :]
        grrg, g2, h2, rot_mat, sw = self.se_atten(
            nlist,
            coord_ext,
            atype_ext,
            atype_embd_ext,
            mapping=None,
        )
        # nf x nloc x (ng x ng1 + tebd_dim)
        if self.concat_output_tebd:
            grrg = np.concatenate([grrg, atype_embd.reshape(nf, nloc, -1)], axis=-1)
        return grrg, rot_mat, None, None, sw

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        obj = self.se_atten
        data = {
            "@class": "Descriptor",
            "type": "dpa1",
            "@version": 1,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "ntypes": obj.ntypes,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "tebd_dim": obj.tebd_dim,
            "tebd_input_mode": obj.tebd_input_mode,
            "set_davg_zero": obj.set_davg_zero,
            "attn": obj.attn,
            "attn_layer": obj.attn_layer,
            "attn_dotr": obj.attn_dotr,
            "attn_mask": False,
            "activation_function": obj.activation_function,
            "resnet_dt": obj.resnet_dt,
            "scaling_factor": obj.scaling_factor,
            "normalize": obj.normalize,
            "temperature": obj.temperature,
            "trainable_ln": obj.trainable_ln,
            "ln_eps": obj.ln_eps,
            "smooth_type_embedding": obj.smooth,
            "type_one_side": obj.type_one_side,
            "concat_output_tebd": self.concat_output_tebd,
            "use_econf_tebd": self.use_econf_tebd,
            "type_map": self.type_map,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[obj.precision]).name,
            "embeddings": obj.embeddings.serialize(),
            "attention_layers": obj.dpa1_attention.serialize(),
            "env_mat": obj.env_mat.serialize(),
            "type_embedding": self.type_embedding.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": obj["davg"],
                "dstd": obj["dstd"],
            },
            ## to be updated when the options are supported.
            "trainable": self.trainable,
            "spin": None,
        }
        if obj.tebd_input_mode in ["strip"]:
            data.update({"embeddings_strip": obj.embeddings_strip.serialize()})
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA1":
        """Deserialize from dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        type_embedding = data.pop("type_embedding")
        attention_layers = data.pop("attention_layers")
        env_mat = data.pop("env_mat")
        tebd_input_mode = data["tebd_input_mode"]
        if tebd_input_mode in ["strip"]:
            embeddings_strip = data.pop("embeddings_strip")
        else:
            embeddings_strip = None
        obj = cls(**data)

        obj.se_atten["davg"] = variables["davg"]
        obj.se_atten["dstd"] = variables["dstd"]
        obj.se_atten.embeddings = NetworkCollection.deserialize(embeddings)
        if tebd_input_mode in ["strip"]:
            obj.se_atten.embeddings_strip = NetworkCollection.deserialize(
                embeddings_strip
            )
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)
        obj.se_atten.dpa1_attention = NeighborGatedAttention.deserialize(
            attention_layers
        )
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[List[str]],
        local_jdata: dict,
    ) -> Tuple[dict, Optional[float]]:
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
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data, type_map, local_jdata_cpy["rcut"], local_jdata_cpy["sel"], True
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist


@DescriptorBlock.register("se_atten")
class DescrptBlockSeAtten(NativeOP, DescriptorBlock):
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[List[int], int],
        ntypes: int,
        neuron: List[int] = [25, 50, 100],
        axis_neuron: int = 8,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        resnet_dt: bool = False,
        type_one_side: bool = False,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        scaling_factor=1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        smooth: bool = True,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> None:
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        if isinstance(sel, int):
            sel = [sel]
        self.sel = sel
        self.nnei = sum(sel)
        self.ntypes = ntypes
        self.neuron = neuron
        self.filter_neuron = self.neuron
        self.axis_neuron = axis_neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.resnet_dt = resnet_dt
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.type_one_side = type_one_side
        self.attn = attn
        self.attn_layer = attn_layer
        self.attn_dotr = attn_dotr
        self.attn_mask = attn_mask
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.smooth = smooth
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)

        self.tebd_dim_input = self.tebd_dim if self.type_one_side else self.tebd_dim * 2
        if self.tebd_input_mode in ["concat"]:
            self.embd_input_dim = 1 + self.tebd_dim_input
        else:
            self.embd_input_dim = 1
        self.embeddings = NetworkCollection(
            ndim=0,
            ntypes=self.ntypes,
            network_type="embedding_network",
        )
        self.embeddings[0] = EmbeddingNet(
            self.embd_input_dim,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
            seed=child_seed(seed, 0),
        )
        if self.tebd_input_mode in ["strip"]:
            self.embeddings_strip = NetworkCollection(
                ndim=0,
                ntypes=self.ntypes,
                network_type="embedding_network",
            )
            self.embeddings_strip[0] = EmbeddingNet(
                self.tebd_dim_input,
                self.neuron,
                self.activation_function,
                self.resnet_dt,
                self.precision,
                seed=child_seed(seed, 1),
            )
        else:
            self.embeddings_strip = None
        self.dpa1_attention = NeighborGatedAttention(
            self.attn_layer,
            self.nnei,
            self.filter_neuron[-1],
            self.attn,
            dotr=self.attn_dotr,
            scaling_factor=self.scaling_factor,
            normalize=self.normalize,
            temperature=self.temperature,
            trainable_ln=self.trainable_ln,
            ln_eps=self.ln_eps,
            smooth=self.smooth,
            precision=self.precision,
            seed=child_seed(seed, 2),
        )

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

    def get_sel(self) -> List[int]:
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

    def __setitem__(self, key, value):
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
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
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
        return self.filter_neuron[-1] * self.axis_neuron

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
        merged: Union[Callable[[], List[dict]], List[dict]],
        path: Optional[DPPath] = None,
    ):
        """Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data."""
        raise NotImplementedError

    def get_stats(self):
        """Get the statistics of the descriptor."""
        raise NotImplementedError

    def reinit_exclude(
        self,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def cal_g(
        self,
        ss,
        embedding_idx,
    ):
        nfnl, nnei = ss.shape[0:2]
        ss = ss.reshape(nfnl, nnei, -1)
        # nfnl x nnei x ng
        gg = self.embeddings[embedding_idx].call(ss)
        return gg

    def cal_g_strip(
        self,
        ss,
        embedding_idx,
    ):
        assert self.embeddings_strip is not None
        nfnl, nnei = ss.shape[0:2]
        ss = ss.reshape(nfnl, nnei, -1)
        # nfnl x nnei x ng
        gg = self.embeddings_strip[embedding_idx].call(ss)
        return gg

    def call(
        self,
        nlist: np.ndarray,
        coord_ext: np.ndarray,
        atype_ext: np.ndarray,
        atype_embd_ext: Optional[np.ndarray] = None,
        mapping: Optional[np.ndarray] = None,
    ):
        # nf x nloc x nnei x 4
        dmatrix, diff, sw = self.env_mat.call(
            coord_ext, atype_ext, nlist, self.mean, self.stddev
        )
        nf, nloc, nnei, _ = dmatrix.shape
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        # nfnl x nnei
        exclude_mask = exclude_mask.reshape(nf * nloc, nnei)
        # nfnl x nnei
        nlist = nlist.reshape(nf * nloc, nnei)
        nlist = np.where(exclude_mask, nlist, -1)
        # nfnl x nnei x 4
        dmatrix = dmatrix.reshape(nf * nloc, nnei, 4)
        # nfnl x nnei x 1
        sw = sw.reshape(nf * nloc, nnei, 1)
        # nfnl x tebd_dim
        atype_embd = atype_embd_ext[:, :nloc, :].reshape(nf * nloc, -1)
        # nfnl x nnei x tebd_dim
        atype_embd_nnei = np.tile(atype_embd[:, np.newaxis, :], (1, nnei, 1))
        # nfnl x nnei
        nlist_mask = nlist != -1
        # nfnl x nnei x 1
        sw = np.where(nlist_mask[:, :, None], sw, 0.0)
        nlist_masked = np.where(nlist_mask, nlist, 0)
        index = np.tile(nlist_masked.reshape(nf, -1, 1), (1, 1, self.tebd_dim))
        # nfnl x nnei x tebd_dim
        atype_embd_nlist = np.take_along_axis(atype_embd_ext, index, axis=1).reshape(
            nf * nloc, nnei, self.tebd_dim
        )
        ng = self.neuron[-1]
        # nfnl x nnei x 4
        rr = dmatrix.reshape(nf * nloc, nnei, 4)
        rr = rr * exclude_mask[:, :, None]
        # nfnl x nnei x 1
        ss = rr[..., 0:1]
        if self.tebd_input_mode in ["concat"]:
            if not self.type_one_side:
                # nfnl x nnei x (1 + 2 * tebd_dim)
                ss = np.concatenate([ss, atype_embd_nlist, atype_embd_nnei], axis=-1)
            else:
                # nfnl x nnei x (1 + tebd_dim)
                ss = np.concatenate([ss, atype_embd_nlist], axis=-1)
                # calculate gg
                # nfnl x nnei x ng
            gg = self.cal_g(ss, 0)
        elif self.tebd_input_mode in ["strip"]:
            # nfnl x nnei x ng
            gg_s = self.cal_g(ss, 0)
            assert self.embeddings_strip is not None
            if not self.type_one_side:
                # nfnl x nnei x (tebd_dim * 2)
                tt = np.concatenate([atype_embd_nlist, atype_embd_nnei], axis=-1)
            else:
                # nfnl x nnei x tebd_dim
                tt = atype_embd_nlist
            # nfnl x nnei x ng
            gg_t = self.cal_g_strip(tt, 0)
            if self.smooth:
                gg_t = gg_t * sw.reshape(-1, self.nnei, 1)
            # nfnl x nnei x ng
            gg = gg_s * gg_t + gg_s
        else:
            raise NotImplementedError

        input_r = rr.reshape(-1, nnei, 4)[:, :, 1:4] / np.maximum(
            np.linalg.norm(rr.reshape(-1, nnei, 4)[:, :, 1:4], axis=-1, keepdims=True),
            1e-12,
        )
        gg = self.dpa1_attention(
            gg, nlist_mask, input_r=input_r, sw=sw
        )  # shape is [nframes*nloc, self.neei, out_size]
        # nfnl x ng x 4
        gr = np.einsum("lni,lnj->lij", gg, rr)
        gr /= self.nnei
        gr1 = gr[:, : self.axis_neuron, :]
        # nfnl x ng x ng1
        grrg = np.einsum("lid,ljd->lij", gr, gr1)
        # nf x nloc x (ng x ng1)
        grrg = grrg.reshape(nf, nloc, ng * self.axis_neuron).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        return (
            grrg.reshape(-1, nloc, self.filter_neuron[-1] * self.axis_neuron),
            gg.reshape(-1, nloc, self.nnei, self.filter_neuron[-1]),
            dmatrix.reshape(-1, nloc, self.nnei, 4)[..., 1:],
            gr[..., 1:].reshape(-1, nloc, self.filter_neuron[-1], 3),
            sw,
        )

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return False


class NeighborGatedAttention(NativeOP):
    def __init__(
        self,
        layer_num: int,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        """Construct a neighbor-wise attention net."""
        super().__init__()
        self.layer_num = layer_num
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.smooth = smooth
        self.precision = precision
        self.network_type = NeighborGatedAttentionLayer

        self.attention_layers = [
            NeighborGatedAttentionLayer(
                nnei,
                embed_dim,
                hidden_dim,
                dotr=dotr,
                do_mask=do_mask,
                scaling_factor=scaling_factor,
                normalize=normalize,
                temperature=temperature,
                trainable_ln=trainable_ln,
                ln_eps=ln_eps,
                smooth=smooth,
                precision=precision,
                seed=child_seed(seed, ii),
            )
            for ii in range(layer_num)
        ]

    def call(
        self,
        input_G,
        nei_mask,
        input_r: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None,
    ):
        out = input_G
        for layer in self.attention_layers:
            out = layer(out, nei_mask, input_r=input_r, sw=sw)
        return out

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.attention_layers[key]
        else:
            raise TypeError(key)

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError(key)
        if isinstance(value, self.network_type):
            pass
        elif isinstance(value, dict):
            value = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        self.attention_layers[key] = value

    def serialize(self):
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "NeighborGatedAttention",
            "@version": 1,
            "layer_num": self.layer_num,
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "precision": self.precision,
            "attention_layers": [layer.serialize() for layer in self.attention_layers],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NeighborGatedAttention":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        attention_layers = data.pop("attention_layers")
        obj = cls(**data)
        obj.attention_layers = [
            NeighborGatedAttentionLayer.deserialize(layer) for layer in attention_layers
        ]
        return obj


class NeighborGatedAttentionLayer(NativeOP):
    def __init__(
        self,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        """Construct a neighbor-wise attention layer."""
        super().__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.precision = precision
        self.attention_layer = GatedAttentionLayer(
            nnei,
            embed_dim,
            hidden_dim,
            dotr=dotr,
            do_mask=do_mask,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            smooth=smooth,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.attn_layer_norm = LayerNorm(
            self.embed_dim,
            eps=ln_eps,
            trainable=self.trainable_ln,
            precision=precision,
            seed=child_seed(seed, 1),
        )

    def call(
        self,
        x,
        nei_mask,
        input_r: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None,
    ):
        residual = x
        x, _ = self.attention_layer(x, nei_mask, input_r=input_r, sw=sw)
        x = residual + x
        x = self.attn_layer_norm(x)
        return x

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "precision": self.precision,
            "attention_layer": self.attention_layer.serialize(),
            "attn_layer_norm": self.attn_layer_norm.serialize(),
        }

    @classmethod
    def deserialize(cls, data) -> "NeighborGatedAttentionLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        attention_layer = data.pop("attention_layer")
        attn_layer_norm = data.pop("attn_layer_norm")
        obj = cls(**data)
        obj.attention_layer = GatedAttentionLayer.deserialize(attention_layer)
        obj.attn_layer_norm = LayerNorm.deserialize(attn_layer_norm)
        return obj


class GatedAttentionLayer(NativeOP):
    def __init__(
        self,
        nnei: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        bias: bool = True,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        """Construct a multi-head neighbor-wise attention net."""
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dotr = dotr
        self.do_mask = do_mask
        self.bias = bias
        self.smooth = smooth
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.precision = precision
        self.scaling = (
            (self.head_dim * scaling_factor) ** -0.5
            if temperature is None
            else temperature
        )
        self.normalize = normalize
        self.in_proj = NativeLayer(
            embed_dim,
            hidden_dim * 3,
            bias=bias,
            use_timestep=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.out_proj = NativeLayer(
            hidden_dim,
            embed_dim,
            bias=bias,
            use_timestep=False,
            precision=precision,
            seed=child_seed(seed, 1),
        )

    def call(self, query, nei_mask, input_r=None, sw=None, attnw_shift=20.0):
        # Linear projection
        q, k, v = np.split(self.in_proj(query), 3, axis=-1)
        # Reshape and normalize
        # (nf x nloc) x num_heads x nnei x head_dim
        q = q.reshape(-1, self.nnei, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(-1, self.nnei, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(-1, self.nnei, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        if self.normalize:
            q = np_normalize(q, axis=-1)
            k = np_normalize(k, axis=-1)
            v = np_normalize(v, axis=-1)
        q = q * self.scaling
        # Attention weights
        # (nf x nloc) x num_heads x nnei x nnei
        attn_weights = q @ k.transpose(0, 1, 3, 2)
        nei_mask = nei_mask.reshape(-1, self.nnei)
        if self.smooth:
            sw = sw.reshape(-1, 1, self.nnei)
            attn_weights = (attn_weights + attnw_shift) * sw[:, :, :, None] * sw[
                :, :, None, :
            ] - attnw_shift
        else:
            attn_weights = np.where(nei_mask[:, None, None, :], attn_weights, -np.inf)
        attn_weights = np_softmax(attn_weights, axis=-1)
        attn_weights = np.where(nei_mask[:, None, :, None], attn_weights, 0.0)
        if self.smooth:
            attn_weights = attn_weights * sw[:, :, :, None] * sw[:, :, None, :]
        if self.dotr:
            angular_weight = (input_r @ input_r.transpose(0, 2, 1)).reshape(
                -1, 1, self.nnei, self.nnei
            )
            attn_weights = attn_weights * angular_weight
        # Output projection
        # (nf x nloc) x num_heads x nnei x head_dim
        o = attn_weights @ v
        # (nf x nloc) x nnei x (num_heads x head_dim)
        o = o.transpose(0, 2, 1, 3).reshape(-1, self.nnei, self.hidden_dim)
        output = self.out_proj(o)
        return output, attn_weights

    def serialize(self):
        return {
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dotr": self.dotr,
            "do_mask": self.do_mask,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "bias": self.bias,
            "smooth": self.smooth,
            "precision": self.precision,
            "in_proj": self.in_proj.serialize(),
            "out_proj": self.out_proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data):
        data = data.copy()
        in_proj = data.pop("in_proj")
        out_proj = data.pop("out_proj")
        obj = cls(**data)
        obj.in_proj = NativeLayer.deserialize(in_proj)
        obj.out_proj = NativeLayer.deserialize(out_proj)
        return obj
