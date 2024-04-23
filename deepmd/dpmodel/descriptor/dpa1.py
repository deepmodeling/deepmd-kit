# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.dpmodel.utils.network import (
    LayerNorm,
    NativeLayer,
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
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"

from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
)

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

from .base_descriptor import (
    BaseDescriptor,
)


def np_softmax(x, axis=-1):
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
            The way to mix the type embeddings. Supported options are `concat`.
            (TODO need to support stripped_type_embedding option)
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
    spin
            (Only support None to keep consistent with other backend references.)
            (Not used in this version. Not-none option is not implemented.)
            The old implementation of deepspin.

    Limitations
    -----------
    The currently implementation does not support the following features
    1. tebd_input_mode != 'concat'

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
        exclude_types: List[List[int]] = [],
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
        # consistent with argcheck, not used though
        seed: Optional[int] = None,
    ) -> None:
        ## seed, uniform_seed, multi_task, not included.
        if spin is not None:
            raise NotImplementedError("old implementation of spin is not supported.")
        if attn_mask:
            raise NotImplementedError(
                "old implementation of attn_mask is not supported."
            )
        # TODO
        if tebd_input_mode != "concat":
            raise NotImplementedError("tebd_input_mode != 'concat' not implemented")
        #  to keep consistent with default value in this backends
        if ln_eps is None:
            ln_eps = 1e-5

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
        self.trainable = trainable
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.type_one_side = type_one_side
        self.attn = attn
        self.attn_layer = attn_layer
        self.attn_dotr = attn_dotr
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.smooth = smooth_type_embedding
        self.concat_output_tebd = concat_output_tebd
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)

        self.type_embedding = TypeEmbedNet(
            ntypes=self.ntypes,
            neuron=[self.tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
        )
        if self.tebd_input_mode in ["concat"]:
            if not self.type_one_side:
                in_dim = 1 + self.tebd_dim * 2
            else:
                in_dim = 1 + self.tebd_dim
        else:
            in_dim = 1
        self.embeddings = NetworkCollection(
            ndim=0,
            ntypes=self.ntypes,
            network_type="embedding_network",
        )
        self.embeddings[0] = EmbeddingNet(
            in_dim,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
        )
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
        )

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.env_mat = EnvMat(self.rcut, self.rcut_smth, protection=self.env_protection)
        self.davg = np.zeros(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.dstd = np.ones(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.orig_sel = self.sel

    def __setitem__(self, key, value):
        if key in ("avg", "data_avg", "davg"):
            self.davg = value
        elif key in ("std", "data_std", "dstd"):
            self.dstd = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.davg
        elif key in ("std", "data_std", "dstd"):
            return self.dstd
        else:
            raise KeyError(key)

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.get_dim_out()

    def get_dim_out(self):
        """Returns the output dimension of this descriptor."""
        return (
            self.neuron[-1] * self.axis_neuron + self.tebd_dim
            if self.concat_output_tebd
            else self.neuron[-1] * self.axis_neuron
        )

    def get_dim_emb(self):
        """Returns the embedding (g2) dimension of this descriptor."""
        return self.neuron[-1]

    def get_rcut(self):
        """Returns cutoff radius."""
        return self.rcut

    def get_sel(self):
        """Returns cutoff radius."""
        return self.sel

    def mixed_types(self):
        """If true, the discriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the discriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def share_params(self, base_class, shared_level, resume=False):
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def compute_input_stats(self, merged: List[dict], path: Optional[DPPath] = None):
        """Update mean and stddev for descriptor elements."""
        raise NotImplementedError

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

    def reinit_exclude(
        self,
        exclude_types: List[Tuple[int, int]] = [],
    ):
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

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
        # nf x nloc x nnei x 4
        dmatrix, sw = self.env_mat.call(
            coord_ext, atype_ext, nlist, self.davg, self.dstd
        )
        nf, nloc, nnei, _ = dmatrix.shape
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        # nfnl x nnei
        nlist = nlist.reshape(nf * nloc, nnei)
        # nfnl x nnei x 4
        dmatrix = dmatrix.reshape(nf * nloc, nnei, 4)
        # nfnl x nnei x 1
        sw = sw.reshape(nf * nloc, nnei, 1)

        # add type embedding into input
        # nf x nall x tebd_dim
        atype_embd_ext = self.type_embedding.call()[atype_ext]
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
        # nfnl x nnei
        exclude_mask = exclude_mask.reshape(nf * nloc, nnei)
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
        else:
            raise NotImplementedError

        # calculate gg
        gg = self.cal_g(ss, 0)
        input_r = dmatrix.reshape(-1, nnei, 4)[:, :, 1:4] / (
            np.linalg.norm(
                dmatrix.reshape(-1, nnei, 4)[:, :, 1:4], axis=-1, keepdims=True
            )
            + 1e-12
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
        # nf x nloc x (ng x ng1 + tebd_dim)
        if self.concat_output_tebd:
            grrg = np.concatenate([grrg, atype_embd.reshape(nf, nloc, -1)], axis=-1)
        return grrg, gr[..., 1:], None, None, sw

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        return {
            "@class": "Descriptor",
            "type": "dpa1",
            "@version": 1,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "neuron": self.neuron,
            "axis_neuron": self.axis_neuron,
            "tebd_dim": self.tebd_dim,
            "tebd_input_mode": self.tebd_input_mode,
            "set_davg_zero": self.set_davg_zero,
            "attn": self.attn,
            "attn_layer": self.attn_layer,
            "attn_dotr": self.attn_dotr,
            "attn_mask": False,
            "activation_function": self.activation_function,
            "resnet_dt": self.resnet_dt,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "smooth_type_embedding": self.smooth,
            "type_one_side": self.type_one_side,
            "concat_output_tebd": self.concat_output_tebd,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "embeddings": self.embeddings.serialize(),
            "attention_layers": self.dpa1_attention.serialize(),
            "env_mat": self.env_mat.serialize(),
            "type_embedding": self.type_embedding.serialize(),
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "@variables": {
                "davg": self.davg,
                "dstd": self.dstd,
            },
            ## to be updated when the options are supported.
            "trainable": True,
            "spin": None,
        }

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
        obj = cls(**data)

        obj["davg"] = variables["davg"]
        obj["dstd"] = variables["dstd"]
        obj.embeddings = NetworkCollection.deserialize(embeddings)
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)
        obj.dpa1_attention = NeighborGatedAttention.deserialize(attention_layers)
        return obj

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
        return UpdateSel().update_one_sel(global_jdata, local_jdata_cpy, True)


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
            )
            for _ in range(layer_num)
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
        )
        self.attn_layer_norm = LayerNorm(
            self.embed_dim, eps=ln_eps, trainable=self.trainable_ln, precision=precision
        )

    def call(
        self,
        x,
        nei_mask,
        input_r: Optional[np.ndarray] = None,
        sw: Optional[np.ndarray] = None,
    ):
        residual = x
        x = self.attention_layer(x, nei_mask, input_r=input_r, sw=sw)
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
        dotr: bool = False,
        do_mask: bool = False,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        bias: bool = True,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
    ):
        """Construct a neighbor-wise attention net."""
        super().__init__()
        self.nnei = nnei
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dotr = dotr
        self.do_mask = do_mask
        self.bias = bias
        self.smooth = smooth
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.precision = precision
        if temperature is None:
            self.scaling = (self.hidden_dim * scaling_factor) ** -0.5
        else:
            self.scaling = temperature
        self.normalize = normalize
        self.in_proj = NativeLayer(
            embed_dim,
            hidden_dim * 3,
            bias=bias,
            use_timestep=False,
            precision=precision,
        )
        self.out_proj = NativeLayer(
            hidden_dim,
            embed_dim,
            bias=bias,
            use_timestep=False,
            precision=precision,
        )

    def call(self, query, nei_mask, input_r=None, sw=None, attnw_shift=20.0):
        # Linear projection
        q, k, v = np.split(self.in_proj(query), 3, axis=-1)
        # Reshape and normalize
        q = q.reshape(-1, self.nnei, self.hidden_dim)
        k = k.reshape(-1, self.nnei, self.hidden_dim)
        v = v.reshape(-1, self.nnei, self.hidden_dim)
        if self.normalize:
            q = np_normalize(q, axis=-1)
            k = np_normalize(k, axis=-1)
            v = np_normalize(v, axis=-1)
        q = q * self.scaling
        # Attention weights
        attn_weights = q @ k.transpose(0, 2, 1)
        nei_mask = nei_mask.reshape(-1, self.nnei)
        if self.smooth:
            sw = sw.reshape(-1, self.nnei)
            attn_weights = (attn_weights + attnw_shift) * sw[:, None, :] * sw[
                :, :, None
            ] - attnw_shift
        else:
            attn_weights = np.where(nei_mask[:, None, :], attn_weights, -np.inf)
        attn_weights = np_softmax(attn_weights, axis=-1)
        attn_weights = np.where(nei_mask[:, :, None], attn_weights, 0.0)
        if self.smooth:
            attn_weights = attn_weights * sw[:, None, :] * sw[:, :, None]
        if self.dotr:
            angular_weight = input_r @ input_r.transpose(0, 2, 1)
            attn_weights = attn_weights * angular_weight
        # Output projection
        o = attn_weights @ v
        output = self.out_proj(o)
        return output

    def serialize(self):
        return {
            "nnei": self.nnei,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
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
