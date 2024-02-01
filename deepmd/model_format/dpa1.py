# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"

import copy
from typing import (
    Any,
    List,
    Optional,
)

from .common import (
    DEFAULT_PRECISION,
    NativeOP,
)
from .env_mat import (
    EnvMat,
)
from .network import (
    EmbdLayer,
    EmbeddingNet,
    NetworkCollection,
)


class DescrptDPA1(NativeOP):
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
    rcut
            The cut-off radius :math:`r_c`
    rcut_smth
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[str]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    ntypes : int
            Number of element types
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    tebd_dim: int
            Dimension of the type embedding
    tebd_input_mode: str
            The way to mix the type embeddings. Supported options are `concat`, `dot_residual_s`.
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable
            If the weights of embedding net are trainable.
    type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
    attn: int
            Hidden dimension of the attention vectors
    attn_layer: int
            Number of attention layers
    attn_dotr: bool
            If dot the angular gate to the attention weights
    attn_mask: bool
            If mask the diagonal of attention weights
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    set_davg_zero
            Set the shift of embedding net input to zero.
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    scaling_factor: float
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
    temperature: Optional[float]
            If not None, the scaling of attention weights is `temperature` itself.
    spin
            The deepspin object.

    Limitations
    -----------
    The currently implementation does not support the following features

    1. type_one_side == False
    2. exclude_types != []
    3. spin is not None
    4. tebd_input_mode != 'concat'
    5. smooth == True

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
        sel: List[str],
        ntypes: int,
        neuron: List[int] = [25, 50, 100],
        axis_neuron: int = 8,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = True,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        scaling_factor=1.0,
        normalize=True,
        temperature=None,
        smooth: bool = True,
        concat_output_tebd: bool = True,
        spin: Optional[Any] = None,
    ) -> None:
        ## seed, uniform_seed, multi_task, not included.
        if not type_one_side:
            raise NotImplementedError("type_one_side == False not implemented")
        if exclude_types != []:
            raise NotImplementedError("exclude_types is not implemented")
        if spin is not None:
            raise NotImplementedError("spin is not implemented")
        # TODO
        if tebd_input_mode != "concat":
            raise NotImplementedError("tebd_input_mode != 'concat' not implemented")
        if not smooth:
            raise NotImplementedError("smooth == False not implemented")

        self.rcut = rcut
        self.rcut_smth = rcut_smth
        if isinstance(sel, int):
            sel = [sel]
        self.sel = sel
        self.ntypes = ntypes
        self.neuron = neuron
        self.axis_neuron = axis_neuron
        self.tebd_dim = tebd_dim
        self.tebd_input_mode = tebd_input_mode
        self.resnet_dt = resnet_dt
        self.trainable = trainable
        self.type_one_side = type_one_side
        self.attn = attn
        self.attn_layer = attn_layer
        self.attn_dotr = attn_dotr
        self.attn_mask = attn_mask
        self.exclude_types = exclude_types
        self.set_davg_zero = set_davg_zero
        self.activation_function = activation_function
        self.precision = precision
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.temperature = temperature
        self.concat_output_tebd = concat_output_tebd
        self.spin = spin

        self.type_embedding = EmbdLayer(
            ntypes, tebd_dim, padding=True, precision=precision
        )
        in_dim = 1 + self.tebd_dim * 2 if self.tebd_input_mode in ["concat"] else 1
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
        # self.dpa1_attention = NeighborGatedAttention
        self.env_mat = EnvMat(self.rcut, self.rcut_smth)
        self.nnei = np.sum(self.sel)
        self.davg = np.zeros([self.ntypes, self.nnei, 4])
        self.dstd = np.ones([self.ntypes, self.nnei, 4])
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
        return (
            self.neuron[-1] * self.axis_neuron + self.tebd_dim * 2
            if self.concat_output_tebd
            else self.neuron[-1] * self.axis_neuron
        )

    def cal_g(
        self,
        ss,
        ll,
    ):
        nf, nloc, nnei = ss.shape[0:3]
        ss = ss.reshape(nf, nloc, nnei, -1)
        # nf x nloc x nnei x ng
        gg = self.embeddings[ll].call(ss)
        return gg

    def call(
        self,
        coord_ext,
        atype_ext,
        nlist,
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
        # nf x nloc x nnei x 4
        rr, ww = self.env_mat.call(coord_ext, atype_ext, nlist, self.davg, self.dstd)
        nf, nloc, nnei, _ = rr.shape

        # add type embedding into input
        # nf x nall x tebd_dim
        atype_embd_ext = self.type_embedding.call(atype_ext)
        atype_embd = atype_embd_ext[:, :nloc, :]
        # nf x nloc x nnei x tebd_dim
        atype_embd_nnei = np.tile(atype_embd[:, :, np.newaxis, :], (1, 1, nnei, 1))
        nlist_mask = nlist != -1
        nlist_masked = np.copy(nlist)
        nlist_masked[nlist_masked == -1] = 0
        index = np.tile(nlist_masked.reshape(nf, -1, 1), (1, 1, self.tebd_dim))
        # nf x nloc x nnei x tebd_dim
        atype_embd_nlist = np.take_along_axis(atype_embd_ext, index, axis=1).reshape(
            nf, nloc, nnei, self.tebd_dim
        )
        ng = self.neuron[-1]
        ss = rr[..., 0:1]
        ss = np.concatenate([ss, atype_embd_nlist, atype_embd_nnei], axis=-1)

        # calculate gg
        gg = self.cal_g(ss, 0)
        # nf x nloc x ng x 4
        gr = np.einsum("flni,flnj->flij", gg, rr)
        # nf x nloc x ng x 4
        gr /= self.nnei
        gr1 = gr[:, :, : self.axis_neuron, :]
        # nf x nloc x ng x ng1
        grrg = np.einsum("flid,fljd->flij", gr, gr1)
        # nf x nloc x (ng x ng1)
        grrg = grrg.reshape(nf, nloc, ng * self.axis_neuron)
        if self.concat_output_tebd:
            grrg = np.concatenate([grrg, atype_embd], axis=-1)
        return grrg, gr[..., 1:], None, None, ww

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        return {
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "neuron": self.neuron,
            "axis_neuron": self.axis_neuron,
            "tebd_dim": self.tebd_dim,
            "tebd_input_mode": self.tebd_input_mode,
            "resnet_dt": self.resnet_dt,
            "trainable": self.trainable,
            "type_one_side": self.type_one_side,
            "exclude_types": self.exclude_types,
            "set_davg_zero": self.set_davg_zero,
            "attn": self.attn,
            "attn_layer": self.attn_layer,
            "attn_dotr": self.attn_dotr,
            "attn_mask": self.attn_mask,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "spin": self.spin,
            "scaling_factor": self.scaling_factor,
            "normalize": self.normalize,
            "temperature": self.temperature,
            "concat_output_tebd": self.concat_output_tebd,
            "embeddings": self.embeddings.serialize(),
            # "attention_layers": self.dpa1_attention.serialize(),
            "env_mat": self.env_mat.serialize(),
            "type_embedding": self.type_embedding.serialize(),
            "@variables": {
                "davg": self.davg,
                "dstd": self.dstd,
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA1":
        """Deserialize from dict."""
        data = copy.deepcopy(data)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        type_embedding = data.pop("type_embedding")
        attention_layers = data.pop("attention_layers", None)
        env_mat = data.pop("env_mat")
        obj = cls(**data)
        obj["davg"] = variables["davg"]
        obj["dstd"] = variables["dstd"]
        obj.type_embedding = EmbdLayer.deserialize(type_embedding)
        obj.embeddings = NetworkCollection.deserialize(embeddings)
        obj.env_mat = EnvMat.deserialize(env_mat)
        # obj.dpa1_attention = NeighborGatedAttention.deserialize(attention_layers)
        return obj
