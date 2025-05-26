# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    TypeEmbedNet,
    TypeEmbedNetConsistent,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.tabulate import (
    DPTabulate,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
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
    extend_descrpt_stat,
)
from .se_atten import (
    DescrptBlockSeAtten,
    NeighborGatedAttention,
)


@BaseDescriptor.register("dpa1")
@BaseDescriptor.register("se_atten")
class DescrptDPA1(BaseDescriptor, torch.nn.Module):
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
    exclude_types : list[list[int]]
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
    seed: int, Optional
            Random seed for parameter initialization.
    use_econf_tebd: bool, Optional
            Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, Optional
            Whether to use bias in the type embedding layer.
    type_map: list[str], Optional
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
        sel: Union[list[int], int],
        ntypes: int,
        neuron: list = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        set_davg_zero: bool = True,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        scaling_factor: int = 1.0,
        normalize=True,
        temperature=None,
        concat_output_tebd: bool = True,
        trainable: bool = True,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        smooth_type_embedding: bool = True,
        type_one_side: bool = False,
        stripped_type_embedding: Optional[bool] = None,
        seed: Optional[Union[int, list[int]]] = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: Optional[list[str]] = None,
        # not implemented
        spin=None,
        type: Optional[str] = None,
    ) -> None:
        super().__init__()
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

        self.tebd_input_mode = tebd_input_mode

        del type, spin, attn_mask
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
            seed=child_seed(seed, 1),
        )
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.compress = False
        self.type_embedding = TypeEmbedNet(
            ntypes,
            tebd_dim,
            precision=precision,
            seed=child_seed(seed, 2),
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
        )
        self.prec = PRECISION_DICT[precision]
        self.tebd_dim = tebd_dim
        self.concat_output_tebd = concat_output_tebd
        self.trainable = trainable
        # set trainable
        for param in self.parameters():
            param.requires_grad = trainable

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.se_atten.get_rcut()

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.se_atten.get_rcut_smth()

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return self.se_atten.get_nsel()

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.se_atten.get_sel()

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.se_atten.get_ntypes()

    def get_type_map(self) -> list[str]:
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
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return self.se_atten.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return self.se_atten.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return self.se_atten.need_sorted_nlist_for_lower()

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.se_atten.get_env_protection()

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
        # share all parameters in both type_embedding and se_atten
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            self.se_atten.share_params(base_class.se_atten, 0, resume=resume)
        # shared_level: 1
        # share all parameters in type_embedding
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
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
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        return self.se_atten.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.se_atten.mean = mean
        self.se_atten.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mean and stddev for descriptor."""
        return self.se_atten.mean, self.se_atten.stddev

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

    def serialize(self) -> dict:
        obj = self.se_atten
        data = {
            "@class": "Descriptor",
            "type": "dpa1",
            "@version": 2,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "ntypes": obj.ntypes,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "tebd_dim": obj.tebd_dim,
            "tebd_input_mode": obj.tebd_input_mode,
            "set_davg_zero": obj.set_davg_zero,
            "attn": obj.attn_dim,
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
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            # make deterministic
            "precision": RESERVED_PRECISION_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "attention_layers": obj.dpa1_attention.serialize(),
            "env_mat": DPEnvMat(
                obj.rcut, obj.rcut_smth, obj.env_protection
            ).serialize(),
            "type_embedding": self.type_embedding.embedding.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            "trainable": self.trainable,
            "spin": None,
        }
        if obj.tebd_input_mode in ["strip"]:
            data.update({"embeddings_strip": obj.filter_layers_strip.serialize()})
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA1":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
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
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.se_atten.prec, device=env.DEVICE)

        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )
        obj.se_atten["davg"] = t_cvt(variables["davg"])
        obj.se_atten["dstd"] = t_cvt(variables["dstd"])
        obj.se_atten.filter_layers = NetworkCollection.deserialize(embeddings)
        if tebd_input_mode in ["strip"]:
            obj.se_atten.filter_layers_strip = NetworkCollection.deserialize(
                embeddings_strip
            )
        obj.se_atten.dpa1_attention = NeighborGatedAttention.deserialize(
            attention_layers
        )
        return obj

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Receive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        # do some checks before the mocel compression process
        if self.compress:
            raise ValueError("Compression is already enabled.")
        assert not self.se_atten.resnet_dt, (
            "Model compression error: descriptor resnet_dt must be false!"
        )
        for tt in self.se_atten.exclude_types:
            if (tt[0] not in range(self.se_atten.ntypes)) or (
                tt[1] not in range(self.se_atten.ntypes)
            ):
                raise RuntimeError(
                    "exclude types"
                    + str(tt)
                    + " must within the number of atomic types "
                    + str(self.se_atten.ntypes)
                    + "!"
                )
        if (
            self.se_atten.ntypes * self.se_atten.ntypes
            - len(self.se_atten.exclude_types)
            == 0
        ):
            raise RuntimeError(
                "Empty embedding-nets are not supported in model compression!"
            )

        if self.se_atten.attn_layer != 0:
            raise RuntimeError("Cannot compress model when attention layer is not 0.")

        if self.tebd_input_mode != "strip":
            raise RuntimeError("Cannot compress model when tebd_input_mode == 'concat'")

        data = self.serialize()
        self.table = DPTabulate(
            self,
            data["neuron"],
            data["type_one_side"],
            data["exclude_types"],
            ActivationFn(data["activation_function"]),
        )
        self.table_config = [
            table_extrapolate,
            table_stride_1,
            table_stride_2,
            check_frequency,
        ]
        self.lower, self.upper = self.table.build(
            min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
        )

        self.se_atten.enable_compression(
            self.table.data, self.table_config, self.lower, self.upper
        )
        self.compress = True

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
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
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        g1_ext = self.type_embedding(extended_atype)
        g1_inp = g1_ext[:, :nloc, :]
        if self.tebd_input_mode in ["strip"]:
            type_embedding = self.type_embedding.get_full_embedding(g1_ext.device)
        else:
            type_embedding = None
        g1, g2, h2, rot_mat, sw = self.se_atten(
            nlist,
            extended_coord,
            extended_atype,
            g1_ext,
            mapping=None,
            type_embedding=type_embedding,
        )
        if self.concat_output_tebd:
            g1 = torch.cat([g1, g1_inp], dim=-1)

        return (
            g1.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            rot_mat.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            g2.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION) if g2 is not None else None,
            h2.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            sw.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
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
