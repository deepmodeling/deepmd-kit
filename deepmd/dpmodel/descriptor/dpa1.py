# SPDX-License-Identifier: LGPL-3.0-or-later
import math
import warnings
from collections.abc import (
    Callable,
)
from typing import (
    Any,
    NoReturn,
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_take_along_axis,
    xp_take_first_n,
)
from deepmd.dpmodel.common import (
    cast_precision,
    get_xp_precision,
    to_numpy_array,
    to_numpy_dtype,
)
from deepmd.dpmodel.utils import (
    EmbeddingNet,
    EnvMat,
    NetworkCollection,
    PairExcludeMask,
    tabulate_fusion,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.dpmodel.utils.network import (
    LayerNorm,
    NativeLayer,
)
from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
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
from deepmd.utils.tabulate_math import (
    DPTabulate,
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


def np_softmax(x: Array, axis: int = -1) -> Array:
    xp = array_api_compat.array_namespace(x)
    # x = xp.nan_to_num(x)  # to avoid value warning
    x = xp.where(xp.isnan(x), xp.zeros_like(x), x)
    e_x = xp.exp(x - xp.max(x, axis=axis, keepdims=True))
    return e_x / xp.sum(e_x, axis=axis, keepdims=True)


def np_normalize(x: Array, axis: int = -1) -> Array:
    xp = array_api_compat.array_namespace(x)
    return x / xp.linalg.vector_norm(x, axis=axis, keepdims=True)


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

    _update_sel_cls = UpdateSel

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: list[int] | int,
        ntypes: int,
        neuron: list[int] = [25, 50, 100],
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
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: float | None = None,
        trainable_ln: bool = True,
        ln_eps: float | None = 1e-5,
        smooth_type_embedding: bool = True,
        concat_output_tebd: bool = True,
        spin: None = None,
        stripped_type_embedding: bool | None = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: list[str] | None = None,
        # consistent with argcheck, not used though
        seed: int | list[int] | None = None,
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
            trainable=trainable,
        )
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
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
            trainable=trainable,
        )
        self.tebd_dim = tebd_dim
        self.concat_output_tebd = concat_output_tebd
        self.trainable = trainable
        self.precision = precision
        self.tebd_compress = False
        self.geo_compress = False
        self.compress = False

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

    def has_message_passing_across_ranks(self) -> bool:
        """Returns whether per-layer node embeddings need MPI ghost exchange.

        DPA1 (se_atten) is single-layer and does not exchange features
        across ranks; same as the base se_e2_a path.
        """
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist when using `forward_lower`."""
        return self.se_atten.need_sorted_nlist_for_lower()

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.se_atten.get_env_protection()

    def get_numb_attn_layer(self) -> int:
        """Returns the number of se_atten attention layers."""
        return self.se_atten.attn_layer

    def uses_graph_lower(self) -> bool:
        """Returns whether this descriptor supports the graph-native lower.

        The graph-native lower (``call_graph``) covers the factorizable path
        AND transformer attention (``attn_layer >= 0``, NeighborGraph PR-D)
        with concat OR strip type-embedding. Remaining ineligible configs
        (``exclude_types``, and compressed descriptors) fall back to the legacy
        dense path, so those models keep working unchanged.

        Eligibility does NOT imply numerical interchangeability with the
        dense route for every config: with ``smooth_type_embedding=True``
        the carry-all graph attention is sel-independent by design and
        differs from the dense lower by up to ~1e-4 (see the Notes of
        :meth:`call_graph`).
        """
        # compressed descriptors have no graph kernel (geo/tebd tabulation is
        # dense-only); keep them on the legacy dense path.
        if self.compress:
            return False
        # exclude_types stays dense (graph exclusion is owned elsewhere); strip is
        # now graph-eligible (per-edge factorized embedding, no neighbor coupling).
        return (
            self.se_atten.tebd_input_mode in ("concat", "strip")
            and not self.se_atten.exclude_types
        )

    def share_params(
        self, base_class: "DescrptDPA1", shared_level: int, resume: bool = False
    ) -> NoReturn:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
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
        mean: Array,
        stddev: Array,
    ) -> None:
        """Update mean and stddev for descriptor."""
        self.se_atten.mean = mean
        self.se_atten.stddev = stddev

    def get_stat_mean_and_stddev(self) -> tuple[Array, Array]:
        """Get mean and stddev for descriptor."""
        return self.se_atten.mean, self.se_atten.stddev

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: Optional["DescrptDPA1"] = None,
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

    @cast_precision
    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> Array:
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
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nloc = nlist.shape[1]
        nall = xp.reshape(coord_ext, (nlist.shape[0], -1)).shape[1] // 3
        # graph-eligible configs route through the graph-native adapter (decision
        # #14: graph = single math source, dense call = thin adapter). Ineligible
        # configs (exclude_types, compressed descriptors) and the ghost case with
        # no mapping fall back to the legacy dense body. The graph needs `mapping`
        # to fold ghosts to local owners; without it only nall == nloc is valid.
        if self.uses_graph_lower() and (mapping is not None or nall == nloc):
            return self._call_graph_adapter(coord_ext, atype_ext, nlist, mapping)
        else:
            return self._call_dense(coord_ext, atype_ext, nlist)

    def _call_graph_adapter(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None,
    ) -> Array:
        """Regime-1 dense->graph adapter (the eligible ``call`` path).

        Builds a NeighborGraph from the dense quartet with the SHAPE-STATIC
        converter (``compact=False``, so this is jit/export-traceable -- no
        ``nonzero``), runs :meth:`call_graph`, and reconstructs the dense-shaped
        ``sw``. Preserves the dense 5-tuple ABI exactly; masked invalid edges
        contribute zero in ``call_graph``'s ``segment_sum`` so the output is
        identical to the legacy dense body.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nall x 3)
        atype_ext
            The extended atom types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping from extended to local region. shape: nf x nall.
            ``None`` is allowed only when nall == nloc (identity mapping).

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant single-particle representation.
            shape: nf x nloc x ng x 3
        g2
            ``None`` for this descriptor.
        h2
            ``None`` for this descriptor.
        sw
            The smooth switch function. shape: nf x nloc x nnei x 1
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            from_dense_quartet,
        )

        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        dev = array_api_compat.device(coord_ext)
        nf, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nf, -1)).shape[1] // 3
        coord_ext_3 = xp.reshape(coord_ext, (nf, nall, 3))
        if mapping is None:
            # default identity mapping (ext == loc, e.g. no-PBC nall == nloc)
            mapping_g = xp.broadcast_to(
                xp.arange(nall, dtype=xp.int64, device=dev)[None, :], (nf, nall)
            )
        else:
            mapping_g = xp.reshape(mapping, (nf, nall))
        graph = from_dense_quartet(
            coord_ext_3, nlist, mapping_g, layout=None, compact=False
        )
        # local atom types, flat (nf * nloc,)
        atype_local = xp.reshape(xp_take_first_n(atype_ext, 1, nloc), (nf * nloc,))
        grrg_flat, rot_mat_flat = self.call_graph(
            graph,
            atype_local,
            type_embedding=self.type_embedding.call(),
            # the adapter graph is shape-static center-major (compact=False):
            # keep the attention pair enumeration nonzero-free (traceable)
            static_nnei=nnei,
        )
        # call_graph returns flat (N, ...) node axis; reshape to (nf, nloc, ...)
        # for the dense 5-tuple ABI -- this reshape is LOCAL to the adapter shim.
        grrg = xp.reshape(grrg_flat, (nf, nloc, *grrg_flat.shape[1:]))
        rot_mat = xp.reshape(rot_mat_flat, (nf, nloc, *rot_mat_flat.shape[1:]))
        # reconstruct the dense-shaped sw the dense way (env_mat switch masked
        # where nlist == -1; the graph path forbids exclude_types, so nlist_mask
        # == nlist != -1, matching DescrptBlockSeAtten.call). A dense-layout
        # artifact tied to neighbor slots, which the graph does not carry.
        _, _, sw = self.se_atten.env_mat.call(
            coord_ext,
            atype_ext,
            nlist,
            self.se_atten.mean[...],
            self.se_atten.stddev[...],
        )
        nlist_mask = (nlist != -1)[:, :, :, None]
        sw = xp.where(nlist_mask, sw, xp.zeros_like(sw))
        sw = xp.reshape(sw, (nf, nloc, nnei, 1))
        return grrg, rot_mat, None, None, sw

    def _call_dense(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
    ) -> Array:
        """Legacy dense descriptor body (the ineligible ``call`` path: attention,
        strip tebd, exclude_types, or the no-mapping ghost case).

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nall x 3)
        atype_ext
            The extended atom types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant single-particle representation.
            shape: nf x nloc x ng x 3
        g2
            ``None`` for this descriptor.
        h2
            ``None`` for this descriptor.
        sw
            The smooth switch function. shape: nf x nloc x nnei x 1
        """
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nf, nloc = nlist.shape[:2]
        nall = xp.reshape(coord_ext, (nf, -1)).shape[1] // 3
        type_embedding = self.type_embedding.call()
        # nf x nall x tebd_dim
        atype_embd_ext = xp.reshape(
            xp.take(type_embedding, xp.reshape(atype_ext, (-1,)), axis=0),
            (nf, nall, self.tebd_dim),
        )
        # nfnl x tebd_dim
        atype_embd = xp_take_first_n(atype_embd_ext, 1, nloc)
        grrg, g2, h2, rot_mat, sw = self.se_atten(
            nlist,
            coord_ext,
            atype_ext,
            atype_embd_ext,
            mapping=None,
            type_embedding=type_embedding,
        )
        # nf x nloc x (ng x ng1 + tebd_dim)
        if self.concat_output_tebd:
            grrg = xp.concat(
                [grrg, xp.reshape(atype_embd, (nf, nloc, self.tebd_dim))], axis=-1
            )
        return grrg, rot_mat, None, None, sw

    def call_graph(
        self,
        graph: Any,
        atype: Array,
        type_embedding: Array | None = None,
        static_nnei: int | None = None,
    ) -> tuple[Array, Array]:
        """Descriptor-level graph-native forward.

        Wraps the block kernel
        :meth:`DescrptBlockSeAtten.call_graph`, adds the descriptor-level
        ``concat_output_tebd`` step, and returns the outputs on the flat ``(N,
        ...)`` node axis (ragged-native; no rectangular ``(nf, nloc)``
        reshape).

        This method is graph-native: it takes no dense quartet inputs and does
        not produce the dense ``sw`` (that lives in the dense :meth:`call`
        adapter, which has the ``nlist``/``coord_ext`` needed to build it).

        Notes
        -----
        **Smooth attention is intentionally sel-independent on the graph
        path.** For ``smooth_type_embedding=True`` the legacy dense attention
        keeps the sel-padding slots in its softmax DENOMINATOR (phantom
        ``exp(-attnw_shift)`` terms), which makes dense output depend on the
        ``sel`` setting by up to ~1e-4 even for identical physical neighbors.
        A carry-all graph has no padding slots, so its softmax runs over the
        real neighbor pairs only: cleaner, sel-independent semantics that
        deliberately DIFFER from the dense route for smooth models. The two
        routes agree bit-tight only for ``smooth_type_embedding=False`` (at
        non-binding ``sel``), or when this kernel is realized on a dense
        layout via ``static_nnei`` (the dense :meth:`call` adapter), which
        reproduces the phantom terms for exact backward compatibility.

        Parameters
        ----------
        graph
            A :class:`~deepmd.dpmodel.utils.neighbor_graph.NeighborGraph`.
        atype
            (N,) flat LOCAL atom types where ``N = sum(n_node)``.
        type_embedding
            (ntypes_with_padding, tebd_dim) type-embedding table.

        Returns
        -------
        grrg : Array
            (N, ng * axis_neuron [+ tebd_dim]) descriptor, flat node axis.
        rot_mat : Array
            (N, ng, 3) equivariant single-particle representation, flat node
            axis.
        """
        import dataclasses

        xp = array_api_compat.array_namespace(graph.edge_vec)
        dev = array_api_compat.device(graph.edge_vec)
        # manual @cast_precision: the decorator casts array ARGUMENTS, but the
        # graph's only float input (edge_vec) is inside the NeighborGraph
        # dataclass, invisible to it. Cast edge_vec down to the descriptor
        # precision on entry and the outputs back to the caller's dtype on
        # exit (differentiable: grad still flows to the caller's edge_vec leaf).
        in_dtype = graph.edge_vec.dtype
        prec = get_xp_precision(xp, self.precision)
        if in_dtype != prec:
            graph = dataclasses.replace(graph, edge_vec=xp.astype(graph.edge_vec, prec))
        grrg, rot_mat = self.se_atten.call_graph(
            graph, atype, type_embedding=type_embedding, static_nnei=static_nnei
        )
        # FLAT node axis (N, ...): no (nf, nloc) reshape -- ragged-native, spec.
        if self.concat_output_tebd:
            # Use type_embedding directly (mirrors the dense path's
            # ``xp.take(type_embedding, ...)``): ``xp.asarray(..., device=dev)``
            # DETACHES under torch, silently severing the type-embedding weight
            # gradient so the tebd net never trains; type_embedding already lives
            # on the model device, so the device cast was redundant anyway.
            atype_local = xp.asarray(atype, device=dev)
            atype_embd = xp.take(type_embedding, atype_local, axis=0)  # (N, tebd_dim)
            grrg = xp.concat([grrg, atype_embd], axis=-1)
        if in_dtype != prec:
            grrg = xp.astype(grrg, in_dtype)
            rot_mat = xp.astype(rot_mat, in_dtype)
        return grrg, rot_mat

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Enable descriptor compression.

        For DPA-1, compression is available for stripped type embeddings. The
        type embedding branch is always precomputed; the radial embedding table
        is enabled only when there is no attention layer, matching the PT/TF
        compression semantics.
        """
        if self.compress:
            raise ValueError("Compression is already enabled.")
        if self.se_atten.tebd_input_mode != "strip":
            raise RuntimeError("Type embedding compression only works in strip mode")
        if self.se_atten.resnet_dt:
            raise RuntimeError(
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

        self.se_atten.type_embedding_compression(self.type_embedding)
        self.type_embd_data = self.se_atten.type_embd_data
        self.tebd_compress = True
        self.compress = True

        if self.se_atten.attn_layer == 0:
            table = DPTabulate(
                self,
                self.se_atten.neuron,
                self.se_atten.type_one_side,
                self.se_atten.exclude_types,
                self.se_atten.activation_function,
            )
            table_config = [
                table_extrapolate,
                table_stride_1,
                table_stride_2,
                check_frequency,
            ]
            lower, upper = table.build(
                min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
            )
            self.se_atten.enable_compression(
                table.data,
                table_config,
                lower,
                upper,
            )
            self.compress_data = self.se_atten.compress_data
            self.compress_info = self.se_atten.compress_info
            self.geo_compress = True
        else:
            self.geo_compress = False
            warnings.warn(
                "Attention layer is not 0, only type embedding is compressed. "
                "Geometric part is not compressed.",
                UserWarning,
                stacklevel=2,
            )

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        obj = self.se_atten
        data = {
            "@class": "Descriptor",
            "type": "dpa1",
            "@version": 3 if self.compress else 2,
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
            "use_tebd_bias": self.use_tebd_bias,
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
                "davg": to_numpy_array(obj["davg"]),
                "dstd": to_numpy_array(obj["dstd"]),
            },
            ## to be updated when the options are supported.
            "trainable": self.trainable,
            "spin": None,
        }
        if obj.tebd_input_mode in ["strip"]:
            data.update({"embeddings_strip": obj.embeddings_strip.serialize()})
        if self.compress:
            type_embd_data = (
                self.type_embd_data
                if hasattr(self, "type_embd_data")
                else obj.type_embd_data
            )
            compress_dict: dict = {
                "@variables": {
                    "type_embd_data": to_numpy_array(type_embd_data),
                },
                "geo_compress": self.geo_compress,
            }
            if self.geo_compress:
                compress_data = (
                    self.compress_data
                    if hasattr(self, "compress_data")
                    else obj.compress_data
                )
                compress_info = (
                    self.compress_info
                    if hasattr(self, "compress_info")
                    else obj.compress_info
                )
                compress_dict["@variables"]["compress_data"] = [
                    to_numpy_array(d) for d in compress_data
                ]
                compress_dict["@variables"]["compress_info"] = [
                    to_numpy_array(i) for i in compress_info
                ]
            data["compress"] = compress_dict
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptDPA1":
        """Deserialize from dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 3, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        type_embedding = data.pop("type_embedding")
        attention_layers = data.pop("attention_layers")
        env_mat = data.pop("env_mat")
        compress = data.pop("compress", None)
        tebd_input_mode = data["tebd_input_mode"]
        if tebd_input_mode in ["strip"]:
            embeddings_strip = data.pop("embeddings_strip")
        else:
            embeddings_strip = None
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
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
        if compress is not None:
            obj._load_compress_data(compress)
        return obj

    def _load_compress_data(self, compress: dict) -> None:
        """Load compression state from serialized data."""
        variables = compress["@variables"]
        self.type_embd_data = variables["type_embd_data"]
        self.geo_compress = compress.get("geo_compress", False)
        self.tebd_compress = True
        self.se_atten.type_embd_data = self.type_embd_data
        self.se_atten.tebd_compress = True
        self.se_atten.geo_compress = self.geo_compress
        if self.geo_compress:
            self.compress_data = variables["compress_data"]
            self.compress_info = variables["compress_info"]
            self.se_atten.compress_data = self.compress_data
            self.se_atten.compress_info = self.compress_info
        self.compress = True

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[Array, Array]:
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
        min_nbor_dist, sel = cls._update_sel_cls().update_one_sel(
            train_data, type_map, local_jdata_cpy["rcut"], local_jdata_cpy["sel"], True
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist


@DescriptorBlock.register("se_atten")
class DescrptBlockSeAtten(NativeOP, DescriptorBlock):
    r"""The attention-based descriptor block.

    This block computes an embedding matrix using attention mechanism and type embedding.
    The descriptor is computed as:

    .. math::
        \mathcal{D}^i = \frac{1}{N_c^2}(\hat{\mathcal{G}}^i)^T \mathcal{R}^i (\mathcal{R}^i)^T \hat{\mathcal{G}}^i_<,

    where :math:`\hat{\mathcal{G}}^i` is the embedding matrix after self-attention layers,
    :math:`\mathcal{R}^i` is the coordinate matrix, and :math:`\hat{\mathcal{G}}^i_<` denotes
    the first `axis_neuron` columns of :math:`\hat{\mathcal{G}}^i`.

    The embedding matrix :math:`\mathcal{G}^i` is computed by:

    .. math::
        (\mathcal{G}^i)_j = \mathcal{N}(s(r_{ji}), \mathcal{T}_i, \mathcal{T}_j),

    where :math:`\mathcal{N}` is the embedding network, :math:`s(r_{ji})` is the smoothed
    radial distance, and :math:`\mathcal{T}` denotes type embedding.

    Parameters
    ----------
    rcut : float
        The cut-off radius.
    rcut_smth : float
        Where to start smoothing.
    sel : Union[list[int], int]
        Maximally possible number of selected neighbors.
    ntypes : int
        Number of element types.
    neuron : list[int], optional
        Number of neurons in each hidden layer of the embedding net.
    axis_neuron : int, optional
        Size of the submatrix of the embedding matrix.
    tebd_dim : int, optional
        Dimension of the type embedding.
    tebd_input_mode : str, optional
        The input mode of the type embedding. Supported modes are ["concat", "strip"].
    resnet_dt : bool, optional
        Time-step `dt` in the resnet construction.
    type_one_side : bool, optional
        If True, only type embeddings of neighbor atoms are considered.
    attn : int, optional
        Hidden dimension of the attention vectors.
    attn_layer : int, optional
        Number of attention layers.
    attn_dotr : bool, optional
        If True, dot the angular gate to the attention weights.
    attn_mask : bool, optional
        If True, mask the diagonal of attention weights.
    exclude_types : list[tuple[int, int]], optional
        The excluded pairs of types which have no interaction.
    env_protection : float, optional
        Protection parameter to prevent division by zero.
    set_davg_zero : bool, optional
        Set the shift of embedding net input to zero.
    activation_function : str, optional
        The activation function in the embedding net.
    precision : str, optional
        The precision of the embedding net parameters.
    scaling_factor : float, optional
        The scaling factor of normalization in attention weights calculation.
    normalize : bool, optional
        Whether to normalize the hidden vectors in attention weights calculation.
    temperature : float, optional
        If not None, the scaling of attention weights is `temperature` itself.
    trainable_ln : bool, optional
        Whether to use trainable shift and scale weights in layer normalization.
    ln_eps : float, optional
        The epsilon value for layer normalization.
    smooth : bool, optional
        Whether to use smoothness in attention weights calculation.
    seed : int, optional
        Random seed for parameter initialization.
    trainable : bool, optional
        If the parameters are trainable.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: list[int] | int,
        ntypes: int,
        neuron: list[int] = [25, 50, 100],
        axis_neuron: int = 8,
        tebd_dim: int = 8,
        tebd_input_mode: str = "concat",
        resnet_dt: bool = False,
        type_one_side: bool = False,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        scaling_factor: float = 1.0,
        normalize: bool = True,
        temperature: float | None = None,
        trainable_ln: bool = True,
        ln_eps: float | None = 1e-5,
        smooth: bool = True,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        if isinstance(sel, int):
            sel = [sel]
        self.sel = sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
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
            trainable=trainable,
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
                trainable=trainable,
            )
            self.embeddings_strip = embeddings_strip
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
            trainable=trainable,
        )

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.env_mat = EnvMat(self.rcut, self.rcut_smth, protection=self.env_protection)
        self.mean = np.zeros(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.orig_sel = self.sel
        self.tebd_compress = False
        self.geo_compress = False
        self.is_sorted = len(self.exclude_types) == 0
        self.compress_data = None
        self.compress_info = None
        self.type_embd_data = None

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

    def __setitem__(self, key: str, value: Array) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> Array:
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
    def dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_in(self) -> int:
        """Returns the atomic input dimension of this descriptor."""
        return self.tebd_dim

    @property
    def dim_emb(self) -> int:
        """Returns the output dimension of embedding."""
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
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
        device = array_api_compat.device(self.stddev)
        if not self.set_davg_zero:
            self.mean = xp.asarray(
                mean, dtype=self.mean.dtype, copy=True, device=device
            )
        self.stddev = xp.asarray(
            stddev, dtype=self.stddev.dtype, copy=True, device=device
        )

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
        self.is_sorted = len(self.exclude_types) == 0
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def cal_g(
        self,
        ss: Array,
        embedding_idx: int,
    ) -> Array:
        xp = array_api_compat.array_namespace(ss)
        nfnl, nnei = ss.shape[0:2]
        shape2 = math.prod(ss.shape[2:])
        ss = xp.reshape(ss, (nfnl, nnei, shape2))
        # nfnl x nnei x ng
        gg = self.embeddings[embedding_idx].call(ss)
        return gg

    def cal_g_strip(
        self,
        ss: Array,
        embedding_idx: int,
    ) -> Array:
        assert self.embeddings_strip is not None
        # nfnl x nnei x ng
        gg = self.embeddings_strip[embedding_idx].call(ss)
        return gg

    def enable_compression(
        self,
        table_data: dict[str, Array],
        table_config: list[int | float],
        lower: dict[str, int],
        upper: dict[str, int],
    ) -> None:
        """Store tabulated geometric embedding-net data."""
        net = "filter_net"
        dtype = to_numpy_dtype(self.mean.dtype)
        self.compress_info = [
            np.asarray(
                [
                    lower[net],
                    upper[net],
                    upper[net] * table_config[0],
                    table_config[1],
                    table_config[2],
                    table_config[3],
                ],
                dtype=dtype,
            )
        ]
        self.compress_data = [np.asarray(table_data[net], dtype=dtype)]
        self.geo_compress = True

    def type_embedding_compression(self, type_embedding_net: TypeEmbedNet) -> None:
        """Precompute stripped type embedding network outputs."""
        if self.tebd_input_mode != "strip":
            raise RuntimeError("Type embedding compression only works in strip mode")
        if self.embeddings_strip is None:
            raise RuntimeError(
                "embeddings_strip must be initialized for type embedding compression"
            )

        full_embd = type_embedding_net.call()
        xp = array_api_compat.array_namespace(full_embd)
        nt, t_dim = full_embd.shape
        if self.type_one_side:
            embd_tensor = self.embeddings_strip[0].call(full_embd)
        else:
            type_embedding_nei = xp.tile(
                xp.reshape(full_embd, (1, nt, t_dim)), (nt, 1, 1)
            )
            type_embedding_center = xp.tile(
                xp.reshape(full_embd, (nt, 1, t_dim)), (1, nt, 1)
            )
            two_side_type_embedding = xp.reshape(
                xp.concat([type_embedding_nei, type_embedding_center], axis=-1),
                (-1, t_dim * 2),
            )
            embd_tensor = self.embeddings_strip[0].call(two_side_type_embedding)
        self.type_embd_data = embd_tensor
        self.tebd_compress = True

    def _tabulate_fusion_se_atten(
        self,
        table: Array,
        table_info: Array,
        em_x: Array,
        em: Array,
        two_embed: Array,
        last_layer_size: int,
    ) -> Array:
        """Pure Array API implementation of tabulate_fusion_se_atten forward."""
        xp = array_api_compat.array_namespace(em_x, em, two_embed)
        values = tabulate_fusion(
            table,
            table_info,
            em_x,
            last_layer_size,
            reference=em,
        )
        values = values * two_embed + values
        return xp.sum(em[:, :, :, None] * values[:, :, None, :], axis=1)

    def call(
        self,
        nlist: Array,
        coord_ext: Array,
        atype_ext: Array,
        atype_embd_ext: Array | None = None,
        mapping: Array | None = None,
        type_embedding: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
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
        atype = xp_take_first_n(atype_ext, 1, nloc)
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        # nfnl x nnei
        exclude_mask = xp.reshape(exclude_mask, (nf * nloc, nnei))
        exclude_mask = xp.astype(exclude_mask, xp.bool)
        # nfnl x nnei
        nlist = xp.reshape(nlist, (nf * nloc, nnei))
        nlist = xp.where(exclude_mask, nlist, xp.full_like(nlist, -1))
        # nfnl x nnei x 4
        dmatrix = xp.reshape(dmatrix, (nf * nloc, nnei, 4))
        # nfnl x nnei x 1
        sw = xp.reshape(sw, (nf * nloc, nnei, 1))
        # nfnl x nnei
        nlist_mask = nlist != -1
        # nfnl x nnei x 1
        sw = xp.where(nlist_mask[:, :, None], sw, xp.full_like(sw, 0.0))
        nlist_masked = xp.where(nlist_mask, nlist, xp.zeros_like(nlist))
        ng = self.neuron[-1]
        nt = self.tebd_dim

        # Gather neighbor info using xp_take_along_axis along axis=1.
        # This avoids flat (nf*nall,) indexing that creates Ne(nall, nloc)
        # constraints in torch.export, breaking NoPbc (nall == nloc).
        nlist_2d = xp.reshape(nlist_masked, (nf, nloc * nnei))  # (nf, nloc*nnei)

        # nfnl x nnei x 4
        rr = xp.reshape(dmatrix, (nf * nloc, nnei, 4))
        rr = rr * xp.astype(exclude_mask[:, :, None], rr.dtype)
        # nfnl x nnei x 1
        ss = rr[..., 0:1]
        geo_gr = None
        if self.tebd_input_mode in ["concat"]:
            # nfnl x tebd_dim
            atype_embd = xp.reshape(
                xp_take_first_n(atype_embd_ext, 1, nloc), (nf * nloc, self.tebd_dim)
            )
            # nfnl x nnei x tebd_dim
            atype_embd_nnei = xp.tile(atype_embd[:, xp.newaxis, :], (1, nnei, 1))
            # Gather neighbor type embeddings: (nf, nall, tebd_dim) -> (nf, nloc*nnei, tebd_dim)
            nlist_idx_tebd = xp.tile(nlist_2d[:, :, xp.newaxis], (1, 1, self.tebd_dim))
            atype_embd_nlist = xp_take_along_axis(
                atype_embd_ext, nlist_idx_tebd, axis=1
            )
            # nfnl x nnei x tebd_dim
            atype_embd_nlist = xp.reshape(
                atype_embd_nlist, (nf * nloc, nnei, self.tebd_dim)
            )
            if not self.type_one_side:
                # nfnl x nnei x (1 + 2 * tebd_dim)
                ss = xp.concat([ss, atype_embd_nlist, atype_embd_nnei], axis=-1)
            else:
                # nfnl x nnei x (1 + tebd_dim)
                ss = xp.concat([ss, atype_embd_nlist], axis=-1)
                # calculate gg
                # nfnl x nnei x ng
            gg = self.cal_g(ss, 0)
        elif self.tebd_input_mode in ["strip"]:
            ss_scalar = ss
            assert self.embeddings_strip is not None
            assert type_embedding is not None
            ntypes_with_padding = type_embedding.shape[0]
            # Gather neighbor types: (nf, nall) -> (nf, nloc*nnei)
            nei_type = xp_take_along_axis(atype_ext, nlist_2d, axis=1)
            nei_type = xp.reshape(nei_type, (-1,))  # (nf * nloc * nnei,)
            # (nf x nl x nnei) x ng
            nei_type_index = xp.tile(xp.reshape(nei_type, (-1, 1)), (1, ng))
            if self.type_one_side:
                if self.tebd_compress:
                    tt_full = xp.asarray(
                        self.type_embd_data[...],
                        dtype=rr.dtype,
                        device=array_api_compat.device(rr),
                    )
                else:
                    tt_full = self.cal_g_strip(type_embedding, 0)
                # (nf x nl x nnei) x ng
                gg_t = xp_take_along_axis(tt_full, nei_type_index, axis=0)
            else:
                idx_i = xp.reshape(
                    xp.tile(
                        (xp.reshape(atype, (-1, 1)) * ntypes_with_padding), (1, nnei)
                    ),
                    (-1,),
                )
                idx_j = xp.reshape(nei_type, (-1,))
                # (nf x nl x nnei) x ng
                idx = xp.tile(xp.reshape((idx_i + idx_j), (-1, 1)), (1, ng))
                # Cast to int64 for PyTorch backend (take_along_dim requires Long indices)
                idx = xp.astype(idx, xp.int64)
                if self.tebd_compress:
                    tt_full = xp.asarray(
                        self.type_embd_data[...],
                        dtype=rr.dtype,
                        device=array_api_compat.device(rr),
                    )
                else:
                    # (ntypes) * ntypes * nt
                    type_embedding_nei = xp.tile(
                        xp.reshape(type_embedding, (1, ntypes_with_padding, nt)),
                        (ntypes_with_padding, 1, 1),
                    )
                    # ntypes * (ntypes) * nt
                    type_embedding_center = xp.tile(
                        xp.reshape(type_embedding, (ntypes_with_padding, 1, nt)),
                        (1, ntypes_with_padding, 1),
                    )
                    # (ntypes * ntypes) * (nt+nt)
                    two_side_type_embedding = xp.reshape(
                        xp.concat([type_embedding_nei, type_embedding_center], axis=-1),
                        (-1, nt * 2),
                    )
                    tt_full = self.cal_g_strip(two_side_type_embedding, 0)
                # (nf x nl x nnei) x ng
                gg_t = xp_take_along_axis(tt_full, idx, axis=0)
            # (nf x nl) x nnei x ng
            gg_t = xp.reshape(gg_t, (nf * nloc, nnei, ng))
            if self.smooth:
                gg_t = gg_t * xp.reshape(sw, (-1, self.nnei, 1))
            if self.geo_compress:
                geo_gr = self._tabulate_fusion_se_atten(
                    self.compress_data[0],
                    self.compress_info[0],
                    ss_scalar,
                    rr,
                    gg_t,
                    self.filter_neuron[-1],
                )
                gg = None
            else:
                # nfnl x nnei x ng
                gg_s = self.cal_g(ss_scalar, 0)
                gg = gg_s * gg_t + gg_s
        else:
            raise NotImplementedError

        if geo_gr is None:
            normed = safe_for_vector_norm(
                xp.reshape(rr, (-1, nnei, 4))[:, :, 1:4], axis=-1, keepdims=True
            )
            input_r = xp.reshape(rr, (-1, nnei, 4))[:, :, 1:4] / xp.maximum(
                normed,
                xp.full_like(normed, 1e-12),
            )
            gg = self.dpa1_attention(
                gg, nlist_mask, input_r=input_r, sw=sw
            )  # shape is [nframes*nloc, self.neei, out_size]
            # nfnl x ng x 4
            # gr = xp.einsum("lni,lnj->lij", gg, rr)
            gr = xp.sum(gg[:, :, :, None] * rr[:, :, None, :], axis=1)
            g2 = xp.reshape(gg, (nf, nloc, self.nnei, self.filter_neuron[-1]))
        else:
            gr = xp.permute_dims(geo_gr, (0, 2, 1))
            g2 = None
        gr /= self.nnei
        gr1 = gr[:, : self.axis_neuron, :]
        # nfnl x ng x ng1
        # grrg = xp.einsum("lid,ljd->lij", gr, gr1)
        grrg = xp.sum(gr[:, :, None, :] * gr1[:, None, :, :], axis=3)
        # nf x nloc x (ng x ng1)
        grrg = xp.astype(
            xp.reshape(grrg, (nf, nloc, ng * self.axis_neuron)), coord_ext.dtype
        )
        return (
            xp.reshape(grrg, (nf, nloc, self.filter_neuron[-1] * self.axis_neuron)),
            g2,
            xp.reshape(dmatrix, (nf, nloc, self.nnei, 4))[..., 1:],
            xp.reshape(gr[..., 1:], (nf, nloc, self.filter_neuron[-1], 3)),
            xp.reshape(sw, (nf, nloc, nnei, 1)),
        )

    def call_graph(
        self,
        graph: Any,
        atype: Array,
        type_embedding: Array | None = None,
        static_nnei: int | None = None,
    ) -> tuple[Array, Array]:
        """Graph-native forward.

        Bit-exact analogue of :meth:`call` on the SAME neighbor list, with the
        neighbor-axis reduction replaced by a ``segment_sum`` over edge centers
        (``dst``) and the dense ``(nnei, nnei)`` transformer attention replaced
        by pairs of edges sharing a center (``center_edge_pairs`` +
        ``segment_softmax``). Geometry enters only through ``graph.edge_vec``.

        Parameters
        ----------
        graph
            A :class:`~deepmd.dpmodel.utils.neighbor_graph.NeighborGraph` whose
            ``edge_index = [src, dst]`` (src = neighbor local owner, dst = center),
            ``edge_vec = r_src - r_dst`` and ``edge_mask`` marks real edges.
        atype
            (N,) flat node atom types (``N = sum(graph.n_node)``).
        type_embedding
            (ntypes_with_padding, tebd_dim) type-embedding table.
        static_nnei
            When the graph uses the shape-static center-major layout
            (``from_dense_quartet(compact=False)``, ``E = n_center * nnei``),
            pass ``nnei`` so the attention edge-pair enumeration stays
            jit/export-traceable (no ``nonzero``). ``None`` (carry-all /
            compact graphs) selects the dynamic eager form.

        Returns
        -------
        grrg : Array
            (N, ng * axis_neuron) per-node descriptor, matching the first output
            of :meth:`call` flattened over the (nf, nloc) axes.
        rot_mat : Array
            (N, ng, 3) per-node equivariant single-particle representation,
            matching ``gr[..., 1:]`` of :meth:`call` flattened over (nf, nloc).

        Notes
        -----
        Known limitations:
        - ``tebd_input_mode`` in {"concat", "strip"}; compressed descriptors stay dense;
        - ``exclude_types`` is not yet supported and raises (lands in a later PR).
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            edge_env_mat,
            segment_sum,
        )

        if self.tebd_input_mode not in ["concat", "strip"]:
            raise NotImplementedError(
                f"graph path does not support tebd_input_mode={self.tebd_input_mode!r}"
            )
        if self.exclude_types:
            raise NotImplementedError(
                "graph path does not yet apply exclude_types (NeighborGraph PR-A); "
                "type exclusion lands in a later PR"
            )
        if type_embedding is None:
            raise ValueError("type_embedding is required for the graph path")
        xp = array_api_compat.array_namespace(graph.edge_vec)
        dev = array_api_compat.device(graph.edge_vec)
        # N == sum(graph.n_node) by contract (atype is (N,)); use the static shape
        # value so the kernel stays jit/export-traceable (no concretize of n_node).
        n_total = atype.shape[0]
        src = graph.edge_index[0, :]
        dst = graph.edge_index[1, :]
        atype = xp.asarray(atype, device=dev)
        center_type = xp.take(atype, dst, axis=0)  # (E,)
        nei_type = xp.take(atype, src, axis=0)  # (E,)
        # per-edge env-mat 4-vector, normalized by the center (dst) atom type.
        # self.mean/self.stddev are slot-independent (ntypes, nnei, 4); slot 0 is
        # the canonical per-type vector.
        rr, sw_e = edge_env_mat(
            graph.edge_vec,
            center_type,
            self.mean[:, 0, :],
            self.stddev[:, 0, :],
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
            edge_mask=graph.edge_mask,
            return_sw=True,
        )  # (E, 4), (E, 1) sw zeroed on padding
        # radial channel
        ss = rr[:, 0:1]  # (E, 1)
        if self.tebd_input_mode == "concat":
            # neighbor / center type embeddings; ghost type == owner type so
            # gathering by the LOCAL owner (src) reproduces the dense neighbor tebd.
            # NB: do NOT wrap in ``xp.asarray(..., device=dev)`` -- that DETACHES
            # under torch and severs the type-embedding weight gradient (the tebd
            # net would never train); type_embedding already lives on the device.
            tebd = type_embedding
            atype_embd_nlist = xp.take(tebd, nei_type, axis=0)  # (E, tebd_dim)
            if not self.type_one_side:
                atype_embd_nnei = xp.take(tebd, center_type, axis=0)  # (E, tebd_dim)
                ss = xp.concat([ss, atype_embd_nlist, atype_embd_nnei], axis=-1)
            else:
                ss = xp.concat([ss, atype_embd_nlist], axis=-1)
            # embedding net (same weights as the dense path); applies on last axis
            gg = self.embeddings[0].call(ss)  # (E, ng)
        else:  # strip: factorized gg_s*gg_t + gg_s (per-edge; no neighbor coupling)
            gg = self._graph_edge_gg_strip(
                ss, center_type, nei_type, type_embedding, sw_e
            )
        # transformer attention over each center's edges — mirrors the dense
        # self.dpa1_attention(gg, nlist_mask, input_r, sw), which also runs on
        # the UNMASKED gg (padding rows are neutralized afterwards).
        if self.attn_layer > 0:
            gg = self._graph_attention(
                gg, rr, dst, n_total, graph.edge_mask, sw_e, static_nnei
            )
        # zero padding/guard edges BEFORE the segment sum
        gg = gg * xp.astype(graph.edge_mask[:, None], gg.dtype)
        # outer product (replaces the dense gg[:,:,:,None] * rr[:,:,None,:])
        outer = gg[:, :, None] * rr[:, None, :]  # (E, ng, 4)
        # neighbor-axis reduction -> segment_sum over centers; divide by nnei
        gr = segment_sum(outer, dst, n_total) / self.nnei  # (N, ng, 4)
        gr1 = gr[:, : self.axis_neuron, :]
        # nf x nloc x (ng x ng1)
        grrg = xp.sum(gr[:, :, None, :] * gr1[:, None, :, :], axis=3)  # (N, ng, ng1)
        ng = self.neuron[-1]
        grrg = xp.astype(
            xp.reshape(grrg, (n_total, ng * self.axis_neuron)),
            graph.edge_vec.dtype,
        )
        # equivariant single-particle representation, dense-ABI slice gr[..., 1:]
        # (N, ng, 3); not cast, mirroring the dense block which leaves rot_mat in
        # the working precision before the descriptor-level @cast_precision.
        rot_mat = gr[:, :, 1:]
        return grrg, rot_mat

    def _graph_edge_gg_strip(
        self,
        ss: Array,
        center_type: Array,
        nei_type: Array,
        type_embedding: Array,
        sw_e: Array,
    ) -> Array:
        """Per-edge stripped-tebd embedding, op-for-op vs the dense strip branch.

        Mirrors the ``tebd_input_mode == "strip"`` block of :meth:`call`: the
        geometric net runs on the radial channel only (``gg_s``), the stripped
        type-embedding net produces a per-type(-pair) factor (``gg_t``,
        optionally switch-smoothed), and the two combine as
        ``gg_s * gg_t + gg_s``. The compression branches (geo/tebd) are NOT
        reached on the graph route: :meth:`DescrptDPA1.uses_graph_lower`
        excludes compressed descriptors, so this kernel assumes no compression.

        Parameters
        ----------
        ss
            (E, 1) per-edge radial channel (``rr[:, 0:1]``).
        center_type
            (E,) center (dst) LOCAL atom type of each edge.
        nei_type
            (E,) neighbor (src) LOCAL atom type of each edge.
        type_embedding
            (ntypes_with_padding, tebd_dim) type-embedding table.
        sw_e
            (E, 1) smooth switch, zeroed on padding edges.

        Returns
        -------
        gg
            (E, ng) per-edge embedding feeding the attention / segment_sum.
        """
        assert self.embeddings_strip is not None
        xp = array_api_compat.array_namespace(ss)
        nt = self.tebd_dim
        ntypes_with_padding = type_embedding.shape[0]
        # geometric net on the radial channel only (dense: gg_s = cal_g(ss_scalar))
        gg_s = self.embeddings[0].call(ss)  # (E, ng)
        if self.type_one_side:
            # one-side strip table indexed by NEIGHBOR type only
            tt_full = self.cal_g_strip(type_embedding, 0)  # (ntypes_pad, ng)
            gg_t = xp.take(tt_full, nei_type, axis=0)  # (E, ng)
        else:
            # two-side type-pair table; row = center * ntypes_pad + nei
            # (dense builds the same (ntypes_pad**2, 2*nt) table, nei-fastest).
            type_embedding_nei = xp.tile(
                xp.reshape(type_embedding, (1, ntypes_with_padding, nt)),
                (ntypes_with_padding, 1, 1),
            )
            type_embedding_center = xp.tile(
                xp.reshape(type_embedding, (ntypes_with_padding, 1, nt)),
                (1, ntypes_with_padding, 1),
            )
            two_side_type_embedding = xp.reshape(
                xp.concat([type_embedding_nei, type_embedding_center], axis=-1),
                (-1, nt * 2),
            )
            tt_full = self.cal_g_strip(
                two_side_type_embedding, 0
            )  # (ntypes_pad**2, ng)
            # int64 for torch take (take_along/take requires Long indices)
            idx = xp.astype(center_type * ntypes_with_padding + nei_type, xp.int64)
            gg_t = xp.take(tt_full, idx, axis=0)  # (E, ng)
        if self.smooth:
            # dense: gg_t = gg_t * sw (per-neighbor); sw_e is (E, 1), zeroed on padding
            gg_t = gg_t * sw_e
        return gg_s * gg_t + gg_s

    def _graph_attention(
        self,
        gg: Array,
        rr: Array,
        dst: Array,
        n_total: int,
        edge_mask: Array,
        sw_e: Array,
        static_nnei: int | None,
    ) -> Array:
        """Graph-native transformer attention over each center's edges.

        Ragged reproduction of :class:`NeighborGatedAttention` /
        :class:`GatedAttentionLayer`: edges sharing a center attend to each
        other. The dense ``(nnei, nnei)`` square per center becomes the
        edge-pair axis from ``center_edge_pairs(ordered=True,
        include_self=True)``; softmax over the key axis becomes
        ``segment_softmax`` grouped by the query edge.

        Parameters
        ----------
        gg : (E, ng) per-edge embedding (UNMASKED, as in the dense path).
        rr : (E, 4) per-edge env-mat vector (``rr[:, 1:4]`` carries direction).
        dst : (E,) center of each edge.
        n_total : number of centers.
        edge_mask : (E,) real-vs-padding edge mask.
        sw_e : (E, 1) smooth switch, zeroed on padding edges.
        static_nnei : shape-static layout ``nnei`` or ``None`` (compact eager).
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            center_edge_pairs,
        )

        xp = array_api_compat.array_namespace(gg)
        # per-edge normalized direction (mirrors the dense input_r,
        # rr[..., 1:4] / max(|rr[..., 1:4]|, 1e-12))
        dir3 = rr[:, 1:4]
        normed = safe_for_vector_norm(dir3, axis=-1, keepdims=True)
        input_r = dir3 / xp.maximum(normed, xp.full_like(normed, 1e-12))  # (E, 3)
        # transformer neighbor-pairs: full ordered square incl. the diagonal
        # (q_m . k_n is not symmetric and self-attention keeps m == n)
        q_e, k_e, pair_mask = center_edge_pairs(
            dst,
            edge_mask,
            n_total,
            include_self=True,
            ordered=True,
            static_nnei=static_nnei,
        )
        for layer in self.dpa1_attention.attention_layers:
            gg = self._graph_attention_one_layer(
                layer, gg, input_r, sw_e, q_e, k_e, pair_mask
            )
        return gg

    def _graph_attention_one_layer(
        self,
        layer: "NeighborGatedAttentionLayer",
        gg: Array,
        input_r: Array,
        sw_e: Array,
        q_e: Array,
        k_e: Array,
        pair_mask: Array,
    ) -> Array:
        """One residual attention layer, op-for-op vs the dense reference.

        Mirrors ``NeighborGatedAttentionLayer.call`` (residual +
        ``GatedAttentionLayer.call`` + LayerNorm). Structural translation:
        per-center ``q @ k^T`` -> per-pair ``q_m . k_n``; softmax over the key
        axis -> ``segment_softmax`` grouped by the query edge. The smooth
        branch keeps padding pairs IN the softmax denominator with ``sw = 0``
        (weight ``exp(-attnw_shift)``), exactly like the dense branch, which
        replaces the ``-inf`` masking by the switch weighting.
        """
        from deepmd.dpmodel.utils.neighbor_graph import (
            segment_softmax,
            segment_sum,
        )

        xp = array_api_compat.array_namespace(gg)
        e_tot = gg.shape[0]
        gal = layer.attention_layer  # GatedAttentionLayer
        if gal.num_heads != 1:
            raise NotImplementedError(
                "graph attention assumes num_heads == 1 (dpa1 never exposes "
                "num_heads; the dense head_dim QKV slicing relies on it)"
            )
        hd = gal.head_dim  # == hidden_dim for num_heads == 1
        residual = gg
        # in_proj -> Q, K, V; mirror the dense HEAD_DIM slicing exactly
        qkv = gal.in_proj.call(gg)  # (E, 3 * hidden)
        q = qkv[:, 0:hd]
        k = qkv[:, hd : hd * 2]
        v = qkv[:, hd * 2 : hd * 3]
        if gal.normalize:
            q = np_normalize(q, axis=-1)
            k = np_normalize(k, axis=-1)
            v = np_normalize(v, axis=-1)
        q = q * gal.scaling
        # per-pair logits q_m . k_n (num_heads == 1)
        logits = xp.sum(
            xp.take(q, q_e, axis=0) * xp.take(k, k_e, axis=0), axis=-1
        )  # (P,)
        if gal.smooth:
            # (logits + shift) * sw_m * sw_n - shift, then softmax WITHOUT the
            # pair mask: padding pairs stay in the denominator at exp(-shift),
            # mirroring the dense smooth branch (sw already zeroed on padding).
            attnw_shift = 20.0  # dense GatedAttentionLayer.call default
            sw_flat = sw_e[:, 0]  # (E,)
            sw_q = xp.take(sw_flat, q_e, axis=0)
            sw_k = xp.take(sw_flat, k_e, axis=0)
            logits = (logits + attnw_shift) * sw_q * sw_k - attnw_shift
            w = segment_softmax(logits, q_e, e_tot)  # (P,)
            w = w * sw_q * sw_k
        else:
            # non-smooth: dense masks padding keys to -inf pre-softmax ==
            # excluding them from the softmax entirely
            w = segment_softmax(logits, q_e, e_tot, mask=pair_mask)
        if gal.dotr:
            angular = xp.sum(
                xp.take(input_r, q_e, axis=0) * xp.take(input_r, k_e, axis=0),
                axis=-1,
            )  # (P,) = input_r_m . input_r_n
            w = w * angular
        # o_m = sum_n w[m, n] v[n] -> segment_sum over the query edge
        wv = w[:, None] * xp.take(v, k_e, axis=0)  # (P, hd)
        o = segment_sum(wv, q_e, e_tot)  # (E, hd)
        out = gal.out_proj.call(o)  # (E, ng)
        x = residual + out
        return layer.attn_layer_norm.call(x)

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
            "@class": "DescriptorBlock",
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
            "attn_mask": obj.attn_mask,
            "activation_function": obj.activation_function,
            "resnet_dt": obj.resnet_dt,
            "scaling_factor": obj.scaling_factor,
            "normalize": obj.normalize,
            "temperature": obj.temperature,
            "trainable_ln": obj.trainable_ln,
            "ln_eps": obj.ln_eps,
            "smooth": obj.smooth,
            "type_one_side": obj.type_one_side,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[obj.precision]).name,
            "embeddings": obj.embeddings.serialize(),
            "attention_layers": obj.dpa1_attention.serialize(),
            "env_mat": obj.env_mat.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": to_numpy_array(obj["davg"]),
                "dstd": to_numpy_array(obj["dstd"]),
            },
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
        attention_layers = data.pop("attention_layers")
        env_mat = data.pop("env_mat")
        tebd_input_mode = data["tebd_input_mode"]
        if tebd_input_mode in ["strip"]:
            embeddings_strip = data.pop("embeddings_strip")
        else:
            embeddings_strip = None
        obj = cls(**data)

        obj["davg"] = variables["davg"]
        obj["dstd"] = variables["dstd"]
        obj.embeddings = NetworkCollection.deserialize(embeddings)
        if tebd_input_mode in ["strip"]:
            obj.embeddings_strip = NetworkCollection.deserialize(embeddings_strip)
        obj.dpa1_attention = NeighborGatedAttention.deserialize(attention_layers)
        return obj


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
        temperature: float | None = None,
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
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
                trainable=trainable,
            )
            for ii in range(layer_num)
        ]

    def call(
        self,
        input_G: Array,
        nei_mask: Array,
        input_r: Array | None = None,
        sw: Array | None = None,
    ) -> Array:
        out = input_G
        for layer in self.attention_layers:
            out = layer(out, nei_mask, input_r=input_r, sw=sw)
        return out

    def __getitem__(self, key: int) -> "NeighborGatedAttentionLayer":
        if isinstance(key, int):
            return self.attention_layers[key]
        else:
            raise TypeError(key)

    def __setitem__(
        self, key: int, value: Union["NeighborGatedAttentionLayer", dict]
    ) -> None:
        if not isinstance(key, int):
            raise TypeError(key)
        if isinstance(value, self.network_type):
            pass
        elif isinstance(value, dict):
            value = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        self.attention_layers[key] = value

    def serialize(self) -> dict:
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
        temperature: float | None = None,
        trainable_ln: bool = True,
        ln_eps: float = 1e-5,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
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
            trainable=trainable,
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
        x: Array,
        nei_mask: Array,
        input_r: Array | None = None,
        sw: Array | None = None,
    ) -> Array:
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
    def deserialize(cls, data: dict) -> "NeighborGatedAttentionLayer":
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
        temperature: float | None = None,
        bias: bool = True,
        smooth: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
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
            trainable=trainable,
        )
        self.out_proj = NativeLayer(
            hidden_dim,
            embed_dim,
            bias=bias,
            use_timestep=False,
            precision=precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )

    def call(
        self,
        query: Array,
        nei_mask: Array,
        input_r: Array | None = None,
        sw: Array | None = None,
        attnw_shift: float = 20.0,
    ) -> tuple[Array, Array]:
        xp = array_api_compat.array_namespace(query, nei_mask)
        # Linear projection
        # q, k, v = xp.split(self.in_proj(query), 3, axis=-1)
        _query = self.in_proj(query)
        q = _query[..., 0 : self.head_dim]
        k = _query[..., self.head_dim : self.head_dim * 2]
        v = _query[..., self.head_dim * 2 : self.head_dim * 3]
        # Reshape and normalize
        # (nf x nloc) x num_heads x nnei x head_dim
        q = xp.permute_dims(
            xp.reshape(q, (-1, self.nnei, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        k = xp.permute_dims(
            xp.reshape(k, (-1, self.nnei, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        v = xp.permute_dims(
            xp.reshape(v, (-1, self.nnei, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        if self.normalize:
            q = np_normalize(q, axis=-1)
            k = np_normalize(k, axis=-1)
            v = np_normalize(v, axis=-1)
        q = q * self.scaling
        # Attention weights
        # (nf x nloc) x num_heads x nnei x nnei
        attn_weights = q @ xp.permute_dims(k, (0, 1, 3, 2))
        nei_mask = xp.reshape(nei_mask, (-1, self.nnei))
        if self.smooth:
            sw = xp.reshape(sw, (-1, 1, self.nnei))
            attn_weights = (attn_weights + attnw_shift) * sw[:, :, :, None] * sw[
                :, :, None, :
            ] - attnw_shift
        else:
            attn_weights = xp.where(
                nei_mask[:, None, None, :],
                attn_weights,
                xp.full_like(attn_weights, -xp.inf),
            )
        attn_weights = np_softmax(attn_weights, axis=-1)
        attn_weights = xp.where(
            nei_mask[:, None, :, None], attn_weights, xp.zeros_like(attn_weights)
        )
        if self.smooth:
            attn_weights = attn_weights * sw[:, :, :, None] * sw[:, :, None, :]
        if self.dotr:
            angular_weight = xp.reshape(
                input_r @ xp.permute_dims(input_r, (0, 2, 1)),
                (-1, 1, self.nnei, self.nnei),
            )
            attn_weights = attn_weights * angular_weight
        # Output projection
        # (nf x nloc) x num_heads x nnei x head_dim
        o = attn_weights @ v
        # (nf x nloc) x nnei x (num_heads x head_dim)
        o = xp.reshape(
            xp.permute_dims(o, (0, 2, 1, 3)), (-1, self.nnei, self.hidden_dim)
        )
        output = self.out_proj(o)
        return output, attn_weights

    def serialize(self) -> dict:
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
    def deserialize(cls, data: dict) -> "GatedAttentionLayer":
        data = data.copy()
        in_proj = data.pop("in_proj")
        out_proj = data.pop("out_proj")
        obj = cls(**data)
        obj.in_proj = NativeLayer.deserialize(in_proj)
        obj.out_proj = NativeLayer.deserialize(out_proj)
        return obj
