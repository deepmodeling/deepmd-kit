# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from deepmd.dpmodel import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.utils import (
    EnvMat,
    PairExcludeMask,
)
from deepmd.dpmodel.utils.network import (
    LayerNorm,
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .descriptor import (
    DescriptorBlock,
)
from .dpa1 import (
    np_softmax,
)


@DescriptorBlock.register("se_repformer")
@DescriptorBlock.register("se_uni")
class DescrptBlockRepformers(NativeOP, DescriptorBlock):
    r"""
    The repformer descriptor block.

    Parameters
    ----------
    rcut : float
        The cut-off radius.
    rcut_smth : float
        Where to start smoothing. For example the 1/r term is smoothed from rcut to rcut_smth.
    sel : int
        Maximally possible number of selected neighbors.
    ntypes : int
        Number of element types
    nlayers : int, optional
        Number of repformer layers.
    g1_dim : int, optional
        Dimension of the first graph convolution layer.
    g2_dim : int, optional
        Dimension of the second graph convolution layer.
    axis_neuron : int, optional
        Size of the submatrix of G (embedding matrix).
    direct_dist : bool, optional
        Whether to use direct distance information (1/r term) in the repformer block.
    update_g1_has_conv : bool, optional
        Whether to update the g1 rep with convolution term.
    update_g1_has_drrd : bool, optional
        Whether to update the g1 rep with the drrd term.
    update_g1_has_grrg : bool, optional
        Whether to update the g1 rep with the grrg term.
    update_g1_has_attn : bool, optional
        Whether to update the g1 rep with the localized self-attention.
    update_g2_has_g1g1 : bool, optional
        Whether to update the g2 rep with the g1xg1 term.
    update_g2_has_attn : bool, optional
        Whether to update the g2 rep with the gated self-attention.
    update_h2 : bool, optional
        Whether to update the h2 rep.
    attn1_hidden : int, optional
        The hidden dimension of localized self-attention to update the g1 rep.
    attn1_nhead : int, optional
        The number of heads in localized self-attention to update the g1 rep.
    attn2_hidden : int, optional
        The hidden dimension of gated self-attention to update the g2 rep.
    attn2_nhead : int, optional
        The number of heads in gated self-attention to update the g2 rep.
    attn2_has_gate : bool, optional
        Whether to use gate in the gated self-attention to update the g2 rep.
    activation_function : str, optional
        The activation function in the embedding net.
    update_style : str, optional
        Style to update a representation.
        Supported options are:
        -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
        -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
        -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
        where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
        and `update_residual_init`.
    update_residual : float, optional
        When update using residual mode, the initial std of residual vector weights.
    update_residual_init : str, optional
        When update using residual mode, the initialization mode of residual vector weights.
    set_davg_zero : bool, optional
        Set the normalization average to zero.
    precision : str, optional
        The precision of the embedding net parameters.
    smooth : bool, optional
        Whether to use smoothness in processes such as attention weights calculation.
    exclude_types : List[List[int]], optional
        The excluded pairs of types which have no interaction with each other.
        For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection : float, optional
        Protection parameter to prevent division by zero errors during environment matrix calculations.
        For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
    trainable_ln : bool, optional
        Whether to use trainable shift and scale weights in layer normalization.
    ln_eps : float, optional
        The epsilon value for layer normalization.
    seed : int, optional
        The random seed for initialization.
    """

    def __init__(
        self,
        rcut,
        rcut_smth,
        sel: int,
        ntypes: int,
        nlayers: int = 3,
        g1_dim=128,
        g2_dim=16,
        axis_neuron: int = 4,
        direct_dist: bool = False,
        update_g1_has_conv: bool = True,
        update_g1_has_drrd: bool = True,
        update_g1_has_grrg: bool = True,
        update_g1_has_attn: bool = True,
        update_g2_has_g1g1: bool = True,
        update_g2_has_attn: bool = True,
        update_h2: bool = False,
        attn1_hidden: int = 64,
        attn1_nhead: int = 4,
        attn2_hidden: int = 16,
        attn2_nhead: int = 4,
        attn2_has_gate: bool = False,
        activation_function: str = "tanh",
        update_style: str = "res_avg",
        update_residual: float = 0.001,
        update_residual_init: str = "norm",
        set_davg_zero: bool = True,
        smooth: bool = True,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.ntypes = ntypes
        self.nlayers = nlayers
        sel = [sel] if isinstance(sel, int) else sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4  # use full descriptor.
        assert len(sel) == 1
        self.sel = sel
        self.sec = self.sel
        self.split_sel = self.sel
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.g1_dim = g1_dim
        self.g2_dim = g2_dim
        self.update_g1_has_conv = update_g1_has_conv
        self.update_g1_has_drrd = update_g1_has_drrd
        self.update_g1_has_grrg = update_g1_has_grrg
        self.update_g1_has_attn = update_g1_has_attn
        self.update_g2_has_g1g1 = update_g2_has_g1g1
        self.update_g2_has_attn = update_g2_has_attn
        self.update_h2 = update_h2
        self.attn1_hidden = attn1_hidden
        self.attn1_nhead = attn1_nhead
        self.attn2_has_gate = attn2_has_gate
        self.attn2_hidden = attn2_hidden
        self.attn2_nhead = attn2_nhead
        self.activation_function = activation_function
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.direct_dist = direct_dist
        self.act = get_activation_fn(self.activation_function)
        self.smooth = smooth
        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.epsilon = 1e-4

        self.g2_embd = NativeLayer(
            1, self.g2_dim, precision=precision, seed=child_seed(seed, 0)
        )
        layers = []
        for ii in range(nlayers):
            layers.append(
                RepformerLayer(
                    self.rcut,
                    self.rcut_smth,
                    self.sel,
                    self.ntypes,
                    self.g1_dim,
                    self.g2_dim,
                    axis_neuron=self.axis_neuron,
                    update_chnnl_2=(ii != nlayers - 1),
                    update_g1_has_conv=self.update_g1_has_conv,
                    update_g1_has_drrd=self.update_g1_has_drrd,
                    update_g1_has_grrg=self.update_g1_has_grrg,
                    update_g1_has_attn=self.update_g1_has_attn,
                    update_g2_has_g1g1=self.update_g2_has_g1g1,
                    update_g2_has_attn=self.update_g2_has_attn,
                    update_h2=self.update_h2,
                    attn1_hidden=self.attn1_hidden,
                    attn1_nhead=self.attn1_nhead,
                    attn2_has_gate=self.attn2_has_gate,
                    attn2_hidden=self.attn2_hidden,
                    attn2_nhead=self.attn2_nhead,
                    activation_function=self.activation_function,
                    update_style=self.update_style,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    smooth=self.smooth,
                    trainable_ln=self.trainable_ln,
                    ln_eps=self.ln_eps,
                    precision=precision,
                    seed=child_seed(child_seed(seed, 1), ii),
                )
            )
        self.layers = layers

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.env_mat = EnvMat(self.rcut, self.rcut_smth, protection=self.env_protection)
        self.mean = np.zeros(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.orig_sel = self.sel

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.rcut

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
        """Returns the embedding dimension g2."""
        return self.g2_dim

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

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.g1_dim

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.g1_dim

    @property
    def dim_emb(self):
        """Returns the embedding dimension g2."""
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

    def call(
        self,
        nlist: np.ndarray,
        coord_ext: np.ndarray,
        atype_ext: np.ndarray,
        atype_embd_ext: Optional[np.ndarray] = None,
        mapping: Optional[np.ndarray] = None,
    ):
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        nlist = np.where(exclude_mask, nlist, -1)
        # nf x nloc x nnei x 4
        dmatrix, diff, sw = self.env_mat.call(
            coord_ext, atype_ext, nlist, self.mean, self.stddev
        )
        nf, nloc, nnei, _ = dmatrix.shape
        # nf x nloc x nnei
        nlist_mask = nlist != -1
        # nf x nloc x nnei
        sw = sw.reshape(nf, nloc, nnei)
        sw = np.where(nlist_mask, sw, 0.0)
        # nf x nloc x tebd_dim
        atype_embd = atype_embd_ext[:, :nloc, :]
        assert list(atype_embd.shape) == [nf, nloc, self.g1_dim]

        g1 = self.act(atype_embd)
        # nf x nloc x nnei x 1,  nf x nloc x nnei x 3
        if not self.direct_dist:
            g2, h2 = np.split(dmatrix, [1], axis=-1)
        else:
            g2, h2 = np.linalg.norm(diff, axis=-1, keepdims=True), diff
            g2 = g2 / self.rcut
            h2 = h2 / self.rcut
        # nf x nloc x nnei x ng2
        g2 = self.act(self.g2_embd(g2))
        # set all padding positions to index of 0
        # if a neighbor is real or not is indicated by nlist_mask
        nlist[nlist == -1] = 0
        # nf x nall x ng1
        mapping = np.tile(mapping.reshape(nf, -1, 1), (1, 1, self.g1_dim))
        for idx, ll in enumerate(self.layers):
            # g1:     nf x nloc x ng1
            # g1_ext: nf x nall x ng1
            g1_ext = np.take_along_axis(g1, mapping, axis=1)
            g1, g2, h2 = ll.call(
                g1_ext,
                g2,
                h2,
                nlist,
                nlist_mask,
                sw,
            )

        # nf x nloc x 3 x ng2
        h2g2 = _cal_hg(g2, h2, nlist_mask, sw, smooth=self.smooth, epsilon=self.epsilon)
        # (nf x nloc) x ng2 x 3
        rot_mat = np.transpose(h2g2, (0, 1, 3, 2))
        return g1, g2, h2, rot_mat.reshape(-1, nloc, self.dim_emb, 3), sw

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return True


# translated by GPT and modified
def get_residual(
    _dim: int,
    _scale: float,
    _mode: str = "norm",
    trainable: bool = True,
    precision: str = "float64",
    seed: Optional[Union[int, List[int]]] = None,
) -> np.ndarray:
    """
    Get residual tensor for one update vector.

    Parameters
    ----------
    _dim : int
        The dimension of the update vector.
    _scale
        The initial scale of the residual tensor. See `_mode` for details.
    _mode
        The mode of residual initialization for the residual tensor.
        - "norm" (default): init residual using normal with `_scale` std.
        - "const": init residual using element-wise constants of `_scale`.
    trainable
        Whether the residual tensor is trainable.
    precision
        The precision of the residual tensor.
    """
    residual = np.zeros(_dim, dtype=PRECISION_DICT[precision])
    rng = np.random.default_rng(seed=seed)
    if trainable:
        if _mode == "norm":
            residual = rng.normal(scale=_scale, size=_dim).astype(
                PRECISION_DICT[precision]
            )
        elif _mode == "const":
            residual.fill(_scale)
        else:
            raise RuntimeError(f"Unsupported initialization mode '{_mode}'!")
    return residual


def _make_nei_g1(
    g1_ext: np.ndarray,
    nlist: np.ndarray,
) -> np.ndarray:
    """
    Make neighbor-wise atomic invariant rep.

    Parameters
    ----------
    g1_ext
        Extended atomic invariant rep, with shape [nf, nall, ng1].
    nlist
        Neighbor list, with shape [nf, nloc, nnei].

    Returns
    -------
    gg1: np.ndarray
        Neighbor-wise atomic invariant rep, with shape [nf, nloc, nnei, ng1].
    """
    # nlist: nf x nloc x nnei
    nf, nloc, nnei = nlist.shape
    # g1_ext: nf x nall x ng1
    ng1 = g1_ext.shape[-1]
    # index: nf x (nloc x nnei) x ng1
    index = np.tile(nlist.reshape(nf, nloc * nnei, 1), (1, 1, ng1))
    # gg1  : nf x (nloc x nnei) x ng1
    gg1 = np.take_along_axis(g1_ext, index, axis=1)
    # gg1  : nf x nloc x nnei x ng1
    gg1 = gg1.reshape(nf, nloc, nnei, ng1)
    return gg1


def _apply_nlist_mask(
    gg: np.ndarray,
    nlist_mask: np.ndarray,
) -> np.ndarray:
    """
    Apply nlist mask to neighbor-wise rep tensors.

    Parameters
    ----------
    gg
        Neighbor-wise rep tensors, with shape [nf, nloc, nnei, d].
    nlist_mask
        Neighbor list mask, where zero means no neighbor, with shape [nf, nloc, nnei].
    """
    masked_gg = np.where(nlist_mask[:, :, :, None], gg, 0.0)
    return masked_gg


def _apply_switch(gg: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """
    Apply switch function to neighbor-wise rep tensors.

    Parameters
    ----------
    gg
        Neighbor-wise rep tensors, with shape [nf, nloc, nnei, d].
    sw
        The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
        and remains 0 beyond rcut, with shape [nf, nloc, nnei].
    """
    # gg: nf x nloc x nnei x d
    # sw: nf x nloc x nnei
    return gg * sw[:, :, :, None]


def _cal_hg(
    g: np.ndarray,
    h: np.ndarray,
    nlist_mask: np.ndarray,
    sw: np.ndarray,
    smooth: bool = True,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Calculate the transposed rotation matrix.

    Parameters
    ----------
    g
        Neighbor-wise/Pair-wise invariant rep tensors, with shape [nf, nloc, nnei, ng].
    h
        Neighbor-wise/Pair-wise equivariant rep tensors, with shape [nf, nloc, nnei, 3].
    nlist_mask
        Neighbor list mask, where zero means no neighbor, with shape [nf, nloc, nnei].
    sw
        The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
        and remains 0 beyond rcut, with shape [nf, nloc, nnei].
    smooth
        Whether to use smoothness in processes such as attention weights calculation.
    epsilon
        Protection of 1./nnei.

    Returns
    -------
    hg
        The transposed rotation matrix, with shape [nf, nloc, 3, ng].
    """
    # g: nf x nloc x nnei x ng
    # h: nf x nloc x nnei x 3
    # msk: nf x nloc x nnei
    nf, nloc, nnei, _ = g.shape
    ng = g.shape[-1]
    # nf x nloc x nnei x ng
    g = _apply_nlist_mask(g, nlist_mask)
    if not smooth:
        # nf x nloc
        invnnei = 1.0 / (epsilon + np.sum(nlist_mask, axis=-1))
        # nf x nloc x 1 x 1
        invnnei = invnnei[:, :, np.newaxis, np.newaxis]
    else:
        g = _apply_switch(g, sw)
        invnnei = (1.0 / float(nnei)) * np.ones((nf, nloc, 1, 1), dtype=g.dtype)
    # nf x nloc x 3 x ng
    hg = np.matmul(np.transpose(h, axes=(0, 1, 3, 2)), g) * invnnei
    return hg


def _cal_grrg(hg: np.ndarray, axis_neuron: int) -> np.ndarray:
    """
    Calculate the atomic invariant rep.

    Parameters
    ----------
    hg
        The transposed rotation matrix, with shape [nf, nloc, 3, ng].
    axis_neuron
        Size of the submatrix.

    Returns
    -------
    grrg
        Atomic invariant rep, with shape [nf, nloc, (axis_neuron * ng)].
    """
    # nf x nloc x 3 x ng
    nf, nloc, _, ng = hg.shape
    # nf x nloc x 3 x axis
    hgm = np.split(hg, [axis_neuron], axis=-1)[0]
    # nf x nloc x axis_neuron x ng
    grrg = np.matmul(np.transpose(hgm, axes=(0, 1, 3, 2)), hg) / (3.0**1)
    # nf x nloc x (axis_neuron * ng)
    grrg = grrg.reshape(nf, nloc, axis_neuron * ng)
    return grrg


def symmetrization_op(
    g: np.ndarray,
    h: np.ndarray,
    nlist_mask: np.ndarray,
    sw: np.ndarray,
    axis_neuron: int,
    smooth: bool = True,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Symmetrization operator to obtain atomic invariant rep.

    Parameters
    ----------
    g
        Neighbor-wise/Pair-wise invariant rep tensors, with shape [nf, nloc, nnei, ng].
    h
        Neighbor-wise/Pair-wise equivariant rep tensors, with shape [nf, nloc, nnei, 3].
    nlist_mask
        Neighbor list mask, where zero means no neighbor, with shape [nf, nloc, nnei].
    sw
        The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
        and remains 0 beyond rcut, with shape [nf, nloc, nnei].
    axis_neuron
        Size of the submatrix.
    smooth
        Whether to use smoothness in processes such as attention weights calculation.
    epsilon
        Protection of 1./nnei.

    Returns
    -------
    grrg
        Atomic invariant rep, with shape [nf, nloc, (axis_neuron * ng)].
    """
    # g: nf x nloc x nnei x ng
    # h: nf x nloc x nnei x 3
    # msk: nf x nloc x nnei
    nf, nloc, nnei, _ = g.shape
    # nf x nloc x 3 x ng
    hg = _cal_hg(g, h, nlist_mask, sw, smooth=smooth, epsilon=epsilon)
    # nf x nloc x (axis_neuron x ng)
    grrg = _cal_grrg(hg, axis_neuron)
    return grrg


class Atten2Map(NativeOP):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_num: int,
        has_gate: bool = False,  # apply gate to attn map
        smooth: bool = True,
        attnw_shift: float = 20.0,
        precision: str = "float64",
        seed: Optional[Union[int, List[int]]] = None,
    ):
        """Return neighbor-wise multi-head self-attention maps, with gate mechanism."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.mapqk = NativeLayer(
            input_dim,
            hidden_dim * 2 * head_num,
            bias=False,
            precision=precision,
            seed=seed,
        )
        self.has_gate = has_gate
        self.smooth = smooth
        self.attnw_shift = attnw_shift
        self.precision = precision

    def call(
        self,
        g2: np.ndarray,  # nf x nloc x nnei x ng2
        h2: np.ndarray,  # nf x nloc x nnei x 3
        nlist_mask: np.ndarray,  # nf x nloc x nnei
        sw: np.ndarray,  # nf x nloc x nnei
    ) -> np.ndarray:
        (
            nf,
            nloc,
            nnei,
            _,
        ) = g2.shape
        nd, nh = self.hidden_dim, self.head_num
        # nf x nloc x nnei x nd x (nh x 2)
        g2qk = self.mapqk(g2).reshape(nf, nloc, nnei, nd, nh * 2)
        # nf x nloc x (nh x 2) x nnei x nd
        g2qk = np.transpose(g2qk, (0, 1, 4, 2, 3))
        # nf x nloc x nh x nnei x nd
        g2q, g2k = np.split(g2qk, [nh], axis=2)
        # g2q = np.linalg.norm(g2q, axis=-1)
        # g2k = np.linalg.norm(g2k, axis=-1)
        # nf x nloc x nh x nnei x nnei
        attnw = np.matmul(g2q, np.transpose(g2k, axes=(0, 1, 2, 4, 3))) / nd**0.5
        if self.has_gate:
            gate = np.matmul(h2, np.transpose(h2, axes=(0, 1, 3, 2))).reshape(
                nf, nloc, 1, nnei, nnei
            )
            attnw = attnw * gate
        # mask the attenmap, nf x nloc x 1 x 1 x nnei
        attnw_mask = ~np.expand_dims(np.expand_dims(nlist_mask, axis=2), axis=2)
        # mask the attenmap, nf x nloc x 1 x nnei x 1
        attnw_mask_c = ~np.expand_dims(np.expand_dims(nlist_mask, axis=2), axis=-1)
        if self.smooth:
            attnw = (attnw + self.attnw_shift) * sw[:, :, None, :, None] * sw[
                :, :, None, None, :
            ] - self.attnw_shift
        else:
            attnw = np.where(attnw_mask, -np.inf, attnw)
        attnw = np_softmax(attnw, axis=-1)
        attnw = np.where(attnw_mask, 0.0, attnw)
        # nf x nloc x nh x nnei x nnei
        attnw = np.where(attnw_mask_c, 0.0, attnw)
        if self.smooth:
            attnw = attnw * sw[:, :, None, :, None] * sw[:, :, None, None, :]
        # nf x nloc x nnei x nnei
        h2h2t = np.matmul(h2, np.transpose(h2, axes=(0, 1, 3, 2))) / 3.0**0.5
        # nf x nloc x nh x nnei x nnei
        ret = attnw * h2h2t[:, :, None, :, :]
        # ret = np.exp(g2qk - np.max(g2qk, axis=-1, keepdims=True))
        # nf x nloc x nnei x nnei x nh
        ret = np.transpose(ret, (0, 1, 3, 4, 2))
        return ret

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2Map",
            "@version": 1,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "head_num": self.head_num,
            "has_gate": self.has_gate,
            "smooth": self.smooth,
            "attnw_shift": self.attnw_shift,
            "precision": self.precision,
            "mapqk": self.mapqk.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2Map":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapqk = data.pop("mapqk")
        obj = cls(**data)
        obj.mapqk = NativeLayer.deserialize(mapqk)
        return obj


class Atten2MultiHeadApply(NativeOP):
    def __init__(
        self,
        input_dim: int,
        head_num: int,
        precision: str = "float64",
        seed: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.mapv = NativeLayer(
            input_dim,
            input_dim * head_num,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.head_map = NativeLayer(
            input_dim * head_num,
            input_dim,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.precision = precision

    def call(
        self,
        AA: np.ndarray,  # nf x nloc x nnei x nnei x nh
        g2: np.ndarray,  # nf x nloc x nnei x ng2
    ) -> np.ndarray:
        nf, nloc, nnei, ng2 = g2.shape
        nh = self.head_num
        # nf x nloc x nnei x ng2 x nh
        g2v = self.mapv(g2).reshape(nf, nloc, nnei, ng2, nh)
        # nf x nloc x nh x nnei x ng2
        g2v = np.transpose(g2v, (0, 1, 4, 2, 3))
        # g2v = np.linalg.norm(g2v, axis=-1)
        # nf x nloc x nh x nnei x nnei
        AA = np.transpose(AA, (0, 1, 4, 2, 3))
        # nf x nloc x nh x nnei x ng2
        ret = np.matmul(AA, g2v)
        # nf x nloc x nnei x ng2 x nh
        ret = np.transpose(ret, (0, 1, 3, 4, 2)).reshape(nf, nloc, nnei, (ng2 * nh))
        # nf x nloc x nnei x ng2
        return self.head_map(ret)

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2MultiHeadApply",
            "@version": 1,
            "input_dim": self.input_dim,
            "head_num": self.head_num,
            "precision": self.precision,
            "mapv": self.mapv.serialize(),
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2MultiHeadApply":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapv = data.pop("mapv")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.mapv = NativeLayer.deserialize(mapv)
        obj.head_map = NativeLayer.deserialize(head_map)
        return obj


class Atten2EquiVarApply(NativeOP):
    def __init__(
        self,
        input_dim: int,
        head_num: int,
        precision: str = "float64",
        seed: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.head_map = NativeLayer(
            head_num, 1, bias=False, precision=precision, seed=seed
        )
        self.precision = precision

    def call(
        self,
        AA: np.ndarray,  # nf x nloc x nnei x nnei x nh
        h2: np.ndarray,  # nf x nloc x nnei x 3
    ) -> np.ndarray:
        nf, nloc, nnei, _ = h2.shape
        nh = self.head_num
        # nf x nloc x nh x nnei x nnei
        AA = np.transpose(AA, (0, 1, 4, 2, 3))
        h2m = np.expand_dims(h2, axis=2)
        # nf x nloc x nh x nnei x 3
        h2m = np.tile(h2m, (1, 1, nh, 1, 1))
        # nf x nloc x nh x nnei x 3
        ret = np.matmul(AA, h2m)
        # nf x nloc x nnei x 3 x nh
        ret = np.transpose(ret, (0, 1, 3, 4, 2)).reshape(nf, nloc, nnei, 3, nh)
        # nf x nloc x nnei x 3
        return np.squeeze(self.head_map(ret), axis=-1)

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "Atten2EquiVarApply",
            "@version": 1,
            "input_dim": self.input_dim,
            "head_num": self.head_num,
            "precision": self.precision,
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Atten2EquiVarApply":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.head_map = NativeLayer.deserialize(head_map)
        return obj


class LocalAtten(NativeOP):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        head_num: int,
        smooth: bool = True,
        attnw_shift: float = 20.0,
        precision: str = "float64",
        seed: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.mapq = NativeLayer(
            input_dim,
            hidden_dim * 1 * head_num,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.mapkv = NativeLayer(
            input_dim,
            (hidden_dim + input_dim) * head_num,
            bias=False,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.head_map = NativeLayer(
            input_dim * head_num,
            input_dim,
            precision=precision,
            seed=child_seed(seed, 2),
        )
        self.smooth = smooth
        self.attnw_shift = attnw_shift
        self.precision = precision

    def call(
        self,
        g1: np.ndarray,  # nf x nloc x ng1
        gg1: np.ndarray,  # nf x nloc x nnei x ng1
        nlist_mask: np.ndarray,  # nf x nloc x nnei
        sw: np.ndarray,  # nf x nloc x nnei
    ) -> np.ndarray:
        nf, nloc, nnei = nlist_mask.shape
        ni, nd, nh = self.input_dim, self.hidden_dim, self.head_num
        assert ni == g1.shape[-1]
        assert ni == gg1.shape[-1]
        # nf x nloc x nd x nh
        g1q = self.mapq(g1).reshape(nf, nloc, nd, nh)
        # nf x nloc x nh x nd
        g1q = np.transpose(g1q, (0, 1, 3, 2))
        # nf x nloc x nnei x (nd+ni) x nh
        gg1kv = self.mapkv(gg1).reshape(nf, nloc, nnei, nd + ni, nh)
        gg1kv = np.transpose(gg1kv, (0, 1, 4, 2, 3))
        # nf x nloc x nh x nnei x nd, nf x nloc x nh x nnei x ng1
        gg1k, gg1v = np.split(gg1kv, [nd], axis=-1)

        # nf x nloc x nh x 1 x nnei
        attnw = (
            np.matmul(
                np.expand_dims(g1q, axis=-2), np.transpose(gg1k, axes=(0, 1, 2, 4, 3))
            )
            / nd**0.5
        )
        # nf x nloc x nh x nnei
        attnw = np.squeeze(attnw, axis=-2)
        # mask the attenmap, nf x nloc x 1 x nnei
        attnw_mask = ~np.expand_dims(nlist_mask, axis=-2)
        # nf x nloc x nh x nnei
        if self.smooth:
            attnw = (attnw + self.attnw_shift) * np.expand_dims(
                sw, axis=-2
            ) - self.attnw_shift
        else:
            attnw = np.where(attnw_mask, -np.inf, attnw)
        attnw = np_softmax(attnw, axis=-1)
        attnw = np.where(attnw_mask, 0.0, attnw)
        if self.smooth:
            attnw = attnw * np.expand_dims(sw, axis=-2)

        # nf x nloc x nh x ng1
        ret = (
            np.matmul(np.expand_dims(attnw, axis=-2), gg1v)
            .squeeze(-2)
            .reshape(nf, nloc, nh * ni)
        )
        # nf x nloc x ng1
        ret = self.head_map(ret)
        return ret

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        return {
            "@class": "LocalAtten",
            "@version": 1,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "head_num": self.head_num,
            "smooth": self.smooth,
            "attnw_shift": self.attnw_shift,
            "precision": self.precision,
            "mapq": self.mapq.serialize(),
            "mapkv": self.mapkv.serialize(),
            "head_map": self.head_map.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "LocalAtten":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        mapq = data.pop("mapq")
        mapkv = data.pop("mapkv")
        head_map = data.pop("head_map")
        obj = cls(**data)
        obj.mapq = NativeLayer.deserialize(mapq)
        obj.mapkv = NativeLayer.deserialize(mapkv)
        obj.head_map = NativeLayer.deserialize(head_map)
        return obj


class RepformerLayer(NativeOP):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel: int,
        ntypes: int,
        g1_dim=128,
        g2_dim=16,
        axis_neuron: int = 4,
        update_chnnl_2: bool = True,
        update_g1_has_conv: bool = True,
        update_g1_has_drrd: bool = True,
        update_g1_has_grrg: bool = True,
        update_g1_has_attn: bool = True,
        update_g2_has_g1g1: bool = True,
        update_g2_has_attn: bool = True,
        update_h2: bool = False,
        attn1_hidden: int = 64,
        attn1_nhead: int = 4,
        attn2_hidden: int = 16,
        attn2_nhead: int = 4,
        attn2_has_gate: bool = False,
        activation_function: str = "tanh",
        update_style: str = "res_avg",
        update_residual: float = 0.001,
        update_residual_init: str = "norm",
        smooth: bool = True,
        precision: str = "float64",
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        seed: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.epsilon = 1e-4  # protection of 1./nnei
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.ntypes = ntypes
        sel = [sel] if isinstance(sel, int) else sel
        self.nnei = sum(sel)
        assert len(sel) == 1
        self.sel = sel
        self.sec = self.sel
        self.axis_neuron = axis_neuron
        self.activation_function = activation_function
        self.act = get_activation_fn(self.activation_function)
        self.update_g1_has_grrg = update_g1_has_grrg
        self.update_g1_has_drrd = update_g1_has_drrd
        self.update_g1_has_conv = update_g1_has_conv
        self.update_g1_has_attn = update_g1_has_attn
        self.update_chnnl_2 = update_chnnl_2
        self.update_g2_has_g1g1 = update_g2_has_g1g1 if self.update_chnnl_2 else False
        self.update_g2_has_attn = update_g2_has_attn if self.update_chnnl_2 else False
        self.update_h2 = update_h2 if self.update_chnnl_2 else False
        del update_g2_has_g1g1, update_g2_has_attn, update_h2
        self.attn1_hidden = attn1_hidden
        self.attn1_nhead = attn1_nhead
        self.attn2_hidden = attn2_hidden
        self.attn2_nhead = attn2_nhead
        self.attn2_has_gate = attn2_has_gate
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.smooth = smooth
        self.g1_dim = g1_dim
        self.g2_dim = g2_dim
        self.trainable_ln = trainable_ln
        self.ln_eps = ln_eps
        self.precision = precision

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.g1_residual = []
        self.g2_residual = []
        self.h2_residual = []

        if self.update_style == "res_residual":
            self.g1_residual.append(
                get_residual(
                    g1_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 0),
                )
            )

        g1_in_dim = self.cal_1_dim(g1_dim, g2_dim, self.axis_neuron)
        self.linear1 = NativeLayer(
            g1_in_dim,
            g1_dim,
            precision=precision,
            seed=child_seed(seed, 1),
        )
        self.linear2 = None
        self.proj_g1g2 = None
        self.proj_g1g1g2 = None
        self.attn2g_map = None
        self.attn2_mh_apply = None
        self.attn2_lm = None
        self.attn2_ev_apply = None
        self.loc_attn = None

        if self.update_chnnl_2:
            self.linear2 = NativeLayer(
                g2_dim,
                g2_dim,
                precision=precision,
                seed=child_seed(seed, 2),
            )
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        g2_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 3),
                    )
                )
        if self.update_g1_has_conv:
            self.proj_g1g2 = NativeLayer(
                g1_dim,
                g2_dim,
                bias=False,
                precision=precision,
                seed=child_seed(seed, 4),
            )
        if self.update_g2_has_g1g1:
            self.proj_g1g1g2 = NativeLayer(
                g1_dim,
                g2_dim,
                bias=False,
                precision=precision,
                seed=child_seed(seed, 5),
            )
            if self.update_style == "res_residual":
                self.g2_residual.append(
                    get_residual(
                        g2_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 6),
                    )
                )
        if self.update_g2_has_attn or self.update_h2:
            self.attn2g_map = Atten2Map(
                g2_dim,
                attn2_hidden,
                attn2_nhead,
                attn2_has_gate,
                self.smooth,
                precision=precision,
                seed=child_seed(seed, 7),
            )
            if self.update_g2_has_attn:
                self.attn2_mh_apply = Atten2MultiHeadApply(
                    g2_dim, attn2_nhead, precision=precision, seed=child_seed(seed, 8)
                )
                self.attn2_lm = LayerNorm(
                    g2_dim,
                    eps=ln_eps,
                    trainable=trainable_ln,
                    precision=precision,
                    seed=child_seed(seed, 9),
                )
                if self.update_style == "res_residual":
                    self.g2_residual.append(
                        get_residual(
                            g2_dim,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 10),
                        )
                    )

            if self.update_h2:
                self.attn2_ev_apply = Atten2EquiVarApply(
                    g2_dim, attn2_nhead, precision=precision, seed=child_seed(seed, 11)
                )
                if self.update_style == "res_residual":
                    self.h2_residual.append(
                        get_residual(
                            1,
                            self.update_residual,
                            self.update_residual_init,
                            precision=precision,
                            seed=child_seed(seed, 12),
                        )
                    )
        if self.update_g1_has_attn:
            self.loc_attn = LocalAtten(
                g1_dim,
                attn1_hidden,
                attn1_nhead,
                self.smooth,
                precision=precision,
                seed=child_seed(seed, 13),
            )
            if self.update_style == "res_residual":
                self.g1_residual.append(
                    get_residual(
                        g1_dim,
                        self.update_residual,
                        self.update_residual_init,
                        precision=precision,
                        seed=child_seed(seed, 14),
                    )
                )

    def cal_1_dim(self, g1d: int, g2d: int, ax: int) -> int:
        ret = g1d
        if self.update_g1_has_grrg:
            ret += g2d * ax
        if self.update_g1_has_drrd:
            ret += g1d * ax
        if self.update_g1_has_conv:
            ret += g2d
        return ret

    def _update_h2(
        self,
        h2: np.ndarray,
        attn: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the attention weights update for pair-wise equivariant rep.

        Parameters
        ----------
        h2
            Pair-wise equivariant rep tensors, with shape nf x nloc x nnei x 3.
        attn
            Attention weights from g2 attention, with shape nf x nloc x nnei x nnei x nh2.
        """
        assert self.attn2_ev_apply is not None
        # nf x nloc x nnei x nh2
        h2_1 = self.attn2_ev_apply(attn, h2)
        return h2_1

    def _update_g1_conv(
        self,
        gg1: np.ndarray,
        g2: np.ndarray,
        nlist_mask: np.ndarray,
        sw: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the convolution update for atomic invariant rep.

        Parameters
        ----------
        gg1
            Neighbor-wise atomic invariant rep, with shape nf x nloc x nnei x ng1.
        g2
            Pair invariant rep, with shape nf x nloc x nnei x ng2.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nf x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nf x nloc x nnei.
        """
        assert self.proj_g1g2 is not None
        nf, nloc, nnei, _ = g2.shape
        ng1 = gg1.shape[-1]
        ng2 = g2.shape[-1]
        # gg1  : nf x nloc x nnei x ng2
        gg1 = self.proj_g1g2(gg1).reshape(nf, nloc, nnei, ng2)
        # nf x nloc x nnei x ng2
        gg1 = _apply_nlist_mask(gg1, nlist_mask)
        if not self.smooth:
            # normalized by number of neighbors, not smooth
            # nf x nloc
            invnnei = 1.0 / (self.epsilon + np.sum(nlist_mask, axis=-1))
            # nf x nloc x 1
            invnnei = invnnei[:, :, np.newaxis]
        else:
            gg1 = _apply_switch(gg1, sw)
            invnnei = (1.0 / float(nnei)) * np.ones((nf, nloc, 1), dtype=gg1.dtype)
        # nf x nloc x ng2
        g1_11 = np.sum(g2 * gg1, axis=2) * invnnei
        return g1_11

    def _update_g2_g1g1(
        self,
        g1: np.ndarray,  # nf x nloc x ng1
        gg1: np.ndarray,  # nf x nloc x nnei x ng1
        nlist_mask: np.ndarray,  # nf x nloc x nnei
        sw: np.ndarray,  # nf x nloc x nnei
    ) -> np.ndarray:
        """
        Update the g2 using element-wise dot g1_i * g1_j.

        Parameters
        ----------
        g1
            Atomic invariant rep, with shape nf x nloc x ng1.
        gg1
            Neighbor-wise atomic invariant rep, with shape nf x nloc x nnei x ng1.
        nlist_mask
            Neighbor list mask, where zero means no neighbor, with shape nf x nloc x nnei.
        sw
            The switch function, which equals 1 within the rcut_smth range, smoothly decays from 1 to 0 between rcut_smth and rcut,
            and remains 0 beyond rcut, with shape nf x nloc x nnei.
        """
        ret = np.expand_dims(g1, axis=-2) * gg1
        # nf x nloc x nnei x ng1
        ret = _apply_nlist_mask(ret, nlist_mask)
        if self.smooth:
            ret = _apply_switch(ret, sw)
        return ret

    def call(
        self,
        g1_ext: np.ndarray,  # nf x nall x ng1
        g2: np.ndarray,  # nf x nloc x nnei x ng2
        h2: np.ndarray,  # nf x nloc x nnei x 3
        nlist: np.ndarray,  # nf x nloc x nnei
        nlist_mask: np.ndarray,  # nf x nloc x nnei
        sw: np.ndarray,  # switch func, nf x nloc x nnei
    ):
        """
        Parameters
        ----------
        g1_ext : nf x nall x ng1         extended single-atom chanel
        g2 : nf x nloc x nnei x ng2  pair-atom channel, invariant
        h2 : nf x nloc x nnei x 3    pair-atom channel, equivariant
        nlist : nf x nloc x nnei        neighbor list (padded neis are set to 0)
        nlist_mask : nf x nloc x nnei   masks of the neighbor list. real nei 1 otherwise 0
        sw : nf x nloc x nnei        switch function

        Returns
        -------
        g1:     nf x nloc x ng1         updated single-atom chanel
        g2:     nf x nloc x nnei x ng2  updated pair-atom channel, invariant
        h2:     nf x nloc x nnei x 3    updated pair-atom channel, equivariant
        """
        cal_gg1 = (
            self.update_g1_has_drrd
            or self.update_g1_has_conv
            or self.update_g1_has_attn
            or self.update_g2_has_g1g1
        )

        nf, nloc, nnei, _ = g2.shape
        nall = g1_ext.shape[1]
        g1, _ = np.split(g1_ext, [nloc], axis=1)
        assert (nf, nloc) == g1.shape[:2]
        assert (nf, nloc, nnei) == h2.shape[:3]

        g2_update: List[np.ndarray] = [g2]
        h2_update: List[np.ndarray] = [h2]
        g1_update: List[np.ndarray] = [g1]
        g1_mlp: List[np.ndarray] = [g1]

        if cal_gg1:
            gg1 = _make_nei_g1(g1_ext, nlist)
        else:
            gg1 = None

        if self.update_chnnl_2:
            # mlp(g2)
            assert self.linear2 is not None
            # nf x nloc x nnei x ng2
            g2_1 = self.act(self.linear2(g2))
            g2_update.append(g2_1)

            if self.update_g2_has_g1g1:
                # linear(g1_i * g1_j)
                assert gg1 is not None
                assert self.proj_g1g1g2 is not None
                g2_update.append(
                    self.proj_g1g1g2(self._update_g2_g1g1(g1, gg1, nlist_mask, sw))
                )

            if self.update_g2_has_attn or self.update_h2:
                # gated_attention(g2, h2)
                assert self.attn2g_map is not None
                # nf x nloc x nnei x nnei x nh
                AAg = self.attn2g_map(g2, h2, nlist_mask, sw)

                if self.update_g2_has_attn:
                    assert self.attn2_mh_apply is not None
                    assert self.attn2_lm is not None
                    # nf x nloc x nnei x ng2
                    g2_2 = self.attn2_mh_apply(AAg, g2)
                    g2_2 = self.attn2_lm(g2_2)
                    g2_update.append(g2_2)

                if self.update_h2:
                    # linear_head(attention_weights * h2)
                    h2_update.append(self._update_h2(h2, AAg))

        if self.update_g1_has_conv:
            assert gg1 is not None
            g1_mlp.append(self._update_g1_conv(gg1, g2, nlist_mask, sw))

        if self.update_g1_has_grrg:
            g1_mlp.append(
                symmetrization_op(
                    g2,
                    h2,
                    nlist_mask,
                    sw,
                    self.axis_neuron,
                    smooth=self.smooth,
                    epsilon=self.epsilon,
                )
            )

        if self.update_g1_has_drrd:
            assert gg1 is not None
            g1_mlp.append(
                symmetrization_op(
                    gg1,
                    h2,
                    nlist_mask,
                    sw,
                    self.axis_neuron,
                    smooth=self.smooth,
                    epsilon=self.epsilon,
                )
            )

            # nf x nloc x [ng1+ng2+(axisxng2)+(axisxng1)]
            #                  conv   grrg      drrd
        g1_1 = self.act(self.linear1(np.concatenate(g1_mlp, axis=-1)))
        g1_update.append(g1_1)

        if self.update_g1_has_attn:
            assert gg1 is not None
            assert self.loc_attn is not None
            g1_update.append(self.loc_attn(g1, gg1, nlist_mask, sw))

        # update
        if self.update_chnnl_2:
            g2_new = self.list_update(g2_update, "g2")
            h2_new = self.list_update(h2_update, "h2")
        else:
            g2_new, h2_new = g2, h2
        g1_new = self.list_update(g1_update, "g1")
        return g1_new, g2_new, h2_new

    def list_update_res_avg(
        self,
        update_list: List[np.ndarray],
    ) -> np.ndarray:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    def list_update_res_incr(self, update_list: List[np.ndarray]) -> np.ndarray:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    def list_update_res_residual(
        self, update_list: List[np.ndarray], update_name: str = "g1"
    ) -> np.ndarray:
        nitem = len(update_list)
        uu = update_list[0]
        if update_name == "g1":
            for ii, vv in enumerate(self.g1_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "g2":
            for ii, vv in enumerate(self.g2_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "h2":
            for ii, vv in enumerate(self.h2_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    def list_update(
        self, update_list: List[np.ndarray], update_name: str = "g1"
    ) -> np.ndarray:
        if self.update_style == "res_avg":
            return self.list_update_res_avg(update_list)
        elif self.update_style == "res_incr":
            return self.list_update_res_incr(update_list)
        elif self.update_style == "res_residual":
            return self.list_update_res_residual(update_list, update_name=update_name)
        else:
            raise RuntimeError(f"unknown update style {self.update_style}")

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        data = {
            "@class": "RepformerLayer",
            "@version": 1,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
            "ntypes": self.ntypes,
            "g1_dim": self.g1_dim,
            "g2_dim": self.g2_dim,
            "axis_neuron": self.axis_neuron,
            "update_chnnl_2": self.update_chnnl_2,
            "update_g1_has_conv": self.update_g1_has_conv,
            "update_g1_has_drrd": self.update_g1_has_drrd,
            "update_g1_has_grrg": self.update_g1_has_grrg,
            "update_g1_has_attn": self.update_g1_has_attn,
            "update_g2_has_g1g1": self.update_g2_has_g1g1,
            "update_g2_has_attn": self.update_g2_has_attn,
            "update_h2": self.update_h2,
            "attn1_hidden": self.attn1_hidden,
            "attn1_nhead": self.attn1_nhead,
            "attn2_hidden": self.attn2_hidden,
            "attn2_nhead": self.attn2_nhead,
            "attn2_has_gate": self.attn2_has_gate,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "smooth": self.smooth,
            "precision": self.precision,
            "trainable_ln": self.trainable_ln,
            "ln_eps": self.ln_eps,
            "linear1": self.linear1.serialize(),
        }
        if self.update_chnnl_2:
            data.update(
                {
                    "linear2": self.linear2.serialize(),
                }
            )
        if self.update_g1_has_conv:
            data.update(
                {
                    "proj_g1g2": self.proj_g1g2.serialize(),
                }
            )
        if self.update_g2_has_g1g1:
            data.update(
                {
                    "proj_g1g1g2": self.proj_g1g1g2.serialize(),
                }
            )
        if self.update_g2_has_attn or self.update_h2:
            data.update(
                {
                    "attn2g_map": self.attn2g_map.serialize(),
                }
            )
            if self.update_g2_has_attn:
                data.update(
                    {
                        "attn2_mh_apply": self.attn2_mh_apply.serialize(),
                        "attn2_lm": self.attn2_lm.serialize(),
                    }
                )

            if self.update_h2:
                data.update(
                    {
                        "attn2_ev_apply": self.attn2_ev_apply.serialize(),
                    }
                )
        if self.update_g1_has_attn:
            data.update(
                {
                    "loc_attn": self.loc_attn.serialize(),
                }
            )
        if self.update_style == "res_residual":
            data.update(
                {
                    "g1_residual": self.g1_residual,
                    "g2_residual": self.g2_residual,
                    "h2_residual": self.h2_residual,
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "RepformerLayer":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        linear1 = data.pop("linear1")
        update_chnnl_2 = data["update_chnnl_2"]
        update_g1_has_conv = data["update_g1_has_conv"]
        update_g2_has_g1g1 = data["update_g2_has_g1g1"]
        update_g2_has_attn = data["update_g2_has_attn"]
        update_h2 = data["update_h2"]
        update_g1_has_attn = data["update_g1_has_attn"]
        update_style = data["update_style"]

        linear2 = data.pop("linear2", None)
        proj_g1g2 = data.pop("proj_g1g2", None)
        proj_g1g1g2 = data.pop("proj_g1g1g2", None)
        attn2g_map = data.pop("attn2g_map", None)
        attn2_mh_apply = data.pop("attn2_mh_apply", None)
        attn2_lm = data.pop("attn2_lm", None)
        attn2_ev_apply = data.pop("attn2_ev_apply", None)
        loc_attn = data.pop("loc_attn", None)
        g1_residual = data.pop("g1_residual", [])
        g2_residual = data.pop("g2_residual", [])
        h2_residual = data.pop("h2_residual", [])

        obj = cls(**data)
        obj.linear1 = NativeLayer.deserialize(linear1)
        if update_chnnl_2:
            assert isinstance(linear2, dict)
            obj.linear2 = NativeLayer.deserialize(linear2)
        if update_g1_has_conv:
            assert isinstance(proj_g1g2, dict)
            obj.proj_g1g2 = NativeLayer.deserialize(proj_g1g2)
        if update_g2_has_g1g1:
            assert isinstance(proj_g1g1g2, dict)
            obj.proj_g1g1g2 = NativeLayer.deserialize(proj_g1g1g2)
        if update_g2_has_attn or update_h2:
            assert isinstance(attn2g_map, dict)
            obj.attn2g_map = Atten2Map.deserialize(attn2g_map)
            if update_g2_has_attn:
                assert isinstance(attn2_mh_apply, dict)
                assert isinstance(attn2_lm, dict)
                obj.attn2_mh_apply = Atten2MultiHeadApply.deserialize(attn2_mh_apply)
                obj.attn2_lm = LayerNorm.deserialize(attn2_lm)
            if update_h2:
                assert isinstance(attn2_ev_apply, dict)
                obj.attn2_ev_apply = Atten2EquiVarApply.deserialize(attn2_ev_apply)
        if update_g1_has_attn:
            assert isinstance(loc_attn, dict)
            obj.loc_attn = LocalAtten.deserialize(loc_attn)
        if update_style == "res_residual":
            obj.g1_residual = g1_residual
            obj.g2_residual = g2_residual
            obj.h2_residual = h2_residual
        return obj
