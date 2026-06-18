# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(2)-equivariant message-passing layers for DPA4/SeZM.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.so2``.
It defines the reduced-layout SO(2) linear operator, the edge-conditioned
radial degree mixer, and the edge convolution used inside SeZM interaction
blocks.

Padded-edge adaptation
----------------------
The pt ``SO2Convolution`` consumes a flat *sparse* edge list and aggregates
per destination node with ``index_add_``. The dpmodel port uses the padded,
frame-explicit edge layout documented in ``edge_cache.EdgeCache``
(``E = nf * nloc * nnei`` with invalid slots marked by ``edge_mask == 0``),
so every destination aggregation becomes a masked sum over the ``nnei`` axis
and the destination-wise softmax becomes a masked softmax over ``nnei``
(see ``attention.segment_envelope_gated_softmax``). Per-edge math (the
SO(2) linear application, the Wigner rotations via the ``D_to_m``
projections, and the radial modulation) is identical to pt, just evaluated
over the padded edge axis.

Branches guarded with ``NotImplementedError`` (flags unused by the core DPA4
config): ``so2_attn_res != "none"``, ``layer_scale``, ``s2_activation``,
``atten_f_mix``, ``atten_v_proj``, ``atten_o_proj``.

The cross-mode SO(3)/S2 grid products (``node_wise_s2``/``node_wise_so3`` and
``message_node_s2``/``message_node_so3``) are ported and wired into the
convolution, mirroring the pt ``SO2Convolution`` forward placement.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
    xp_sigmoid,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .activation import (
    GatedActivation,
)
from .attention import (
    segment_envelope_gated_softmax,
)
from .grid_net import (
    S2GridNet,
    SO3GridNet,
)
from .indexing import (
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim_of_lmax,
    map_degree_idx,
    project_D_to_m,
    project_Dt_from_m,
)
from .norm import (
    ReducedEquivariantRMSNorm,
    ScalarRMSNorm,
)
from .projection import (
    resolve_s2_grid_resolution,
)
from .so3 import (
    ChannelLinear,
    FocusLinear,
    SO3Linear,
)
from .utils import (
    ATTN_RES_MODES,
    init_trunc_normal_fan_in_out,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeCache,
    )


def _compute_precision(precision: str) -> str:
    """Promote fp16/bf16 to fp32 (dpmodel analog of pt ``get_promoted_dtype``)."""
    name = np.dtype(PRECISION_DICT[precision.lower()]).name
    if "float16" in name:  # matches float16 and bfloat16
        return "float32"
    return precision


def _check_shape_assign(obj: Any, attr: str, value: Any, dtype: Any, key: str) -> None:
    """Assign ``value`` (cast to ``dtype``) to ``obj.attr`` with a shape check."""
    expected = getattr(obj, attr)
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != expected.shape:
        raise ValueError(
            f"{key} shape {arr.shape} does not match the expected shape "
            f"{expected.shape}"
        )
    setattr(obj, attr, arr)


def _check_index_table(expected: np.ndarray, value: Any, key: str) -> None:
    """Validate that a serialized integer index table matches the rebuilt one."""
    arr = np.asarray(value, dtype=np.int64)
    # ``expected`` is a rebuilt buffer that may be a (possibly CUDA) torch
    # tensor in the pt_expt backend; ``np.asarray`` raises on CUDA tensors and
    # ``np.array_equal`` would silently swallow that into ``False``. Convert via
    # ``to_numpy_array`` (dlpack-through-CPU fallback) before comparing.
    if not np.array_equal(arr.reshape(-1), to_numpy_array(expected).reshape(-1)):
        raise ValueError(f"{key} does not match the table derived from the config")


class SO2Linear(NativeOP):
    """
    SO(2)-equivariant linear mixing in the edge-aligned local frame.

    Coefficient layout (m-major, truncated by mmax)
    ------------------------------------------------
    The coefficient axis D_m_trunc is ordered by |m| groups::

        [  m=0: l=0..lmax  |  m=1: (l,-1) then (l,+1)  |  ...  |  m=mmax: ... ]
         |___ lmax+1 ____|   |_______ 2*(lmax) ________|

    Each |m| group is contiguous, enabling per-group block matmuls.

    Block-diagonal weight structure
    -------------------------------
    The conceptual full weight matrix is block-diagonal over |m| groups::

        W = diag[W_m0, B_m1, B_m2, ..., B_mmax]

    - ``W_m0``: unconstrained ``(num_l*Cin, num_l*Cout)`` block for m=0.
      Cross-l mixing is allowed since m=0 coefficients are real scalars.

    - ``B_m`` (|m|>0): SO(2)-constrained 2x2 block coupling (-m, +m) pairs::

          B_m = [ W_u^T , -W_v^T ]     where W_u, W_v are learnable
                [ W_v^T ,  W_u^T ]     (num_l*Cin, num_l*Cout) each.

      This structure is the real-valued form of complex multiplication
      ``(u + iv)(a + ib) = (ua - vb) + i(va + ub)``, which guarantees
      SO(2) equivariance.

    Unlike pt (which assembles the dense block-diagonal matrix and applies a
    single ``einsum``), the dpmodel forward contracts the diagonal blocks
    directly with slicing + matmul + concat, which is array-API friendly and
    numerically equivalent (the off-block entries are exact zeros).

    Parameters
    ----------
    lmax
        Maximum spherical harmonic degree.
    mmax
        Maximum SO(2) order (|m|) to mix. If None, defaults to ``lmax``.
    in_channels
        Number of input channels per (l, m) coefficient.
    out_channels
        Number of output channels per (l, m) coefficient.
    n_focus
        Number of independent focus streams. Each stream has its own
        weight matrices.
    precision
        Parameter precision.
    mlp_bias
        Whether to use bias for l=0 (scalar) components.
    seed
        Random seed for weight initialization.
    trainable
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        precision: str = DEFAULT_PRECISION,
        mlp_bias: bool = False,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.precision = precision
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]

        # === Step 1. Build m-major coefficient layout ===
        # Map each |m| group to contiguous index ranges in the flattened axis.
        # Example for lmax=2, mmax=2:
        #   m=0 : indices [0, 1, 2]        (l=0,1,2)
        #   m=1-: indices [3, 4]            (l=1,2 with -m)
        #   m=1+: indices [5, 6]            (l=1,2 with +m)
        #   m=2-: index  [7]               (l=2   with -m)
        #   m=2+: index  [8]               (l=2   with +m)
        #   => reduced_dim = 9
        m0_size = self.lmax + 1
        self.m0_idx = np.arange(m0_size, dtype=np.int64)

        pos_indices_list: list[np.ndarray] = []
        neg_indices_list: list[np.ndarray] = []
        # Each entry: (neg_start, pos_start, num_l) for a fixed |m|.
        # These ranges are contiguous in m-major layout.
        m_ranges: list[tuple[int, int, int]] = []

        offset = m0_size
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            neg_start = offset
            pos_start = offset + num_l
            neg_indices_list.append(
                np.arange(neg_start, neg_start + num_l, dtype=np.int64)
            )
            pos_indices_list.append(
                np.arange(pos_start, pos_start + num_l, dtype=np.int64)
            )
            m_ranges.append((neg_start, pos_start, num_l))
            offset += 2 * num_l

        self.reduced_dim = int(offset)

        if len(pos_indices_list) > 0:
            self.pos_indices = np.concatenate(pos_indices_list)
            self.neg_indices = np.concatenate(neg_indices_list)
        else:
            self.pos_indices = np.empty(0, dtype=np.int64)
            self.neg_indices = np.empty(0, dtype=np.int64)
        self._m_ranges = m_ranges

        # === Step 2. Learnable weight parameters ===
        # weight_m0: folded (num_l*Cin, F*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, num_l*Cout).
        #   Cross-l mixing is allowed because m=0 coefficients are real.
        num_m0 = self.lmax + 1
        num_in_m0 = num_m0 * self.in_channels
        num_out_m0 = num_m0 * self.out_channels
        weight_m0 = np.empty((num_in_m0, self.n_focus * num_out_m0), dtype=prec)
        weight_m0_view = weight_m0.reshape(num_in_m0, self.n_focus, num_out_m0)
        for focus_idx in range(self.n_focus):
            init_trunc_normal_fan_in_out(
                weight_m0_view[:, focus_idx, :], child_seed(seed, 1000 + focus_idx)
            )
        self.weight_m0 = weight_m0
        if self.mlp_bias:
            self.bias0: np.ndarray | None = np.zeros(
                (self.n_focus * self.out_channels,), dtype=prec
            )
        else:
            self.bias0 = None

        # weight_m[i]: folded (num_l*Cin, F*2*num_l*Cout) storage — (in, out)
        #   convention. Runtime view: (num_l*Cin, F, 2*num_l*Cout).
        #   The factor of 2 comes from storing W_u and W_v concatenated along the
        #   output axis. Scaling by 1/sqrt(2) compensates for the doubled
        #   parameter count.
        self.weight_m: list[np.ndarray] = []
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            num_in = num_l * self.in_channels
            num_out = 2 * num_l * self.out_channels
            weight = np.empty((num_in, self.n_focus * num_out), dtype=prec)
            weight_view = weight.reshape(num_in, self.n_focus, num_out)
            for focus_idx in range(self.n_focus):
                init_trunc_normal_fan_in_out(
                    weight_view[:, focus_idx, :],
                    child_seed(seed, 2000 + m * 100 + focus_idx),
                )
            # Apply scaling for SO(2) equivariance
            weight *= 1.0 / math.sqrt(2.0)
            self.weight_m.append(weight)

        # === Step 3. Precompute flattened slice ranges for the block matmuls ===
        # Each |m|>0 group occupies two sub-blocks (neg, pos) in the flattened
        # coefficient*channel axis.
        # Tuple layout: (neg_i0, neg_i1, pos_i0, pos_i1,   <- input ranges
        #                neg_o0, neg_o1, pos_o0, pos_o1)   <- output ranges
        self._m0_in = (self.lmax + 1) * self.in_channels
        self._m0_out = (self.lmax + 1) * self.out_channels
        self._block_slices: list[tuple[int, int, int, int, int, int, int, int]] = []
        for neg_start, pos_start, num_l in self._m_ranges:
            ib = num_l * self.in_channels
            ob = num_l * self.out_channels
            self._block_slices.append(
                (
                    neg_start * self.in_channels,
                    neg_start * self.in_channels + ib,
                    pos_start * self.in_channels,
                    pos_start * self.in_channels + ib,
                    neg_start * self.out_channels,
                    neg_start * self.out_channels + ob,
                    pos_start * self.out_channels,
                    pos_start * self.out_channels + ob,
                )
            )

    @staticmethod
    def _focus_matmul(xp: Any, x: Any, w: Any) -> Any:
        """Per-focus matmul: einsum("efi,fio->efo") via broadcast batched matmul.

        Parameters
        ----------
        x
            Input with shape (E, F, in_blk).
        w
            Weight with shape (F, in_blk, out_blk).
        """
        return xp.matmul(x[:, :, None, :], w[None, ...])[..., 0, :]

    def call(self, x: Any) -> Any:
        """
        Parameters
        ----------
        x
            Input with shape (E, F, D_m_trunc, Cin), where D_m_trunc is the
            coefficient dimension of the m-major layout truncated by `mmax`.

        Returns
        -------
        Array
            Output with shape (E, F, D_m_trunc, Cout).
        """
        xp = array_api_compat.array_namespace(x)
        # === Step 1. Flatten coefficient + channel axes for matmul ===
        # (E, F, D_m, Cin) -> (E, F, D_m*Cin)
        n_edge = x.shape[0]
        in_dim_total = self.reduced_dim * self.in_channels
        x_flat = xp.reshape(x, (n_edge, self.n_focus, in_dim_total))

        # === Step 2. Contract the diagonal |m| blocks ===
        # m=0 block: unconstrained (num_l*Cin, num_l*Cout) per focus.
        num_m0 = self.lmax + 1
        device = array_api_compat.device(x)
        weight_m0 = xp.reshape(
            xp_asarray_nodetach(xp, self.weight_m0[...], device=device),
            (num_m0 * self.in_channels, self.n_focus, num_m0 * self.out_channels),
        )
        weight_m0 = xp.permute_dims(weight_m0, (1, 0, 2))  # (F, in, out)
        out_blocks = [self._focus_matmul(xp, x_flat[:, :, : self._m0_in], weight_m0)]

        # |m|>0 blocks: real-valued complex multiplication on (-m, +m) pairs.
        for m_idx, w in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0  # in_block size
            ob = no1 - no0  # out_block size
            w = xp.reshape(
                xp_asarray_nodetach(xp, w[...], device=device),
                (ib, self.n_focus, 2 * ob),
            )
            w = xp.permute_dims(w, (1, 0, 2))  # (F, in_blk, 2*out_blk)
            w_u = w[:, :, :ob]  # (F, in_blk, out_blk)
            w_v = w[:, :, ob:]  # (F, in_blk, out_blk)
            x_neg = x_flat[:, :, ni0:ni1]
            x_pos = x_flat[:, :, pi0:pi1]
            # 2x2 coupling: neg_out = x_neg @ W_u - x_pos @ W_v
            #               pos_out = x_neg @ W_v + x_pos @ W_u
            out_blocks.append(
                self._focus_matmul(xp, x_neg, w_u) - self._focus_matmul(xp, x_pos, w_v)
            )
            out_blocks.append(
                self._focus_matmul(xp, x_neg, w_v) + self._focus_matmul(xp, x_pos, w_u)
            )

        out_flat = (
            xp.concat(out_blocks, axis=-1) if len(out_blocks) > 1 else out_blocks[0]
        )
        out = xp.reshape(
            out_flat, (n_edge, self.n_focus, self.reduced_dim, self.out_channels)
        )

        # === Step 3. Bias on l=0 scalar index ===
        if self.mlp_bias:
            bias0 = xp.reshape(
                xp_asarray_nodetach(xp, self.bias0[...], device=device),
                (self.n_focus, self.out_channels),
            )
            out0 = out[:, :, :1, :] + bias0[None, :, None, :]
            out = xp.concat([out0, out[:, :, 1:, :]], axis=2)
        return out

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables = {
            "m0_idx": to_numpy_array(self.m0_idx),
            "pos_indices": to_numpy_array(self.pos_indices),
            "neg_indices": to_numpy_array(self.neg_indices),
            "weight_m0": to_numpy_array(self.weight_m0),
        }
        if self.mlp_bias:
            variables["bias0"] = to_numpy_array(self.bias0)
        for m_idx, w in enumerate(self.weight_m):
            variables[f"weight_m.{m_idx}"] = to_numpy_array(w)
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        prec = PRECISION_DICT[self.precision.lower()]
        _check_index_table(self.m0_idx, variables["m0_idx"], "m0_idx")
        _check_index_table(self.pos_indices, variables["pos_indices"], "pos_indices")
        _check_index_table(self.neg_indices, variables["neg_indices"], "neg_indices")
        _check_shape_assign(
            self, "weight_m0", variables["weight_m0"], prec, "weight_m0"
        )
        if self.mlp_bias:
            self.bias0 = np.asarray(variables["bias0"], dtype=prec).reshape(
                self.bias0.shape
            )
        # Rebuild the list and assign the whole attribute (rather than
        # item-assignment) so that pt_expt, which converts the list to a
        # torch ParameterList, can re-convert the new value cleanly.
        new_weight_m = []
        for m_idx in range(len(self.weight_m)):
            key = f"weight_m.{m_idx}"
            value = np.asarray(variables[key], dtype=prec)
            if value.shape != tuple(self.weight_m[m_idx].shape):
                raise ValueError(
                    f"{key} shape {value.shape} does not match the expected "
                    f"shape {tuple(self.weight_m[m_idx].shape)}"
                )
            new_weight_m.append(value)
        self.weight_m = new_weight_m

    def serialize(self) -> dict[str, Any]:
        """Serialize the SO2Linear to a dict (pt-compatible format)."""
        return {
            "@class": "SO2Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "mlp_bias": self.mlp_bias,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Linear:
        """Deserialize an SO2Linear from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Linear":
            raise ValueError(f"Invalid class for SO2Linear: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            mmax=int(config["mmax"]),
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            n_focus=int(config["n_focus"]),
            precision=str(config["precision"]),
            mlp_bias=bool(config["mlp_bias"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        obj._load_variables(variables)
        return obj


class DynamicRadialDegreeMixer(NativeOP):
    """
    Edge-conditioned degree mixer in the SO(2) reduced local layout.

    The mixer replaces per-degree scalar radial modulation by an edge-conditioned
    degree kernel without channel output mixing:

        degree:
            y[e, l_out, m, c] = sum_l_in W[e, l_in, l_out, |m|] x[e, l_in, m, c]
        degree_channel:
            y[e, l_out, m, c] = sum_l_in W[e, l_in, l_out, |m|, c] x[e, l_in, m, c]

    `mode="degree"` shares W across channels. `mode="degree_channel"` gives each
    channel its own W, optionally with a low-rank channel factorization.

    The pt ``index_copy_`` scatter of the compact kernel into the dense
    ``(D_m, D_m)`` layout is replaced by a precomputed gather index + mask
    (functionally identical, array-API friendly).
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        channels: int,
        mode: str,
        rank: int = 0,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.channels = int(channels)
        if self.channels < 1:
            raise ValueError("`channels` must be positive")
        self.mode = str(mode).lower()
        if self.mode not in {"degree", "degree_channel"}:
            raise ValueError("`mode` must be one of 'degree' or 'degree_channel'")
        self.rank = int(rank)
        if self.rank < 0:
            raise ValueError("`rank` must be non-negative")
        self.precision = precision
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]

        # m-major reduced layout: m=0 block followed by (-m, +m) blocks.
        self.reduced_dim = (self.lmax + 1) + sum(
            2 * (self.lmax - m + 1) for m in range(1, self.mmax + 1)
        )
        self.degree_kernel_size = sum(
            (self.lmax - m + 1) ** 2 for m in range(self.mmax + 1)
        )
        self.input_dim = (self.lmax + 1) * self.channels
        if self.mode == "degree":
            self.proj_out_dim = self.degree_kernel_size
        elif self.rank > 0:
            self.proj_out_dim = self.degree_kernel_size * self.rank
        else:
            self.proj_out_dim = self.degree_kernel_size * self.channels

        weight = np.empty((self.input_dim, self.proj_out_dim), dtype=prec)
        init_trunc_normal_fan_in_out(weight, child_seed(seed, 0))
        self.weight = weight

        if self.mode == "degree_channel" and self.rank > 0:
            channel_basis = np.empty((self.rank, self.channels), dtype=prec)
            init_trunc_normal_fan_in_out(channel_basis, child_seed(seed, 1))
            self.channel_basis: np.ndarray | None = channel_basis
        else:
            self.channel_basis = None

        compact_idx, dense_idx = self._build_dense_scatter_indices()
        self.kernel_compact_index = compact_idx
        self.kernel_dense_index = dense_idx
        # Gather-form of pt's index_copy_ scatter:
        #   dense[:, dense_idx[j]] = compact[:, compact_idx[j]]
        # becomes
        #   dense = take(compact, gather_index, axis=1) * scatter_mask
        dense_size = self.reduced_dim * self.reduced_dim
        gather_index = np.zeros(dense_size, dtype=np.int64)
        scatter_mask = np.zeros(dense_size, dtype=prec)
        gather_index[dense_idx] = compact_idx
        scatter_mask[dense_idx] = 1.0
        self._dense_gather_index = gather_index
        self._dense_scatter_mask = scatter_mask

    def _build_dense_scatter_indices(self) -> tuple[np.ndarray, np.ndarray]:
        compact_indices: list[int] = []
        dense_indices: list[int] = []
        compact_offset = 0
        reduced_dim = self.reduced_dim

        def append_block(start_in: int, start_out: int, num_l: int) -> None:
            for l_in in range(num_l):
                for l_out in range(num_l):
                    compact_indices.append(compact_offset + l_in * num_l + l_out)
                    # Store dense kernels in matmul layout (out, in) so forward
                    # can use a batched matmul without transposing.
                    dense_indices.append(
                        (start_out + l_out) * reduced_dim + start_in + l_in
                    )

        # m=0: single real block.
        num_l0 = self.lmax + 1
        append_block(0, 0, num_l0)
        compact_offset += num_l0 * num_l0

        # |m|>0: same degree kernel is applied to the negative and positive
        # signed-m blocks. No cross signed-m mixing is introduced.
        offset = num_l0
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            neg_start = offset
            pos_start = offset + num_l
            append_block(neg_start, neg_start, num_l)
            append_block(pos_start, pos_start, num_l)
            compact_offset += num_l * num_l
            offset += 2 * num_l

        return (
            np.asarray(compact_indices, dtype=np.int64),
            np.asarray(dense_indices, dtype=np.int64),
        )

    def _project_radial(self, xp: Any, radial_feat: Any) -> Any:
        radial_m0 = xp.reshape(
            radial_feat[:, : self.lmax + 1, :],
            (radial_feat.shape[0], self.input_dim),
        )
        weight = xp_asarray_nodetach(
            xp, self.weight[...], device=array_api_compat.device(radial_feat)
        )
        return xp.matmul(radial_m0, weight)

    def _scatter_dense(self, xp: Any, compact: Any, device: Any) -> Any:
        """Scatter the compact per-block kernel into the dense (D_m*D_m, ...) layout."""
        gather_index = xp_asarray_nodetach(xp, self._dense_gather_index, device=device)
        scatter_mask = xp.astype(
            xp_asarray_nodetach(xp, self._dense_scatter_mask, device=device),
            compact.dtype,
        )
        dense = xp.take(compact, gather_index, axis=1)
        if compact.ndim == 2:
            return dense * scatter_mask[None, :]
        return dense * scatter_mask[None, :, None]

    def call(self, x_local: Any, radial_feat: Any) -> Any:
        """
        Parameters
        ----------
        x_local
            Local reduced features with shape (E, D_m, C_wide).
        radial_feat
            Invariant radial/type features with shape (E, D_m, C_wide).
        """
        if x_local.shape != radial_feat.shape:
            raise ValueError("`x_local` and `radial_feat` must have the same shape")
        if x_local.shape[1] != self.reduced_dim or x_local.shape[2] != self.channels:
            raise ValueError("Input shape is incompatible with this mixer")

        xp = array_api_compat.array_namespace(x_local)
        device = array_api_compat.device(x_local)
        n_edge = x_local.shape[0]
        kernel_flat = self._project_radial(xp, radial_feat)
        if self.mode == "degree":
            kernel = xp.reshape(
                self._scatter_dense(xp, kernel_flat, device),
                (n_edge, self.reduced_dim, self.reduced_dim),
            )
            return xp.matmul(kernel, x_local)

        if self.rank > 0:
            compact = xp.reshape(
                kernel_flat, (n_edge, self.degree_kernel_size, self.rank)
            )
            kernel = xp.reshape(
                self._scatter_dense(xp, compact, device),
                (n_edge, self.reduced_dim, self.reduced_dim, self.rank),
            )
            # einsum "eoir,eic->eorc" as a broadcast batched matmul:
            # (E, o, r, i) @ (E, 1, i, c) -> (E, o, r, c)
            kernel = xp.permute_dims(kernel, (0, 1, 3, 2))
            mixed = xp.matmul(kernel, x_local[:, None, :, :])
            channel_basis = xp.reshape(
                xp_asarray_nodetach(xp, self.channel_basis[...], device=device),
                (1, 1, self.rank, self.channels),
            )
            return xp.sum(mixed * channel_basis, axis=2)

        compact = xp.reshape(
            kernel_flat, (n_edge, self.degree_kernel_size, self.channels)
        )
        kernel = xp.reshape(
            self._scatter_dense(xp, compact, device),
            (n_edge, self.reduced_dim, self.reduced_dim, self.channels),
        )
        # einsum "eoic,eic->eoc"
        return xp.sum(kernel * x_local[:, None, :, :], axis=2)

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables = {"weight": to_numpy_array(self.weight)}
        if self.channel_basis is not None:
            variables["channel_basis"] = to_numpy_array(self.channel_basis)
        variables["kernel_compact_index"] = to_numpy_array(self.kernel_compact_index)
        variables["kernel_dense_index"] = to_numpy_array(self.kernel_dense_index)
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        prec = PRECISION_DICT[self.precision.lower()]
        _check_index_table(
            self.kernel_compact_index,
            variables["kernel_compact_index"],
            "kernel_compact_index",
        )
        _check_index_table(
            self.kernel_dense_index,
            variables["kernel_dense_index"],
            "kernel_dense_index",
        )
        _check_shape_assign(self, "weight", variables["weight"], prec, "weight")
        if self.channel_basis is not None:
            _check_shape_assign(
                self, "channel_basis", variables["channel_basis"], prec, "channel_basis"
            )

    def serialize(self) -> dict[str, Any]:
        """Serialize the DynamicRadialDegreeMixer to a dict.

        The pt class has no ``serialize()``; the ``@variables`` keys here
        match the pt ``state_dict`` key names.
        """
        return {
            "@class": "DynamicRadialDegreeMixer",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "channels": self.channels,
                "mode": self.mode,
                "rank": self.rank,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DynamicRadialDegreeMixer:
        """Deserialize a DynamicRadialDegreeMixer from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "DynamicRadialDegreeMixer":
            raise ValueError(f"Invalid class for DynamicRadialDegreeMixer: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(
            lmax=int(config["lmax"]),
            mmax=int(config["mmax"]),
            channels=int(config["channels"]),
            mode=str(config["mode"]),
            rank=int(config["rank"]),
            precision=str(config["precision"]),
            trainable=bool(config["trainable"]),
            seed=config.get("seed"),
        )
        obj._load_variables(variables)
        return obj


class SO2Convolution(NativeOP):
    """
    SO(2)-equivariant edge convolution with cached geometry and rotations.

    This module consumes node features in packed SO(3) layout `(N, D, C)` and
    performs edge message passing in the reduced m-major local layout. The
    operation pipeline is:

    1. `pre_focus_mix`: project node features `(N, D, C)` to the SO(2) hidden width.
    2. rotate global -> local reduced basis with cached `D_to_m`.
    3. radial modulation in reduced layout.
    4. `so2_layers` stacked local mixers:
       `inter_norm -> SO2Linear -> non_linearity -> residual`.
    5. rotate local -> global with cached `Dt_from_m`.
    6. edge aggregation (plain envelope masked sum or envelope-aware masked
       softmax attention with output-side head gate); see the module
       docstring for the padded-edge adaptation.
    7. `post_focus_mix`: project aggregated hidden messages back to `(N, D, C)`.

    See the pt ``SO2Convolution`` docstring for the full parameter
    documentation; this port keeps the same constructor parameters with
    ``dtype`` replaced by ``precision``. Flags unused by the core DPA4 config
    raise ``NotImplementedError`` (listed in the module docstring).
    """

    def __init__(
        self,
        *,
        lmax: int,
        mmax: int | None = None,
        kmax: int = 1,
        channels: int,
        n_focus: int = 1,
        focus_dim: int = 0,
        focus_compete: bool = True,
        so2_norm: bool = False,
        so2_layers: int = 4,
        so2_attn_res: str = "none",
        layer_scale: bool = False,
        n_atten_head: int = 1,
        atten_f_mix: bool = False,
        atten_v_proj: bool = False,
        atten_o_proj: bool = False,
        s2_activation: bool = False,
        node_wise_grid_mlp: bool = False,
        node_wise_grid_branch: int = 0,
        message_node_grid_mlp: bool = False,
        message_node_grid_branch: int = 0,
        node_wise_s2: bool = False,
        node_wise_so3: bool = False,
        message_node_s2: bool = False,
        message_node_so3: bool = False,
        lebedev_quadrature: bool = False,
        activation_function: str = "silu",
        mlp_bias: bool = False,
        radial_so2_mode: str = "none",
        radial_so2_rank: int = 0,
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
        trainable: bool = True,
    ) -> None:
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.kmax = int(kmax)
        if self.kmax < 0:
            raise ValueError("`kmax` must be non-negative")
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        if self.n_focus < 1:
            raise ValueError("`n_focus` must be >= 1")
        self.focus_dim = int(focus_dim)
        if self.focus_dim < 0:
            raise ValueError("`focus_dim` must be >= 0")
        self.so2_focus_dim = self.channels if self.focus_dim == 0 else self.focus_dim
        self.hidden_channels = int(self.n_focus * self.so2_focus_dim)
        self.use_hidden_projection = self.hidden_channels != self.channels
        self.focus_compete = bool(focus_compete)
        self.focus_softmax_tau = 1.0
        self.focus_label_smoothing = 0.02
        self.so2_norm = bool(so2_norm)
        self.so2_layers = int(so2_layers)
        if self.so2_layers < 1:
            raise ValueError("`so2_layers` must be >= 1")
        self.so2_attn_res_mode = str(so2_attn_res).lower()
        if self.so2_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`so2_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        if self.so2_attn_res_mode != "none":
            raise NotImplementedError(
                "so2_attn_res != 'none' (DepthAttnRes) is not ported to dpmodel"
            )
        self.layer_scale = bool(layer_scale)
        if self.layer_scale:
            raise NotImplementedError("layer_scale=True is not ported to dpmodel")
        self.n_atten_head = int(n_atten_head)
        if self.n_atten_head < 0:
            raise ValueError("`n_atten_head` must be non-negative")
        self.atten_f_mix = bool(atten_f_mix)
        if self.atten_f_mix:
            raise NotImplementedError("atten_f_mix=True is not ported to dpmodel")
        self.use_atten_v_proj = bool(atten_v_proj)
        if self.use_atten_v_proj:
            raise NotImplementedError("atten_v_proj=True is not ported to dpmodel")
        self.use_atten_o_proj = bool(atten_o_proj)
        if self.use_atten_o_proj:
            raise NotImplementedError("atten_o_proj=True is not ported to dpmodel")
        self.s2_activation = bool(s2_activation)
        if self.s2_activation:
            raise NotImplementedError(
                "s2_activation=True (so2_s2_activation) is not ported to dpmodel"
            )
        self.node_wise_grid_mlp = bool(node_wise_grid_mlp)
        self.node_wise_grid_branch = int(node_wise_grid_branch)
        self.message_node_grid_mlp = bool(message_node_grid_mlp)
        self.message_node_grid_branch = int(message_node_grid_branch)
        if min(self.node_wise_grid_branch, self.message_node_grid_branch) < 0:
            raise ValueError("grid branch counts must be non-negative")
        self.node_wise_s2 = bool(node_wise_s2)
        self.node_wise_so3 = bool(node_wise_so3)
        self.message_node_s2 = bool(message_node_s2)
        self.message_node_so3 = bool(message_node_so3)
        self.lebedev_quadrature = bool(lebedev_quadrature)
        self.s2_grid_method = "lebedev" if self.lebedev_quadrature else "e3nn"
        self.s2_grid_resolution = resolve_s2_grid_resolution(
            self.lmax,
            self.mmax,
            method=self.s2_grid_method,
        )
        base_full_grid_resolution = resolve_s2_grid_resolution(
            self.lmax,
            self.lmax,
            method=self.s2_grid_method,
        )
        # Mirror pt: the e3nn product-grid branch squares the max resolution.
        # dpmodel only ports the Lebedev backend (the e3nn S2GridNet raises),
        # so this just preserves the config-recorded resolution for parity.
        self.s2_full_grid_resolution = (
            [max(base_full_grid_resolution), max(base_full_grid_resolution)]
            if self.s2_grid_method == "e3nn"
            else base_full_grid_resolution
        )
        self.activation_function = str(activation_function)
        self.attn_n_focus = self.n_focus
        self.attn_focus_dim = self.so2_focus_dim
        if self.n_atten_head > 0 and self.attn_focus_dim % self.n_atten_head != 0:
            raise ValueError(
                "`n_atten_head` must divide the attention width "
                "(`focus_dim` or `n_focus * focus_dim` when `atten_f_mix=True`)"
            )
        self.head_dim = (
            None
            if self.n_atten_head == 0
            else int(self.attn_focus_dim // self.n_atten_head)
        )
        self.mlp_bias = bool(mlp_bias)
        self.radial_so2_mode = str(radial_so2_mode).lower()
        if self.radial_so2_mode not in {"none", "degree", "degree_channel"}:
            raise ValueError(
                "`radial_so2_mode` must be one of 'none', 'degree', or 'degree_channel'"
            )
        self.radial_so2_rank = int(radial_so2_rank)
        if self.radial_so2_rank < 0:
            raise ValueError("`radial_so2_rank` must be non-negative")
        self.eps = float(eps)
        self.ebed_dim_full = get_so3_dim_of_lmax(self.lmax)
        self.precision = precision
        self.compute_precision = _compute_precision(precision)
        self.trainable = bool(trainable)
        prec = PRECISION_DICT[self.precision.lower()]

        # === Step 1. Precompute coefficient indices for m-major reduced layout ===
        self.coeff_index_m = build_m_major_index(self.lmax, self.mmax)
        self.degree_index_m = build_m_major_l_index(self.lmax, self.mmax)
        degree_index_full = map_degree_idx(self.lmax)
        self.rotate_inv_rescale_full = build_rotate_inv_rescale(
            self.lmax,
            self.mmax,
            degree_index_full,
            dtype=prec,
        )
        self.reduced_dim = int(self.coeff_index_m.shape[0])

        # === Step 2. Split deterministic seeds at the module top-level ===
        seed_so2_stack = child_seed(seed, 0)
        seed_non_linearities = child_seed(seed, 1)
        seed_so3_pre = child_seed(seed, 2)
        seed_so3_post = child_seed(seed, 3)
        seed_gate = child_seed(seed, 4)
        seed_radial_hidden = child_seed(seed, 6)
        seed_radial_degree = child_seed(seed, 7)
        seed_node_wise_s2 = child_seed(seed, 8)
        seed_message_node_s2 = child_seed(seed, 9)

        # === Step 3. Multiple SO2Linear layers ===
        # (s2_activation is guarded above, so out_channels == so2_focus_dim.)
        self.so2_linears = [
            SO2Linear(
                lmax=self.lmax,
                mmax=self.mmax,
                in_channels=self.so2_focus_dim,
                out_channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                precision=self.precision,
                mlp_bias=self.mlp_bias,
                seed=child_seed(seed_so2_stack, i),
                trainable=trainable,
            )
            for i in range(self.so2_layers)
        ]

        # === Step 4. Intermediate norms (Optional) ===
        # pt appends nn.Identity() entries; dpmodel uses None for Identity.
        inter_norms: list[ReducedEquivariantRMSNorm | None] = []
        if self.so2_norm:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(
                    ReducedEquivariantRMSNorm(
                        lmax=self.lmax,
                        mmax=self.mmax,
                        channels=self.so2_focus_dim,
                        degree_index_m=self.degree_index_m,
                        n_focus=self.n_focus,
                        precision=self.compute_precision,
                        trainable=trainable,
                    )
                )
        else:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(None)
        inter_norms.append(None)
        self.so2_inter_norms = inter_norms

        # === Step 5. Intermediate non-linearity ===
        # pt appends nn.Identity() as the last entry; dpmodel uses None.
        non_linearities: list[GatedActivation | None] = []
        for i in range(max(0, self.so2_layers - 1)):
            non_linearities.append(
                GatedActivation(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    precision=self.compute_precision,
                    activation_function=self.activation_function,
                    mlp_bias=self.mlp_bias,
                    layout="nfdc",
                    trainable=trainable,
                    seed=child_seed(seed_non_linearities, i),
                )
            )
        non_linearities.append(None)
        self.non_linearities = non_linearities

        # === Step 7. Optional attention projections (n_atten_head > 0) ===
        self.attn_qk_norm: ScalarRMSNorm | None = None
        self.attn_q_proj: FocusLinear | None = None
        self.attn_k_proj: FocusLinear | None = None
        self.adamw_attn_logit_w: np.ndarray | None = None
        self.adamw_attn_z_bias_raw: np.ndarray | None = None
        self.attn_output_gate_norm: ScalarRMSNorm | None = None
        self.adamw_attn_gate_w: np.ndarray | None = None
        cprec = PRECISION_DICT[self.compute_precision.lower()]
        if self.n_atten_head > 0:
            self.attn_qk_norm = ScalarRMSNorm(
                channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                eps=self.eps,
                precision=self.compute_precision,
                trainable=trainable,
            )
            self.attn_q_proj = FocusLinear(
                in_channels=self.attn_focus_dim,
                out_channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                precision=self.compute_precision,
                bias=False,
                seed=child_seed(seed_gate, 0),
                trainable=trainable,
            )
            self.attn_k_proj = FocusLinear(
                in_channels=self.attn_focus_dim,
                out_channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                precision=self.compute_precision,
                bias=False,
                seed=child_seed(seed_gate, 1),
                trainable=trainable,
            )
            rng = np.random.default_rng(child_seed(seed_gate, 2))
            self.adamw_attn_logit_w = rng.normal(
                0.0,
                0.01,
                size=(self.attn_focus_dim, self.attn_n_focus, self.n_atten_head),
            ).astype(cprec)
            # softplus(0.5413) ~= 1.0 provides balanced initial competition.
            self.adamw_attn_z_bias_raw = np.full(
                (self.attn_n_focus, self.n_atten_head), 0.5413, dtype=cprec
            )
            self.attn_output_gate_norm = ScalarRMSNorm(
                channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                eps=self.eps,
                precision=self.compute_precision,
                trainable=trainable,
            )
            rng = np.random.default_rng(child_seed(seed_gate, 3))
            self.adamw_attn_gate_w = rng.normal(
                0.0,
                0.01,
                size=(self.attn_focus_dim, self.attn_n_focus, self.n_atten_head),
            ).astype(cprec)

        # === Step 7.5. Optional cross-focus competition ===
        self.focus_compete_norm: ScalarRMSNorm | None = None
        self.adamw_focus_compete_w: np.ndarray | None = None
        self.focus_compete_bias: np.ndarray | None = None
        if self.focus_compete and self.n_focus > 1:
            self.focus_compete_norm = ScalarRMSNorm(
                channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                precision=self.compute_precision,
                trainable=trainable,
            )
            rng = np.random.default_rng(child_seed(seed_gate, 4))
            self.adamw_focus_compete_w = rng.normal(
                0.0, 0.01, size=(self.so2_focus_dim, self.n_focus)
            ).astype(cprec)
            if self.mlp_bias:
                self.focus_compete_bias = np.zeros((self.n_focus,), dtype=cprec)

        # === Step 8. Optional radial hidden projection ===
        self.radial_hidden_proj: ChannelLinear | None = None
        if self.use_hidden_projection:
            self.radial_hidden_proj = ChannelLinear(
                in_channels=self.channels,
                out_channels=self.hidden_channels,
                precision=self.precision,
                bias=False,
                seed=seed_radial_hidden,
                trainable=trainable,
            )
        self.radial_degree_mixer: DynamicRadialDegreeMixer | None = None
        if self.radial_so2_mode != "none":
            self.radial_degree_mixer = DynamicRadialDegreeMixer(
                lmax=self.lmax,
                mmax=self.mmax,
                channels=self.hidden_channels,
                mode=self.radial_so2_mode,
                rank=self.radial_so2_rank,
                precision=self.precision,
                seed=seed_radial_degree,
                trainable=trainable,
            )

        # === Step 8.5. Optional cross-mode grid products ===
        # ``op_type`` selection mirrors pt: ``branch`` (count > 0) takes
        # precedence over ``mlp``, else ``glu``. When both ``*_s2`` and
        # ``*_so3`` are set the SO(3) branch wins (per the argcheck doc).
        node_wise_op = (
            "branch"
            if self.node_wise_grid_branch > 0
            else ("mlp" if self.node_wise_grid_mlp else "glu")
        )
        node_wise_branches = max(1, self.node_wise_grid_branch)
        message_node_op = (
            "branch"
            if self.message_node_grid_branch > 0
            else ("mlp" if self.message_node_grid_mlp else "glu")
        )
        message_node_branches = max(1, self.message_node_grid_branch)
        self.node_wise_grid_product: S2GridNet | SO3GridNet | None = None
        if self.node_wise_s2 or self.node_wise_so3:
            if self.node_wise_so3:
                self.node_wise_grid_product = SO3GridNet(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    kmax=self.kmax,
                    channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    mode="cross",
                    op_type=node_wise_op,
                    precision=self.compute_precision,
                    layout="flat",
                    coefficient_layout="m_major",
                    grid_branches=node_wise_branches,
                    mlp_bias=self.mlp_bias,
                    residual_scale_init=1e-3,
                    trainable=trainable,
                    seed=seed_node_wise_s2,
                )
            else:
                self.node_wise_grid_product = S2GridNet(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    mode="cross",
                    op_type=node_wise_op,
                    precision=self.compute_precision,
                    layout="flat",
                    grid_resolution_list=self.s2_grid_resolution,
                    coefficient_layout="m_major",
                    grid_method=self.s2_grid_method,
                    grid_branches=node_wise_branches,
                    mlp_bias=self.mlp_bias,
                    residual_scale_init=1e-3,
                    trainable=trainable,
                    seed=seed_node_wise_s2,
                )
        self.message_node_grid_product: S2GridNet | SO3GridNet | None = None
        if self.message_node_s2 or self.message_node_so3:
            if self.message_node_so3:
                self.message_node_grid_product = SO3GridNet(
                    lmax=self.lmax,
                    kmax=self.kmax,
                    channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    mode="cross",
                    op_type=message_node_op,
                    precision=self.compute_precision,
                    layout="flat",
                    coefficient_layout="packed",
                    grid_branches=message_node_branches,
                    mlp_bias=self.mlp_bias,
                    residual_scale_init=1e-3,
                    trainable=trainable,
                    seed=seed_message_node_s2,
                )
            else:
                self.message_node_grid_product = S2GridNet(
                    lmax=self.lmax,
                    mmax=self.lmax,
                    channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    mode="cross",
                    op_type=message_node_op,
                    precision=self.compute_precision,
                    layout="flat",
                    grid_resolution_list=self.s2_full_grid_resolution,
                    grid_method=self.s2_grid_method,
                    grid_branches=message_node_branches,
                    mlp_bias=self.mlp_bias,
                    residual_scale_init=1e-3,
                    trainable=trainable,
                    coefficient_layout="packed",
                    seed=seed_message_node_s2,
                )

        # === Step 9. Pre-focus channel mixing ===
        # This projects the full channel width before the SO(2) focus split.
        self.pre_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=self.hidden_channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_pre,
        )

        # === Step 10. Post-focus channel mixing ===
        self.post_focus_mix = SO3Linear(
            lmax=self.lmax,
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            n_focus=1,
            precision=self.precision,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_post,
            init_std=0.0,
        )

    def call(
        self,
        x: Any,
        edge_cache: EdgeCache,
        radial_feat: Any,
    ) -> Any:
        """
        Parameters
        ----------
        x
            Node features with shape (N, D, C), where D=(lmax+1)^2 is the
            SO(3) coefficient dimension and N = nf * nloc is the local node
            axis.
        edge_cache
            Precomputed edge cache in the padded-edge layout
            (``E = N * nnei``; see ``edge_cache.EdgeCache``). Must be
            compatible with this block's lmax.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C), already fused
            with edge type features.

        Returns
        -------
        Array
            Message updates with shape (N, D, C).
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        src, dst = edge_cache.src, edge_cache.dst
        n_node = x.shape[0]
        # Keep ``n_edge``/``n_node`` symbolic (no ``int()``): they are the
        # products ``nf*nloc*nnei`` / ``nf*nloc``. Casting to a Python int
        # specializes them to the trace-time sample shape (breaking
        # torch.export with a dynamic ``nloc`` dim); the ``Mod`` check stays
        # statically known and the ``(n_node, nnei, ...)`` reshape below
        # recovers the layout symbolically.
        n_edge = src.shape[0]
        if n_node <= 0 or n_edge % n_node != 0:
            raise ValueError(
                "padded-edge layout requires E to be a multiple of N; "
                f"got E={n_edge}, N={n_node}"
            )
        nnei = n_edge // n_node
        # Validity mask for the padded-edge layout (1 on real edges).
        edge_mask = edge_cache.edge_mask
        if edge_mask is not None:
            mask_f = xp.astype(xp.reshape(edge_mask, (n_edge,)), x.dtype)
        else:
            mask_f = xp.ones((n_edge,), dtype=x.dtype, device=device)

        # === Step 1. Pre-focus channel mixing on full width ===
        # (N, D, C_wide), C_wide = F * Cf
        x = self.pre_focus_mix(x[:, :, None, :])[:, :, 0, :]

        # === Step 2. Rotate to edge-aligned local frame ===
        D_full = edge_cache.D_full
        D_m_prime = project_D_to_m(
            D_full=D_full,
            coeff_index_m=self.coeff_index_m,
            ebed_dim_full=self.ebed_dim_full,
            cache=edge_cache.D_to_m_cache,
            key_lmax=self.lmax,
            key_mmax=self.mmax,
        )
        src_idx = xp.astype(xp.reshape(src, (n_edge,)), xp.int64)
        x_src = xp.take(x, src_idx, axis=0)  # (E, D, C_wide)
        x_local = xp.matmul(D_m_prime, x_src)  # (E, D_m, C_wide)
        # pt rotates the *destination* node into the same edge frame for the
        # node-wise cross-mode grid product (raw, before radial modulation).
        x_dst_local: Any = None
        if self.node_wise_grid_product is not None:
            dst_idx_nw = xp.astype(xp.reshape(dst, (n_edge,)), xp.int64)
            x_dst = xp.take(x, dst_idx_nw, axis=0)  # (E, D, C_wide)
            x_dst_local = xp.matmul(D_m_prime, x_dst)  # (E, D_m, C_wide)

        # === Step 3. Select radial/type features for reduced layout ===
        degree_index_m = xp_asarray_nodetach(xp, self.degree_index_m, device=device)
        rad_feat = xp.take(radial_feat, degree_index_m, axis=1)  # (E, D_m, C)
        if self.radial_hidden_proj is not None:
            rad_feat = self.radial_hidden_proj(rad_feat)
        if self.radial_degree_mixer is None:
            x_local = x_local * rad_feat
        else:
            x_local = self.radial_degree_mixer(x_local, rad_feat)
        # pt Step 3: edge-local cross-mode grid product between the
        # radial-fused source (query) and the raw destination (context),
        # added as a residual in the reduced m-major layout (E, D_m, C_wide).
        if self.node_wise_grid_product is not None:
            x_local = x_local + self.node_wise_grid_product(x_local, x_dst_local)
        rad_feat_l0_focus = xp.reshape(
            rad_feat[:, 0, :], (n_edge, self.n_focus, self.so2_focus_dim)
        )  # (E, F, Cf)

        # === Step 4. Convert to SO(2) internal focus layout ===
        focus_gate_src: Any = None
        x_local = xp.permute_dims(
            xp.reshape(
                x_local, (n_edge, self.reduced_dim, self.n_focus, self.so2_focus_dim)
            ),
            (0, 2, 1, 3),
        )  # (E, F, D_m, Cf)
        if self.focus_compete and self.n_focus > 1:
            focus_gate_src = x_local[:, :, 0, :]

        # === Step 5. Multi-layer SO(2) mixing (pre-norm + residual) ===
        def apply_bias_correction(
            x_local: Any,
            so2_linear: SO2Linear,
            layer_idx: int,
        ) -> Any:
            if layer_idx != 0 or so2_linear.bias0 is None:
                return x_local
            bias0 = xp.reshape(
                xp_asarray_nodetach(xp, so2_linear.bias0[...], device=device),
                (1, self.n_focus, so2_linear.out_channels),
            )
            if so2_linear.out_channels == self.so2_focus_dim:
                radial_factor = rad_feat_l0_focus
            else:
                raise RuntimeError(
                    "Unexpected SO2Linear output width in bias correction"
                )
            edge_env = xp.reshape(
                xp.astype(edge_cache.edge_env, x_local.dtype), (n_edge, 1, 1)
            )
            bias_correction = bias0 * (radial_factor * edge_env - 1.0)
            x0 = x_local[:, :, :1, :] + bias_correction[:, :, None, :]
            return xp.concat([x0, x_local[:, :, 1:, :]], axis=2)

        for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
            zip(
                self.so2_linears,
                self.so2_inter_norms,
                self.non_linearities,
                strict=True,
            )
        ):
            residual = x_local
            if inter_norm is not None:
                x_local = inter_norm(x_local)
            x_local = so2_linear(x_local)
            x_local = apply_bias_correction(x_local, so2_linear, layer_idx)

            if non_linear is not None:
                x_local = non_linear(x_local)

            x_local = residual + x_local

        # === Step 6. Cross-focus softmax competition ===
        if self.focus_compete and self.n_focus > 1:
            compete_w = xp_asarray_nodetach(
                xp, self.adamw_focus_compete_w[...], device=device
            )
            gate_in = xp.astype(focus_gate_src, compete_w.dtype)
            gate_normed = self.focus_compete_norm(gate_in)  # (E, F, Cf)
            # einsum "efi,if->ef"
            focus_logits = xp.sum(
                gate_normed * xp.permute_dims(compete_w, (1, 0))[None, ...],
                axis=-1,
            )
            if self.mlp_bias:
                focus_logits = (
                    focus_logits
                    + xp_asarray_nodetach(
                        xp, self.focus_compete_bias[...], device=device
                    )[None, :]
                )
            focus_logits = focus_logits / self.focus_softmax_tau
            logits_max = xp.max(focus_logits, axis=1, keepdims=True)
            exp_logits = xp.exp(focus_logits - logits_max)
            alpha = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)
            alpha = xp.astype(alpha, x_local.dtype)
            alpha = alpha * (1.0 - self.focus_label_smoothing) + (
                self.focus_label_smoothing / float(self.n_focus)
            )
            x_local = x_local * alpha[:, :, None, None]

        # === Step 7. Rotate back to global frame ===
        Dt_full = edge_cache.Dt_full
        # Restore reduced global layout (E, D_m, C_wide) for inverse rotation.
        x_local = xp.reshape(
            xp.permute_dims(x_local, (0, 2, 1, 3)),
            (n_edge, self.reduced_dim, self.hidden_channels),
        )
        Dt_from_m = project_Dt_from_m(
            Dt_full=Dt_full,
            coeff_index_m=self.coeff_index_m,
            ebed_dim_full=self.ebed_dim_full,
            cache=edge_cache.Dt_from_m_cache,
            key_lmax=self.lmax,
            key_mmax=self.mmax,
        )
        x_message = xp.matmul(Dt_from_m, x_local)  # (E, D, C_wide)
        # Reduced layouts keep only 2*mmax+1 orders for l>mmax. Applying the
        # inverse-rotation degree rescale after the global lift restores the
        # full-basis amplitude expected by the block output contract.
        rescale = xp.astype(
            xp_asarray_nodetach(xp, self.rotate_inv_rescale_full, device=device),
            x_message.dtype,
        )
        x_message = x_message * xp.reshape(rescale, (1, -1, 1))

        # === Step 8. Aggregate with optional head-wise gating ===
        # Source Freeze Propagation Gate: broadcast the per-edge scalar
        # eta[src] to the edge message before destination aggregation.
        edge_src_gate = edge_cache.edge_src_gate
        if self.n_atten_head == 0:
            # Baseline path: envelope-weighted masked sum -> degree norm.
            edge_weight = xp.astype(edge_cache.edge_env, x_message.dtype)  # (E, 1)
            edge_weight = xp.reshape(edge_weight, (n_edge, 1))
            if edge_src_gate is not None:
                edge_weight = edge_weight * xp.astype(
                    xp.reshape(edge_src_gate, (n_edge, 1)), edge_weight.dtype
                )
            x_message = x_message * edge_weight[:, :, None]
            # pt: out.index_add_(0, dst, x_message) — padded-edge masked sum
            # over the nnei axis (dst is slot-implicit).
            x_message = x_message * mask_f[:, None, None]
            out = xp.sum(
                xp.reshape(
                    x_message,
                    (n_node, nnei, self.ebed_dim_full, self.hidden_channels),
                ),
                axis=1,
            )
            inv_sqrt_deg = xp.astype(edge_cache.inv_sqrt_deg, out.dtype)
            out = out * inv_sqrt_deg  # (N, D, C_wide)
        else:
            # === Step 8.1. Build attention logits from scalar channels ===
            qk_w = xp_asarray_nodetach(xp, self.attn_q_proj.weight[...], device=device)
            x_l0_node = xp.reshape(
                x[:, 0, :], (n_node, self.attn_n_focus, self.attn_focus_dim)
            )  # (N, Fa, Ca)
            x_l0_node = xp.astype(x_l0_node, qk_w.dtype)
            qk_input = self.attn_qk_norm(x_l0_node)
            q_node = self.attn_q_proj(qk_input)  # (N, Fa, Ca)
            k_node = self.attn_k_proj(qk_input)  # (N, Fa, Ca)
            dst_idx = xp.astype(xp.reshape(dst, (n_edge,)), xp.int64)
            q_edge = xp.reshape(
                xp.take(q_node, dst_idx, axis=0),
                (n_edge, self.attn_n_focus, self.n_atten_head, self.head_dim),
            )  # (E, Fa, H, Ch), Ca = H * Ch
            k_edge = xp.reshape(
                xp.take(k_node, src_idx, axis=0),
                (n_edge, self.attn_n_focus, self.n_atten_head, self.head_dim),
            )  # (E, Fa, H, Ch)
            radial_l0 = xp.reshape(
                rad_feat[:, 0, :], (n_edge, self.attn_n_focus, self.attn_focus_dim)
            )  # (E, Fa, Ca)
            radial_l0 = xp.astype(radial_l0, qk_w.dtype)
            # einsum "efi,ifo->efo" as a broadcast batched matmul.
            logit_w = xp.permute_dims(
                xp_asarray_nodetach(xp, self.adamw_attn_logit_w[...], device=device),
                (1, 0, 2),
            )  # (Fa, Ca, H)
            radial_bias = xp.matmul(radial_l0[:, :, None, :], logit_w[None, ...])[
                ..., 0, :
            ]  # (E, Fa, H)
            attn_logits = xp.sum(q_edge * k_edge, axis=-1) * (self.head_dim**-0.5)
            attn_logits = attn_logits + radial_bias

            # === Step 8.2. Destination-wise stable envelope-gated softmax ===
            # pt: scatter-based segment softmax keyed by dst — padded-edge
            # masked softmax over the nnei axis. ``src_weight=edge_src_gate``
            # folds SFPG into both the numerator and the denominator so a
            # muted source drops out of the normalization entirely.
            attn_alpha = segment_envelope_gated_softmax(
                logits=attn_logits,
                edge_env=xp.astype(edge_cache.edge_env, attn_logits.dtype),
                n_nodes=n_node,
                z_bias_raw=xp_asarray_nodetach(
                    xp, self.adamw_attn_z_bias_raw, device=device
                ),
                eps=self.eps,
                src_weight=(
                    None
                    if edge_src_gate is None
                    else xp.astype(edge_src_gate, attn_logits.dtype)
                ),
                edge_mask=mask_f,
            )  # (E, F, H)

            # === Step 8.3. Value projection and head-wise aggregation ===
            value_heads = xp.reshape(
                xp.astype(x_message, qk_w.dtype),
                (
                    n_edge,
                    self.ebed_dim_full,
                    self.attn_n_focus,
                    self.n_atten_head,
                    self.head_dim,
                ),
            )  # (E, D, Fa, H, Ch)
            weighted_value = value_heads * xp.reshape(
                attn_alpha, (n_edge, 1, self.attn_n_focus, self.n_atten_head, 1)
            )
            # pt: out_heads.index_add_(0, dst, weighted_value) — padded-edge
            # masked sum over the nnei axis (dst is slot-implicit).
            weighted_value = (
                weighted_value
                * xp.astype(mask_f, weighted_value.dtype)[:, None, None, None, None]
            )
            out_heads = xp.sum(
                xp.reshape(
                    weighted_value,
                    (
                        n_node,
                        nnei,
                        self.ebed_dim_full,
                        self.attn_n_focus,
                        self.n_atten_head,
                        self.head_dim,
                    ),
                ),
                axis=1,
            )  # (N, D, Fa, H, Ch)

            # === Step 8.4. Output-side head gate ===
            gate_w = xp.permute_dims(
                xp_asarray_nodetach(xp, self.adamw_attn_gate_w[...], device=device),
                (1, 0, 2),
            )  # (Fa, Ca, H)
            gate_in = self.attn_output_gate_norm(x_l0_node)
            attn_output_gate = xp_sigmoid(
                xp.matmul(gate_in[:, :, None, :], gate_w[None, ...])[..., 0, :]
            )  # (N, F, H)
            out_heads = out_heads * xp.reshape(
                attn_output_gate,
                (n_node, 1, self.attn_n_focus, self.n_atten_head, 1),
            )  # (N, D, Fa, H, Ch)

            # === Step 8.5. Merge heads ===
            out = xp.astype(
                xp.reshape(
                    out_heads, (n_node, self.ebed_dim_full, self.hidden_channels)
                ),
                x.dtype,
            )  # (N, D, C_wide)

        # === Step 9. Optional message-node grid product ===
        # pt: post-aggregation packed-layout cross-mode product between the
        # aggregated message (query) and the pre-focus-mixed node features
        # (context), added as a residual before the final channel mixing.
        if self.message_node_grid_product is not None:
            out = out + self.message_node_grid_product(out, x)

        # === Step 10. Final channel mixing ===
        out = self.post_focus_mix(out[:, :, None, :])[:, :, 0, :]
        return out  # (N, D, C)

    def _variables(self) -> dict[str, np.ndarray]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables: dict[str, np.ndarray] = {}
        for i, so2_linear in enumerate(self.so2_linears):
            for key, value in so2_linear._variables().items():
                variables[f"so2_linears.{i}.{key}"] = value
        for i, inter_norm in enumerate(self.so2_inter_norms):
            if inter_norm is not None:
                for key, value in inter_norm.serialize()["@variables"].items():
                    variables[f"so2_inter_norms.{i}.{key}"] = value
        for i, non_linear in enumerate(self.non_linearities):
            if non_linear is not None:
                for key, value in non_linear.serialize()["@variables"].items():
                    variables[f"non_linearities.{i}.{key}"] = value
        if self.n_atten_head > 0:
            variables["adamw_attn_logit_w"] = to_numpy_array(self.adamw_attn_logit_w)
            variables["adamw_attn_z_bias_raw"] = to_numpy_array(
                self.adamw_attn_z_bias_raw
            )
            variables["adamw_attn_gate_w"] = to_numpy_array(self.adamw_attn_gate_w)
            variables["attn_qk_norm.adam_scale"] = to_numpy_array(
                self.attn_qk_norm.adam_scale
            )
            variables["attn_q_proj.weight"] = to_numpy_array(self.attn_q_proj.weight)
            variables["attn_k_proj.weight"] = to_numpy_array(self.attn_k_proj.weight)
            variables["attn_output_gate_norm.adam_scale"] = to_numpy_array(
                self.attn_output_gate_norm.adam_scale
            )
        if self.focus_compete_norm is not None:
            variables["adamw_focus_compete_w"] = to_numpy_array(
                self.adamw_focus_compete_w
            )
            variables["focus_compete_norm.adam_scale"] = to_numpy_array(
                self.focus_compete_norm.adam_scale
            )
            if self.mlp_bias:
                variables["focus_compete_bias"] = to_numpy_array(
                    self.focus_compete_bias
                )
        if self.radial_hidden_proj is not None:
            variables["radial_hidden_proj.weight"] = to_numpy_array(
                self.radial_hidden_proj.weight
            )
        if self.radial_degree_mixer is not None:
            for key, value in self.radial_degree_mixer._variables().items():
                variables[f"radial_degree_mixer.{key}"] = value
        # Cross-mode grid products: nest each net's @variables under the pt
        # state_dict attribute name so ``deserialize(pt.serialize())`` matches.
        for name, grid in (
            ("node_wise_grid_product", self.node_wise_grid_product),
            ("message_node_grid_product", self.message_node_grid_product),
        ):
            if grid is not None:
                for key, value in grid.serialize()["@variables"].items():
                    variables[f"{name}.{key}"] = value
        for name, mix in (
            ("pre_focus_mix", self.pre_focus_mix),
            ("post_focus_mix", self.post_focus_mix),
        ):
            for key, value in mix.serialize()["@variables"].items():
                variables[f"{name}.{key}"] = value
        variables["coeff_index_m"] = to_numpy_array(self.coeff_index_m)
        variables["degree_index_m"] = to_numpy_array(self.degree_index_m)
        variables["rotate_inv_rescale_full"] = to_numpy_array(
            self.rotate_inv_rescale_full
        )
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        variables = dict(variables)
        prec = PRECISION_DICT[self.precision.lower()]
        cprec = PRECISION_DICT[self.compute_precision.lower()]

        def pop(key: str) -> Any:
            try:
                return variables.pop(key)
            except KeyError:
                raise KeyError(f"Missing variable: {key}") from None

        def sub_vars(prefix: str) -> dict[str, Any]:
            full = f"{prefix}."
            out = {
                key[len(full) :]: value
                for key, value in variables.items()
                if key.startswith(full)
            }
            for key in list(variables):
                if key.startswith(full):
                    del variables[key]
            if not out:
                raise KeyError(f"Missing variables with prefix: {full}")
            return out

        # Top-level index buffers: validate against the config-derived tables.
        _check_index_table(self.coeff_index_m, pop("coeff_index_m"), "coeff_index_m")
        _check_index_table(self.degree_index_m, pop("degree_index_m"), "degree_index_m")
        _check_shape_assign(
            self,
            "rotate_inv_rescale_full",
            pop("rotate_inv_rescale_full"),
            prec,
            "rotate_inv_rescale_full",
        )

        for i, so2_linear in enumerate(self.so2_linears):
            so2_linear._load_variables(sub_vars(f"so2_linears.{i}"))
        for i, inter_norm in enumerate(self.so2_inter_norms):
            if inter_norm is not None:
                sv = sub_vars(f"so2_inter_norms.{i}")
                _check_index_table(
                    inter_norm.degree_index_m,
                    sv["degree_index_m"],
                    f"so2_inter_norms.{i}.degree_index_m",
                )
                for name in ("balance_weight", "adam_scale", "bias0"):
                    _check_shape_assign(
                        inter_norm,
                        name,
                        sv[name],
                        cprec,
                        f"so2_inter_norms.{i}.{name}",
                    )
        for i, non_linear in enumerate(self.non_linearities):
            if non_linear is not None:
                sv = sub_vars(f"non_linearities.{i}")
                _check_index_table(
                    non_linear.expand_index,
                    sv["expand_index"],
                    f"non_linearities.{i}.expand_index",
                )
                _check_shape_assign(
                    non_linear.gate_linear,
                    "weight",
                    sv["gate_linear.weight"],
                    cprec,
                    f"non_linearities.{i}.gate_linear.weight",
                )
                if self.mlp_bias:
                    _check_shape_assign(
                        non_linear.gate_linear,
                        "bias",
                        sv["gate_linear.bias"],
                        cprec,
                        f"non_linearities.{i}.gate_linear.bias",
                    )
        if self.n_atten_head > 0:
            for name in (
                "adamw_attn_logit_w",
                "adamw_attn_z_bias_raw",
                "adamw_attn_gate_w",
            ):
                _check_shape_assign(self, name, pop(name), cprec, name)
            _check_shape_assign(
                self.attn_qk_norm,
                "adam_scale",
                pop("attn_qk_norm.adam_scale"),
                cprec,
                "attn_qk_norm.adam_scale",
            )
            _check_shape_assign(
                self.attn_q_proj,
                "weight",
                pop("attn_q_proj.weight"),
                cprec,
                "attn_q_proj.weight",
            )
            _check_shape_assign(
                self.attn_k_proj,
                "weight",
                pop("attn_k_proj.weight"),
                cprec,
                "attn_k_proj.weight",
            )
            _check_shape_assign(
                self.attn_output_gate_norm,
                "adam_scale",
                pop("attn_output_gate_norm.adam_scale"),
                cprec,
                "attn_output_gate_norm.adam_scale",
            )
        if self.focus_compete_norm is not None:
            _check_shape_assign(
                self,
                "adamw_focus_compete_w",
                pop("adamw_focus_compete_w"),
                cprec,
                "adamw_focus_compete_w",
            )
            _check_shape_assign(
                self.focus_compete_norm,
                "adam_scale",
                pop("focus_compete_norm.adam_scale"),
                cprec,
                "focus_compete_norm.adam_scale",
            )
            if self.mlp_bias:
                _check_shape_assign(
                    self,
                    "focus_compete_bias",
                    pop("focus_compete_bias"),
                    cprec,
                    "focus_compete_bias",
                )
        if self.radial_hidden_proj is not None:
            _check_shape_assign(
                self.radial_hidden_proj,
                "weight",
                pop("radial_hidden_proj.weight"),
                prec,
                "radial_hidden_proj.weight",
            )
        if self.radial_degree_mixer is not None:
            self.radial_degree_mixer._load_variables(sub_vars("radial_degree_mixer"))
        # Grid products have no ``_load_variables``; reuse their config (from a
        # fresh ``serialize()``) plus the loaded @variables and re-deserialize
        # in place. This exercises the full grid-net serialize round-trip.
        if self.node_wise_grid_product is not None:
            template = self.node_wise_grid_product.serialize()
            template["@variables"] = sub_vars("node_wise_grid_product")
            self.node_wise_grid_product = type(self.node_wise_grid_product).deserialize(
                template
            )
        if self.message_node_grid_product is not None:
            template = self.message_node_grid_product.serialize()
            template["@variables"] = sub_vars("message_node_grid_product")
            self.message_node_grid_product = type(
                self.message_node_grid_product
            ).deserialize(template)
        for name, mix in (
            ("pre_focus_mix", self.pre_focus_mix),
            ("post_focus_mix", self.post_focus_mix),
        ):
            sv = sub_vars(name)
            _check_index_table(
                mix.expand_index, sv["expand_index"], f"{name}.expand_index"
            )
            _check_shape_assign(mix, "weight", sv["weight"], prec, f"{name}.weight")
            if self.mlp_bias:
                _check_shape_assign(mix, "bias", sv["bias"], prec, f"{name}.bias")

        if variables:
            raise KeyError(f"Unknown variables: {sorted(variables)}")

    def serialize(self) -> dict[str, Any]:
        """Serialize the SO2Convolution to a dict (pt-compatible format)."""
        return {
            "@class": "SO2Convolution",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "kmax": self.kmax,
                "channels": self.channels,
                "n_focus": self.n_focus,
                "focus_dim": self.focus_dim,
                "focus_compete": self.focus_compete,
                "so2_norm": self.so2_norm,
                "so2_layers": self.so2_layers,
                "so2_attn_res": self.so2_attn_res_mode,
                "layer_scale": self.layer_scale,
                "n_atten_head": self.n_atten_head,
                "atten_f_mix": self.atten_f_mix,
                "atten_v_proj": self.use_atten_v_proj,
                "atten_o_proj": self.use_atten_o_proj,
                "s2_activation": self.s2_activation,
                "node_wise_grid_mlp": self.node_wise_grid_mlp,
                "node_wise_grid_branch": self.node_wise_grid_branch,
                "message_node_grid_mlp": self.message_node_grid_mlp,
                "message_node_grid_branch": self.message_node_grid_branch,
                "node_wise_s2": self.node_wise_s2,
                "node_wise_so3": self.node_wise_so3,
                "message_node_s2": self.message_node_s2,
                "message_node_so3": self.message_node_so3,
                "lebedev_quadrature": self.lebedev_quadrature,
                "activation_function": self.activation_function,
                "mlp_bias": self.mlp_bias,
                "radial_so2_mode": self.radial_so2_mode,
                "radial_so2_rank": self.radial_so2_rank,
                "eps": self.eps,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Convolution:
        """Deserialize an SO2Convolution from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Convolution":
            raise ValueError(f"Invalid class for SO2Convolution: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = dict(data.pop("config"))
        variables = data.pop("@variables")
        config["precision"] = str(config.pop("precision"))
        obj = cls(**config)
        obj._load_variables(variables)
        return obj
