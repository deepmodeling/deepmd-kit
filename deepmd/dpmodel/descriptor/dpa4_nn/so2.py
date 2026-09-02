# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(2)-equivariant message-passing layers for DPA4/SeZM.

This module defines the reduced-layout SO(2) linear operator and the
edge convolution used inside SeZM interaction blocks.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.so2``.
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
    xp_add_at,
    xp_asarray_nodetach,
    xp_einsum,
    xp_sigmoid,
)
from deepmd.dpmodel.common import (
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    Identity,
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
from .attn_res import (
    DepthAttnRes,
)
from .cartesian import (
    EdgeCartesianTensorProduct,
    NodeCartesianTensorProduct,
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
    get_promoted_dtype,
    init_trunc_normal_fan_in_out,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )

    from .edge_cache import (
        EdgeCache,
    )


class SO2Linear(NativeOP):
    """
    SO(2)-equivariant linear mixing in the edge-aligned local frame.

    Coefficient layout (m-major, truncated by mmax)
    ------------------------------------------------
    The coefficient axis D_m_trunc is ordered by |m| groups::

        [  m=0: l=0..lmax  |  m=1: (l,-1) then (l,+1)  |  ...  |  m=mmax: ... ]
         |___ lmax+1 ____|   |_______ 2*(lmax) ________|

    Each |m| group is contiguous, enabling a single block-diagonal matmul.

    Block-diagonal weight structure
    -------------------------------
    The full weight matrix W has shape ``(F, D_m_trunc*Cout, D_m_trunc*Cin)``
    and is block-diagonal over |m| groups::

        W = diag[W_m0, B_m1, B_m2, ..., B_mmax]

    - ``W_m0``: unconstrained ``(num_l*Cout, num_l*Cin)`` block for m=0.
      Cross-l mixing is allowed since m=0 coefficients are real scalars.

    - ``B_m`` (|m|>0): SO(2)-constrained 2x2 block coupling (-m, +m) pairs::

          B_m = [ W_u^T , -W_v^T ]     where W_u, W_v are learnable
                [ W_v^T ,  W_u^T ]     (num_l*Cin, num_l*Cout) each.

      This structure is the real-valued form of complex multiplication
      ``(u + iv)(a + ib) = (ua - vb) + i(va + ub)``, which guarantees
      SO(2) equivariance: rotating the input by angle phi around z
      rotates the output by the same angle.

    The weight is assembled once per forward (training) or cached (eval)
    by ``_build_so2_weight()`` in the ``(D_m*Cin, F, D_m*Cout)`` layout, then
    applied as a per-``|m|``-block batched matmul over the focus streams. The
    activation is carried in the focus-major layout ``(F, E, D_m, Cf)`` so that
    the focus stream is the batch axis of the matmul: the assembled weight is
    presented as ``(F, D_m*Cin, D_m*Cout)`` (a transient view, never a stored
    parameter) and each block contracts with no transpose of the edge axis.

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
        weight matrices; the batched matmul vectorizes over all streams.
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
        seed: int | list[int] | None,
        trainable: bool,
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
            neg_idx = np.arange(neg_start, neg_start + num_l, dtype=np.int64)
            pos_idx = np.arange(pos_start, pos_start + num_l, dtype=np.int64)
            neg_indices_list.append(neg_idx)
            pos_indices_list.append(pos_idx)
            m_ranges.append((neg_start, pos_start, num_l))
            offset += 2 * num_l

        self.reduced_dim = int(offset)

        if len(pos_indices_list) > 0:
            self.pos_indices = np.concatenate(pos_indices_list)
            self.neg_indices = np.concatenate(neg_indices_list)
            self._m_ranges = m_ranges
        else:
            self.pos_indices = np.empty(0, dtype=np.int64)
            self.neg_indices = np.empty(0, dtype=np.int64)
            self._m_ranges = []

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
        self.weight_m0 = weight_m0.astype(prec)
        if self.mlp_bias:
            self.bias0: np.ndarray | None = np.zeros(
                self.n_focus * self.out_channels, dtype=prec
            )
        else:
            self.bias0 = None

        # weight_m[i]: folded (num_l*Cin, F*2*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, 2*num_l*Cout).
        #   The factor of 2 comes from storing W_u and W_v concatenated along the
        #   output axis. _build_so2_weight() splits them and fills the 2x2 block.
        #   Scaling by 1/sqrt(2) compensates for the doubled parameter count.
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
            self.weight_m.append(weight.astype(prec))

        self.trainable = bool(trainable)

        # === Step 3. Precompute flattened slice ranges for _build_so2_weight ===
        # Each |m|>0 group occupies two sub-blocks (neg, pos) in the flattened
        # weight matrix. Pre-computing the row/col ranges avoids repeated
        # arithmetic in the hot path.
        # Tuple layout: (neg_i0, neg_i1, pos_i0, pos_i1,   <- input row ranges
        #                neg_o0, neg_o1, pos_o0, pos_o1)   <- output col ranges
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

        # The assembled SO(2) weight is block-diagonal over |m| groups; the
        # forward contracts only the diagonal blocks (see _block_diagonal_matmul).
        # Each |m| group occupies a contiguous (in, out) block on the diagonal.
        self._block_diag_slices = self._build_block_diag_slices()

    def call(self, x: Array) -> Array:
        """
        Parameters
        ----------
        x
            Input with shape (F, E, D_m_trunc, Cin), where F is the focus stream
            (the matmul batch axis), E the edge count, and D_m_trunc the
            coefficient dimension of the m-major layout truncated by `mmax`.

        Returns
        -------
        Array
            Output with shape (F, E, D_m_trunc, Cout), where Cout is output channels.
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        # === Step 1. Flatten coefficient + channel axes for the matmul ===
        # (F, E, D_m, Cin) -> (F, E, D_m*Cin); the focus stream stays the batch axis.
        n_focus, n_edge = x.shape[0], x.shape[1]
        in_dim_total = self.reduced_dim * self.in_channels
        x_flat = xp.reshape(x, (n_focus, n_edge, in_dim_total))

        # === Step 2. Get block-diagonal weight ===
        weight = self._build_so2_weight(xp, device)

        # === Step 3. Block-diagonal matmul over focus streams + reshape back ===
        out_flat = self._block_diagonal_matmul(x_flat, weight)
        out = xp.reshape(
            out_flat, (n_focus, n_edge, self.reduced_dim, self.out_channels)
        )

        # === Step 4. Bias on l=0 scalar index ===
        if self.mlp_bias:
            bias0 = xp.reshape(
                xp_asarray_nodetach(xp, self.bias0[...], device=device),
                (self.n_focus, self.out_channels),
            )
            out = xp.concat(
                [out[:, :, :1, :] + bias0[:, None, None, :], out[:, :, 1:, :]], axis=2
            )
        return out

    def _build_block_diag_slices(self) -> list[tuple[int, int, int, int]]:
        """Return the ``(in_start, in_end, out_start, out_end)`` diagonal blocks.

        One entry per ``|m|`` group in m-major order: ``m = 0`` spans
        ``lmax + 1`` coefficients and each ``|m| > 0`` spans ``2 * (lmax - m + 1)``
        coefficients (negative and positive orders).
        """
        group_sizes = [self.lmax + 1] + [
            2 * (self.lmax - m + 1) for m in range(1, self.mmax + 1)
        ]
        slices: list[tuple[int, int, int, int]] = []
        in_off = out_off = 0
        for num in group_sizes:
            in_width = num * self.in_channels
            out_width = num * self.out_channels
            slices.append((in_off, in_off + in_width, out_off, out_off + out_width))
            in_off += in_width
            out_off += out_width
        return slices

    def _build_so2_weight(self, xp: Any = None, device: Any = None) -> Array:
        """
        Assemble the per-focus block-diagonal SO(2) weight matrix.

        The flattened weight has shape ``(D_m*Cin, F, D_m*Cout)`` (in, out)
        where both axes follow the same m-major coefficient ordering.
        Off-diagonal blocks (cross-|m|) are zero, enforcing SO(2) equivariance.

        Parameters
        ----------
        xp : Any, optional
            The array namespace. Derived from ``self.weight_m0`` when either
            ``xp`` or ``device`` is ``None``.
        device : Any, optional
            The target device. Derived from ``self.weight_m0`` when either
            ``xp`` or ``device`` is ``None``.

        Returns
        -------
        Array
            Weight with shape (D_m*Cin, F, D_m*Cout).
        """
        # The ``call`` hot path passes both explicitly; the kernel value-path
        # factories invoke ``_build_so2_weight()`` with no args, so fall back to
        # the stored weight for the namespace/device (numpy or torch buffer).
        if xp is None or device is None:
            xp = array_api_compat.array_namespace(self.weight_m0)
            device = array_api_compat.device(self.weight_m0)
        in_total = self.reduced_dim * self.in_channels
        out_total = self.reduced_dim * self.out_channels
        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0 = (self.lmax + 1) * self.out_channels
        weight_m0 = xp.reshape(
            xp_asarray_nodetach(xp, self.weight_m0[...], device=device),
            (num_in_m0, self.n_focus, num_out_m0),
        )

        # m=0 block: (Cin_blk, F, Cout_blk) — (in, out) convention. The m=0 input
        # rows carry the m=0 output block followed by zero pads spanning the
        # |m|>0 output columns.
        row_blocks = [
            xp.concat(
                [
                    weight_m0,
                    xp.zeros(
                        (self._m0_in, self.n_focus, out_total - self._m0_out),
                        dtype=weight_m0.dtype,
                        device=device,
                    ),
                ],
                axis=2,
            )
        ]

        # |m|>0 blocks: fill the 2x2 SO(2) coupling structure.
        # For each |m|, the learnable param w has shape (in_blk, F, 2*out_blk)
        # which is split into W_u and W_v along the output axis.
        for m_idx, w in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0  # in_block size
            ob = no1 - no0  # out_block size
            w = xp.reshape(
                xp_asarray_nodetach(xp, w[...], device=device),
                (ib, self.n_focus, 2 * ob),
            )
            w_u = w[:, :, :ob]  # (in_blk, F, out_blk)
            w_v = w[:, :, ob:]  # (in_blk, F, out_blk)
            # Fill the 2x2 coupling:
            #   Row = input (neg/pos), Col = output (neg/pos).
            #   [ W_u^T, -W_v^T ]^T  =>  row=neg_in: W_u to neg_out, W_v to pos_out
            #   [ W_v^T,  W_u^T ]^T  =>  row=pos_in: -W_v to neg_out, W_u to pos_out
            # neg_out and pos_out are contiguous (no1 == po0); each input row band
            # is built by concatenating [left pad, two coupling sub-blocks, right pad].
            left_pad = xp.zeros((ib, self.n_focus, no0), dtype=w.dtype, device=device)
            right_pad = xp.zeros(
                (ib, self.n_focus, out_total - po1), dtype=w.dtype, device=device
            )
            neg_row = xp.concat([left_pad, w_u, w_v, right_pad], axis=2)
            pos_row = xp.concat([left_pad, -w_v, w_u, right_pad], axis=2)
            row_blocks.append(neg_row)  # neg_in -> [neg_out, pos_out]
            row_blocks.append(pos_row)  # pos_in -> [neg_out, pos_out]
        return xp.concat(row_blocks, axis=0)

    def _block_diagonal_matmul(self, x_flat: Array, weight: Array) -> Array:
        """Contract only the diagonal ``|m|`` blocks of the assembled weight.

        ``weight`` is block-diagonal over ``|m|`` (cross-``|m|`` blocks are
        exactly zero), so concatenating the per-group matmuls reproduces the
        dense contraction over the full ``(D_m*Cin, D_m*Cout)`` matrix while
        skipping the structural zeros. The result is fp32-equivalent to the
        dense path up to the matmul reduction order.

        The focus stream is the batch axis of the per-block ``bmm``: the input
        already carries it as ``(F, E, .)`` and the assembled weight is presented
        as ``(F, D_m*Cin, D_m*Cout)``, so no edge-axis transpose is needed on
        either operand and each block writes directly into the concatenated
        output. The weight view is transient (the stored parameters keep their
        assembled ``(D_m*Cin, F, D_m*Cout)`` layout).

        Parameters
        ----------
        x_flat : Array
            Flattened input with shape ``(F, E, D_m*Cin)``.
        weight : Array
            Assembled block-diagonal weight with shape ``(D_m*Cin, F, D_m*Cout)``.

        Returns
        -------
        Array
            Flattened output with shape ``(F, E, D_m*Cout)``.
        """
        xp = array_api_compat.array_namespace(x_flat)
        weight = xp.permute_dims(weight, (1, 0, 2))  # (F, D_m*Cin, D_m*Cout)
        blocks = [
            # torch.bmm(x_flat[:, :, in0:in1], weight[:, in0:in1, out0:out1]): a
            # per-focus matmul batched over the focus axis (the leading batch axis
            # of xp.matmul), contracting the input coefficient/channel index i.
            xp.matmul(
                x_flat[:, :, in0:in1],
                weight[:, in0:in1, out0:out1],
            )
            for in0, in1, out0, out1 in self._block_diag_slices
        ]
        return xp.concat(blocks, axis=-1)

    def serialize(self) -> dict[str, Any]:
        variables = {
            "m0_idx": to_numpy_array(self.m0_idx),
            "pos_indices": to_numpy_array(self.pos_indices),
            "neg_indices": to_numpy_array(self.neg_indices),
            "weight_m0": to_numpy_array(self.weight_m0),
        }
        if self.mlp_bias:
            variables["bias0"] = to_numpy_array(self.bias0)
        for i, w in enumerate(self.weight_m):
            variables[f"weight_m.{i}"] = to_numpy_array(w)
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
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Linear:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Linear":
            raise ValueError(f"Invalid class for SO2Linear: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.m0_idx = np.asarray(variables["m0_idx"], dtype=np.int64)
        obj.pos_indices = np.asarray(variables["pos_indices"], dtype=np.int64)
        obj.neg_indices = np.asarray(variables["neg_indices"], dtype=np.int64)
        obj.weight_m0 = np.asarray(variables["weight_m0"], dtype=prec)
        if obj.mlp_bias:
            obj.bias0 = np.asarray(variables["bias0"], dtype=prec)
        obj.weight_m = [
            np.asarray(variables[f"weight_m.{i}"], dtype=prec)
            for i in range(len(obj.weight_m))
        ]
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
        seed: int | list[int] | None,
        trainable: bool,
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
        self.weight = weight.astype(prec)

        if self.mode == "degree_channel" and self.rank > 0:
            channel_basis = np.empty((self.rank, self.channels), dtype=prec)
            init_trunc_normal_fan_in_out(channel_basis, child_seed(seed, 1))
            self.channel_basis: np.ndarray | None = channel_basis.astype(prec)
        else:
            self.channel_basis = None

        compact_idx, dense_idx = self._build_dense_scatter_indices()
        self.kernel_compact_index = compact_idx
        self.kernel_dense_index = dense_idx
        self.trainable = bool(trainable)

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
                    # can call bmm/einsum without transposing the degree matrix.
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
            np.array(compact_indices, dtype=np.int64),
            np.array(dense_indices, dtype=np.int64),
        )

    def _project_radial(self, radial_feat: Array) -> Array:
        xp = array_api_compat.array_namespace(radial_feat)
        device = array_api_compat.device(radial_feat)
        radial_m0 = xp.reshape(
            radial_feat[:, : self.lmax + 1, :],
            (-1, self.input_dim),
        )
        weight = xp_asarray_nodetach(xp, self.weight[...], device=device)
        return xp.matmul(radial_m0, weight)

    def _scatter_degree_kernel(self, compact: Array) -> Array:
        xp = array_api_compat.array_namespace(compact)
        device = array_api_compat.device(compact)
        n_edge = compact.shape[0]
        compact_index = xp_asarray_nodetach(
            xp, self.kernel_compact_index[...], device=device
        )
        dense_index = xp_asarray_nodetach(
            xp, self.kernel_dense_index[...], device=device
        )
        source = xp.take(compact, compact_index, axis=1)
        dense = xp.zeros(
            (self.reduced_dim * self.reduced_dim, n_edge),
            dtype=compact.dtype,
            device=device,
        )
        dense = xp_add_at(dense, dense_index, xp.permute_dims(source, (1, 0)))
        dense = xp.permute_dims(dense, (1, 0))
        return xp.reshape(dense, (n_edge, self.reduced_dim, self.reduced_dim))

    def _scatter_rank_kernel(self, compact: Array) -> Array:
        xp = array_api_compat.array_namespace(compact)
        device = array_api_compat.device(compact)
        n_edge = compact.shape[0]
        compact_index = xp_asarray_nodetach(
            xp, self.kernel_compact_index[...], device=device
        )
        dense_index = xp_asarray_nodetach(
            xp, self.kernel_dense_index[...], device=device
        )
        source = xp.take(compact, compact_index, axis=1)
        dense = xp.zeros(
            (self.reduced_dim * self.reduced_dim, n_edge, self.rank),
            dtype=compact.dtype,
            device=device,
        )
        dense = xp_add_at(dense, dense_index, xp.permute_dims(source, (1, 0, 2)))
        dense = xp.permute_dims(dense, (1, 0, 2))
        return xp.reshape(
            dense, (n_edge, self.reduced_dim, self.reduced_dim, self.rank)
        )

    def _scatter_channel_kernel(self, compact: Array) -> Array:
        xp = array_api_compat.array_namespace(compact)
        device = array_api_compat.device(compact)
        n_edge = compact.shape[0]
        compact_index = xp_asarray_nodetach(
            xp, self.kernel_compact_index[...], device=device
        )
        dense_index = xp_asarray_nodetach(
            xp, self.kernel_dense_index[...], device=device
        )
        source = xp.take(compact, compact_index, axis=1)
        dense = xp.zeros(
            (self.reduced_dim * self.reduced_dim, n_edge, self.channels),
            dtype=compact.dtype,
            device=device,
        )
        dense = xp_add_at(dense, dense_index, xp.permute_dims(source, (1, 0, 2)))
        dense = xp.permute_dims(dense, (1, 0, 2))
        return xp.reshape(
            dense, (n_edge, self.reduced_dim, self.reduced_dim, self.channels)
        )

    def call(self, x_local: Array, radial_feat: Array) -> Array:
        """
        Parameters
        ----------
        x_local
            Local reduced features with shape (E, D_m, C_wide).
        radial_feat
            Invariant radial/type features with shape (E, D_m, C_wide).
        """
        xp = array_api_compat.array_namespace(x_local)
        x_shape = x_local.shape
        radial_shape = radial_feat.shape

        def static_rank(shape: Any) -> int | None:
            rank = getattr(shape, "rank", None)
            if rank is not None:
                return int(rank)
            try:
                return len(shape)
            except (TypeError, ValueError):
                return None

        def static_dim(shape: Any, axis: int) -> int | None:
            try:
                dim = shape[axis]
            except (IndexError, TypeError, ValueError):
                return None
            dim = getattr(dim, "value", dim)
            return int(dim) if isinstance(dim, (int, np.integer)) else None

        x_rank = static_rank(x_shape)
        radial_rank = static_rank(radial_shape)
        if (x_rank is not None and x_rank != 3) or (
            radial_rank is not None and radial_rank != 3
        ):
            raise ValueError("DynamicRadialDegreeMixer inputs must have rank 3")
        if any(
            x_dim is not None and radial_dim is not None and x_dim != radial_dim
            for x_dim, radial_dim in (
                (static_dim(x_shape, axis), static_dim(radial_shape, axis))
                for axis in range(3)
            )
        ):
            raise ValueError("`x_local` and `radial_feat` must have the same shape")
        reduced_dim = static_dim(x_shape, 1)
        channel_dim = static_dim(x_shape, 2)
        if (reduced_dim is not None and reduced_dim != self.reduced_dim) or (
            channel_dim is not None and channel_dim != self.channels
        ):
            raise ValueError("Input shape is incompatible with this mixer")

        kernel_flat = self._project_radial(radial_feat)
        if self.mode == "degree":
            kernel = self._scatter_degree_kernel(kernel_flat)
            return xp.matmul(kernel, x_local)

        if self.rank > 0:
            compact = xp.reshape(kernel_flat, (-1, self.degree_kernel_size, self.rank))
            return self._mix_rank_compact(compact, x_local)

        compact = xp.reshape(kernel_flat, (-1, self.degree_kernel_size, self.channels))
        kernel = self._scatter_channel_kernel(compact)
        # einsum("eoic,eic->eoc"): contract l_in i per channel c (no channel mix).
        return xp.sum(kernel * x_local[:, None, :, :], axis=2)

    def _mix_rank_compact(self, compact: Array, x_local: Array) -> Array:
        """
        Mix the reduced features by the low-rank dynamic degree kernel.

        Parameters
        ----------
        compact : Array
            Projected per-edge degree kernels with shape
            (E, degree_kernel_size, R).
        x_local : Array
            Edge-local reduced features with shape (E, D_m, C).

        Returns
        -------
        Array
            Mixed features with shape (E, D_m, C).
        """
        xp = array_api_compat.array_namespace(compact)
        device = array_api_compat.device(compact)
        kernel = self._scatter_rank_kernel(compact)
        # einsum("eoir,eic->eorc"): contract l_in i, batched over (l_out, rank)
        # via a single matmul, then weight the rank channels by channel_basis.
        kernel_or = xp.reshape(
            xp.permute_dims(kernel, (0, 1, 3, 2)),
            (-1, self.reduced_dim * self.rank, self.reduced_dim),
        )
        mixed = xp.matmul(kernel_or, x_local)
        mixed = xp.reshape(
            mixed,
            (-1, self.reduced_dim, self.rank, self.channels),
        )
        channel_basis = xp.reshape(
            xp_asarray_nodetach(xp, self.channel_basis[...], device=device),
            (1, 1, self.rank, self.channels),
        )
        return xp.sum(mixed * channel_basis, axis=2)

    def serialize(self) -> dict[str, Any]:
        variables = {
            "weight": to_numpy_array(self.weight),
            "kernel_compact_index": to_numpy_array(self.kernel_compact_index),
            "kernel_dense_index": to_numpy_array(self.kernel_dense_index),
        }
        if self.channel_basis is not None:
            variables["channel_basis"] = to_numpy_array(self.channel_basis)
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
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> DynamicRadialDegreeMixer:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "DynamicRadialDegreeMixer":
            raise ValueError(f"Invalid class for DynamicRadialDegreeMixer: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        obj = cls(**config)
        prec = PRECISION_DICT[obj.precision.lower()]
        obj.weight = np.asarray(variables["weight"], dtype=prec)
        obj.kernel_compact_index = np.asarray(
            variables["kernel_compact_index"], dtype=np.int64
        )
        obj.kernel_dense_index = np.asarray(
            variables["kernel_dense_index"], dtype=np.int64
        )
        if obj.channel_basis is not None:
            obj.channel_basis = np.asarray(variables["channel_basis"], dtype=prec)
        return obj


def _parse_node_cartesian(spec: str) -> tuple[bool, bool, int]:
    """
    Parse the ``node_cartesian`` configuration string.

    Grammar: ``"<mode>:<layers>"`` where ``mode`` is ``"default"`` (the one-sided
    product ``Y N``) or ``"parity"`` (the symmetrized product ``Y N + N Y``), and
    ``layers`` is a non-negative integer. A bare mode defaults to one layer; a
    bare integer uses the default mode. ``"none"``, an empty string, or any zero
    layer count disables the per-node product.

    Parameters
    ----------
    spec : str
        The configuration string.

    Returns
    -------
    tuple[bool, bool, int]
        ``(enabled, symmetric, n_layers)``.

    Raises
    ------
    ValueError
        If the mode is not ``"default"`` or ``"parity"``, or the layer count is
        negative.
    """
    text = str(spec).strip().lower()
    if text in ("", "none"):
        return False, False, 0
    if ":" in text:
        mode, _, num = text.partition(":")
        mode = mode.strip() or "default"
        layers = int(num.strip())
    elif text.isdigit():
        mode, layers = "default", int(text)
    else:
        mode, layers = text, 1
    if mode not in ("default", "parity"):
        raise ValueError(
            "`node_cartesian` mode must be 'default' or 'parity', got "
            f"'{mode}' (expected '<mode>:<layers>', 'none', or a layer count)"
        )
    if layers < 0:
        raise ValueError("`node_cartesian` layer count must be non-negative")
    return layers > 0, mode == "parity", layers


class SO2Convolution(NativeOP):
    """
    SO(2)-equivariant edge convolution with cached geometry and rotations.

    This module consumes node features in packed SO(3) layout `(N, D, C)` and
    performs edge message passing in the reduced m-major local layout. The
    operation pipeline is:

    1. `pre_focus_mix`: project node features `(N, D, C)` to the SO(2) hidden width.
    2. rotate global -> local reduced basis with cached `D_to_m`.
    3. radial modulation in reduced layout.
    4. `mixing_layers` stacked local mixers:
       `inter_norm -> SO2Linear -> non_linearity -> residual(+LayerScale)`.
    5. rotate local -> global with cached `Dt_from_m`.
    6. edge aggregation (plain envelope scatter or envelope-aware grouped
       softmax attention with exact envelope-gated competition and
       output-side head gate).
    7. `post_focus_mix`: project aggregated hidden messages back to `(N, D, C)`.

    Equivariance is preserved because both `pre_focus_mix` and `post_focus_mix`
    only mix the channel axis for each `(l, m)` coefficient and never mix
    coefficient indices across `(l, m)`.

    Parameters
    ----------
    lmax
        Maximum degree.
    mmax
        Maximum SO(2) order (|m|). If None, defaults to lmax.
    kmax
        Maximum Wigner-D frame order (|k|) used by SO(3) grid branches.
    channels
        Number of channels per (l, m) coefficient.
    n_focus
        Number of focus streams inside the SO(2) branch.
    focus_dim
        Hidden width per focus stream inside SO(2).
        ``focus_dim=0`` means using ``channels``.
    focus_compete
        If True, apply cross-focus softmax competition in SO(2) local layout.
        Competition logits are constructed only from l=0 scalar channels and the
        resulting invariant weights are broadcast to all (l, m) components.
    focus_norm
        If True, RMS-normalize the competition l=0 scalars before the softmax.
        Those scalars are envelope-gated and vanish at the cutoff, so ``False``
        drops the norm to keep the competition smooth there.
    so2_norm
        If True, apply intermediate ReducedEquivariantRMSNorm as pre-norm before
        each SO(2) mixing layer. The last SO(2) layer always uses Identity.
    mixing_layers
        Number of learnable mixing layers in the per-edge message core (SO2Linear
        layers for the SO(2) path, or refinement layers for ``edge_cartesian``).
        ``0`` applies only the edge-condition modulation: the rotation-free
        per-degree radial scaling for the SO(2) path, or a single ``x @ T_e`` for
        ``edge_cartesian``.
    so2_attn_res
        Depth-wise attention residual mode across the internal SO(2) layer
        history. Must be one of ``"none"``, ``"independent"``, or
        ``"dependent"``. The same scalar weights are broadcast to the full
        reduced equivariant tensor.
    layer_scale
        If True, apply per-layer learnable LayerScale (per-focus-channel,
        init 1e-3) on each SO(2) residual branch.
    n_atten_head
        Number of attention heads used during aggregation.
        - 0: plain envelope-weighted scatter-sum.
        - >0: envelope-gated grouped softmax attention with output-side head
          gates. Attention uses ``w**2 * exp(logit)`` in the numerator and
          ``zeta + sum(w**2 * exp(logit))`` in the denominator.
    atten_f_mix
        If True, merge the internal focus streams into one attention stream
        after rotate-back. Attention heads then split the full hidden width
        ``n_focus * focus_dim`` instead of each focus stream independently.
    atten_v_proj
        If True, apply an explicit degree-aware value projection before
        attention aggregation.
    atten_o_proj
        If True, apply an explicit degree-aware output projection after the
        output-side attention gate.
    s2_activation
        If True, replace each intermediate reduced-layout gate with S2-grid
        SwiGLU. Intermediate ``SO2Linear`` layers then output ``2 * focus_dim``
        channels before the activation folds them back to ``focus_dim``.
    node_wise_grid_mlp
        If True, select the polynomial grid MLP operation for the node-wise
        source-destination grid product.
    node_wise_grid_branch
        Number of scalar-routed polynomial product branches for the node-wise
        grid product. ``0`` disables branch mixing; positive values take
        precedence over ``node_wise_grid_mlp``.
    message_node_grid_mlp
        If True, select the polynomial grid MLP operation for the message-node
        grid product.
    message_node_grid_branch
        Number of scalar-routed polynomial product branches for the
        message-node grid product. ``0`` disables branch mixing; positive
        values take precedence over ``message_node_grid_mlp``.
    node_wise_s2
        If True, add an edge-local S2 product branch between radial-fused source
        features and destination features in the same edge frame.
    node_wise_so3
        If True, use the corresponding edge-local SO(3) Wigner-D grid branch.
    message_node_s2
        If True, add a packed-layout S2 product branch between the aggregated
        hidden message and the destination node features before ``post_focus_mix``.
    message_node_so3
        If True, use the corresponding post-aggregation SO(3) Wigner-D grid
        branch.
    lebedev_quadrature
        If True, use Lebedev quadrature for the S2 projector.
    activation_function
        Activation function for the gated activation path when
        ``s2_activation=False``.
    mlp_bias
        Whether to use bias in SO2Linear (l=0 bias) and GatedActivation
        (gate linear bias).
    radial_so2_mode
        Dynamic radial degree mixer mode. ``"none"`` applies elementwise
        radial modulation, ``"degree"`` applies a channel-shared dynamic
        cross-degree kernel, and ``"degree_channel"`` applies a
        per-channel dynamic cross-degree kernel.
    radial_so2_rank
        Low-rank channel factorization rank for ``radial_so2_mode="degree_channel"``.
        ``0`` uses the full per-channel dynamic degree kernel.
    edge_cartesian
        If True, replace the rotate-to-local / ``SO2Linear`` stack / rotate-back
        core with the per-edge global-frame Cartesian rank-2 tensor product.
        Requires ``lmax`` in ``{1, 2}`` and is incompatible with the S2/SO(3)
        grid product branches. The dynamic radial degree mixer is bypassed
        because the radial edge condition is carried by the Cartesian edge tensor
        instead.
    node_cartesian
        Per-node global-frame Cartesian rank-2 tensor product applied to the
        aggregated message, coupling it with the destination node feature after
        the optional message-node grid product and before ``post_focus_mix``. The
        Cartesian analog of the message-node grid product. Configured by a string
        ``"<mode>:<layers>"`` where ``mode`` is ``"default"`` (one-sided product)
        or ``"parity"`` (symmetrized product), and ``layers`` is the stack depth;
        a bare integer ``N`` is shorthand for ``"default:N"``, and ``"none"`` (or
        ``0``) disables it. Requires ``lmax`` in ``{1, 2}`` and is orthogonal to
        ``edge_cartesian``.
    eps
        Small epsilon for normalization modules.
    precision
        Parameter precision.
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
        kmax: int = 1,
        channels: int,
        n_focus: int = 1,
        focus_dim: int = 0,
        focus_compete: bool = True,
        focus_norm: bool = True,
        so2_norm: bool = False,
        mixing_layers: int = 4,
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
        edge_cartesian: bool = False,
        node_cartesian: str | int = "none",
        eps: float = 1e-7,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None,
        trainable: bool,
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
        self.focus_norm = bool(focus_norm)
        self.focus_softmax_tau = 1.0
        self.focus_label_smoothing = 0.02
        self.so2_norm = bool(so2_norm)
        self.mixing_layers = int(mixing_layers)
        if self.mixing_layers < 0:
            raise ValueError("`mixing_layers` must be >= 0")
        self.so2_attn_res_mode = str(so2_attn_res).lower()
        if self.so2_attn_res_mode not in ATTN_RES_MODES:
            raise ValueError(
                "`so2_attn_res` must be one of 'none', 'independent', or 'dependent'"
            )
        self.use_so2_attn_res = self.so2_attn_res_mode != "none"
        self.layer_scale = bool(layer_scale)
        self.n_atten_head = int(n_atten_head)
        self.atten_f_mix = bool(atten_f_mix)
        self.use_atten_v_proj = bool(atten_v_proj)
        self.use_atten_o_proj = bool(atten_o_proj)
        self.s2_activation = bool(s2_activation)
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
        self.s2_full_grid_resolution = (
            [max(base_full_grid_resolution), max(base_full_grid_resolution)]
            if self.s2_grid_method == "e3nn"
            else base_full_grid_resolution
        )
        self.activation_function = str(activation_function)
        if self.n_atten_head < 0:
            raise ValueError("`n_atten_head` must be non-negative")
        self.attn_n_focus = (
            1 if self.atten_f_mix and self.n_atten_head > 0 else self.n_focus
        )
        self.attn_focus_dim = (
            self.hidden_channels
            if self.atten_f_mix and self.n_atten_head > 0
            else self.so2_focus_dim
        )
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
        self.edge_cartesian = bool(edge_cartesian)
        self.node_cartesian = str(node_cartesian)
        (
            self._node_cartesian_enabled,
            self._node_cartesian_symmetric,
            self._node_cartesian_layers,
        ) = _parse_node_cartesian(self.node_cartesian)
        if self.edge_cartesian:
            if self.lmax not in (1, 2):
                raise ValueError("`edge_cartesian` requires lmax in {1, 2}")
            if (
                self.node_wise_s2
                or self.node_wise_so3
                or self.message_node_s2
                or self.message_node_so3
            ):
                raise ValueError(
                    "`edge_cartesian` is incompatible with the S2/SO(3) grid "
                    "product branches"
                )
        if self._node_cartesian_enabled and self.lmax not in (1, 2):
            raise ValueError("`node_cartesian` requires lmax in {1, 2}")
        self.eps = float(eps)
        self.ebed_dim_full = get_so3_dim_of_lmax(self.lmax)
        self.precision = precision
        self.compute_precision = np.dtype(
            get_promoted_dtype(PRECISION_DICT[precision])
        ).name

        # === Step 1. Split deterministic seeds at the module top-level ===
        seed_so2_stack = child_seed(seed, 0)
        seed_non_linearities = child_seed(seed, 1)
        seed_so3_pre = child_seed(seed, 2)
        seed_so3_post = child_seed(seed, 3)
        seed_gate = child_seed(seed, 4)
        seed_depth_attn = child_seed(seed, 5)
        seed_radial_hidden = child_seed(seed, 6)
        seed_radial_degree = child_seed(seed, 7)
        seed_node_wise_s2 = child_seed(seed, 8)
        seed_message_node_s2 = child_seed(seed, 9)
        seed_node_cartesian = child_seed(seed, 10)

        # === Step 2. Edge mixing core: SO(2) rotation stack or Cartesian product ===
        if self.edge_cartesian:
            self.edge_cartesian_tp = EdgeCartesianTensorProduct(
                lmax=self.lmax,
                focus_dim=self.so2_focus_dim,
                n_focus=self.n_focus,
                n_layers=self.mixing_layers,
                activation_function=self.activation_function,
                mlp_bias=self.mlp_bias,
                eps=self.eps,
                precision=self.precision,
                seed=seed_so2_stack,
                trainable=trainable,
            )
        else:
            self._build_so2_mixing(
                seed_so2_stack=seed_so2_stack,
                seed_non_linearities=seed_non_linearities,
                seed_depth_attn=seed_depth_attn,
                trainable=trainable,
            )

        # === Step 2b. Optional per-node Cartesian mixing on the aggregated message ===
        self.node_cartesian_tp: NodeCartesianTensorProduct | None = None
        if self._node_cartesian_enabled:
            self.node_cartesian_tp = NodeCartesianTensorProduct(
                lmax=self.lmax,
                focus_dim=self.so2_focus_dim,
                n_focus=self.n_focus,
                n_layers=self._node_cartesian_layers,
                symmetric=self._node_cartesian_symmetric,
                activation_function=self.activation_function,
                mlp_bias=self.mlp_bias,
                precision=self.precision,
                seed=seed_node_cartesian,
                trainable=trainable,
            )

        # === Step 7. Optional attention projections (n_atten_head > 0) ===
        self.attn_qk_norm: ScalarRMSNorm | None = None
        self.attn_q_proj: FocusLinear | None = None
        self.attn_k_proj: FocusLinear | None = None
        self.attn_focus_mix: SO3Linear | None = None
        self.attn_v_proj: SO3Linear | None = None
        self.attn_o_proj: SO3Linear | None = None
        self.adamw_attn_logit_w: np.ndarray | None = None
        self.adamw_attn_z_bias_raw: np.ndarray | None = None
        self.attn_output_gate_norm: ScalarRMSNorm | None = None
        self.adamw_attn_gate_w: np.ndarray | None = None
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
            if self.atten_f_mix:
                self.attn_focus_mix = SO3Linear(
                    lmax=self.lmax,
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    n_focus=1,
                    precision=self.compute_precision,
                    mlp_bias=False,
                    seed=child_seed(seed_gate, 19),
                    trainable=trainable,
                )
            if self.use_atten_v_proj:
                self.attn_v_proj = SO3Linear(
                    lmax=self.lmax,
                    in_channels=self.attn_focus_dim,
                    out_channels=self.attn_focus_dim,
                    n_focus=self.attn_n_focus,
                    precision=self.compute_precision,
                    mlp_bias=False,
                    seed=child_seed(seed_gate, 20),
                    trainable=trainable,
                )
            if self.use_atten_o_proj:
                self.attn_o_proj = SO3Linear(
                    lmax=self.lmax,
                    in_channels=self.attn_focus_dim,
                    out_channels=self.attn_focus_dim,
                    n_focus=self.attn_n_focus,
                    precision=self.compute_precision,
                    mlp_bias=False,
                    seed=child_seed(seed_gate, 21),
                    trainable=trainable,
                )
            self.adamw_attn_logit_w = (
                np.random.default_rng(child_seed(seed_gate, 2))
                .normal(
                    0.0,
                    0.01,
                    size=(self.attn_focus_dim, self.attn_n_focus, self.n_atten_head),
                )
                .astype(PRECISION_DICT[self.compute_precision])
            )
            # softplus(0.5413) ~= 1.0 provides balanced initial competition.
            self.adamw_attn_z_bias_raw = np.full(
                (self.attn_n_focus, self.n_atten_head),
                0.5413,
                dtype=PRECISION_DICT[self.compute_precision],
            )
            self.attn_output_gate_norm = ScalarRMSNorm(
                channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                eps=self.eps,
                precision=self.compute_precision,
                trainable=trainable,
            )
            self.adamw_attn_gate_w = (
                np.random.default_rng(child_seed(seed_gate, 3))
                .normal(
                    0.0,
                    0.01,
                    size=(self.attn_focus_dim, self.attn_n_focus, self.n_atten_head),
                )
                .astype(PRECISION_DICT[self.compute_precision])
            )

        # === Step 7.5. Optional cross-focus competition ===
        self.focus_compete_norm: ScalarRMSNorm | None = None
        self.adamw_focus_compete_w: np.ndarray | None = None
        self.focus_compete_bias: np.ndarray | None = None
        if self.focus_compete and self.n_focus > 1:
            # ``focus_norm=False`` drops the competition-input RMSNorm (which
            # would cross its eps floor as the envelope-gated scalars vanish at
            # rcut); the competition weights then decay smoothly to uniform.
            if self.focus_norm:
                self.focus_compete_norm = ScalarRMSNorm(
                    channels=self.so2_focus_dim,
                    n_focus=self.n_focus,
                    eps=self.eps,
                    precision=self.compute_precision,
                    trainable=trainable,
                )
            self.adamw_focus_compete_w = (
                np.random.default_rng(child_seed(seed_gate, 4))
                .normal(
                    0.0,
                    0.01,
                    size=(self.so2_focus_dim, self.n_focus),
                )
                .astype(PRECISION_DICT[self.compute_precision])
            )
            if self.mlp_bias:
                self.focus_compete_bias = np.zeros(
                    self.n_focus,
                    dtype=PRECISION_DICT[self.compute_precision],
                )

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
        if not self.edge_cartesian and self.radial_so2_mode != "none":
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
            precision=precision,
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
            precision=precision,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_post,
            init_std=0.0,
        )

        # === Step 11. Edge-frame requirement for the SO(2) message ===
        self.needs_local_frame = (not self.edge_cartesian) and (
            self.mixing_layers > 0
            or self.radial_so2_mode != "none"
            or self.node_wise_grid_product is not None
        )

        # === Step 12. Optional fused flash-attention aggregation kernel ===
        # Folds the entire ``n_atten_head > 0`` value aggregation -- block-diagonal
        # rotate-back, inverse-rotation rescale, envelope-gated softmax weighting,
        # and the destination scatter -- into a single destination-segmented
        # kernel, removing the transient ``x_message`` and weighted-value edge
        # tensors and the ``index_add`` round trip; the op itself dispatches to an
        # eager reference off the CUDA fp32 path. The output-side head gate stays
        # a cheap node-level elementwise applied after the kernel.
        #
        # Layout support is a property of the block, so it is expressed
        # independently of the backend: the kernel only serves the ``mmax == 1``
        # attention layout without the optional focus-mix / value / output
        # projections (the deployed DPA4 configuration). Whichever of the
        # mutually exclusive inference gates is active then supplies the
        # implementation, and ``self._flash_atten_fn`` being bound is what marks
        # the fused path as live. The array-API reference leaves the hook unbound.
        self._flash_atten_layout_ok = (
            self.n_atten_head > 0
            and self.mmax == 1
            and self.needs_local_frame
            and not self.edge_cartesian
            and not self.atten_f_mix
            and self.attn_v_proj is None
            and self.attn_o_proj is None
            and self.attn_focus_mix is None
        )
        self._flash_atten_fn = None
        self._cuda_conv_fn = None
        self._cached_edge_csr_fn = None

        # === Step 13. Optional fused SO(2) value-path seam ===
        # The fused value path folds the rotate-to-local projection, radial
        # mixing, and the full SO(2) mixing stack into a single kernel, emitting
        # the pre-rotate-back per-focus local features directly. The pure
        # array-API reference has no such kernel, so it never runs the fused
        # value path: every hook stays ``None`` and ``so2_message`` takes the
        # dense branch. The ``pt_expt`` backend binds the selected implementation.
        self._triton_value_path = None
        self._cutile_value_path = None

        # === Step 14. Optional fused training seams ===
        # Training differentiates the convolution twice under a force loss, so
        # its accelerated forms carry analytic backward and second-order
        # implementations of their own: one fused operator for the value stream
        # up to the attention aggregation (``_cuda_value_train``), and the
        # segmented attention softmax / flash aggregation pair for the
        # attention span (``_flash_atten_trains`` marks the bound aggregation
        # as training-capable). The array-API reference leaves every hook
        # unbound and trains through the dense expression.
        self._cuda_value_train = None
        self._flash_atten_trains = False
        self.trainable = bool(trainable)

    def call(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
    ) -> Array:
        """
        Parameters
        ----------
        x
            Node features with shape (N, D, C), where D=(lmax+1)^2 is the
            SO(3) coefficient dimension.
        edge_cache
            Precomputed edge cache. Must be compatible with this block's lmax.
        radial_feat
            Per-edge radial features with shape (E, lmax+1, C), already fused
            with edge type features.

        Returns
        -------
        Array
            Message updates with shape (N, D, C).
        """
        # === Step 1. Pre-focus channel mixing on full width ===
        # (N, D, C_wide), C_wide = F * Cf
        x = self.pre_focus_mix(x[:, :, None, :])[:, :, 0, :]

        # === Step 2. Node update from the edge messages ===
        if self.n_atten_head == 0:
            out = self.forward_envelope(x, edge_cache, radial_feat)
        else:
            out = self.forward_attention(x, edge_cache, radial_feat)
        # (N, D, C_wide)

        # === Step 3. Optional message-node grid product ===
        if self.message_node_grid_product is not None:
            out = out + self.message_node_grid_product(out, x)

        # === Step 4. Optional per-node Cartesian tensor-product mixing ===
        # Couples the aggregated message with the destination node feature ``x``,
        # the Cartesian analog of the message-node grid product.
        if self.node_cartesian_tp is not None:
            out = self.node_cartesian_tp(out, x)

        # === Step 5. Final channel mixing ===
        out = self.post_focus_mix(out[:, :, None, :])[:, :, 0, :]
        return out  # (N, D, C)

    def forward_envelope(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
    ) -> Array:
        """
        Reduce the edge messages with the scalar envelope weight.

        The attention-free path: an envelope-weighted scatter add followed by the
        degree normalization. Folding the Source Freeze Propagation Gate into the
        envelope keeps the operation count unchanged; ``edge_src_gate`` is ``None``
        outside bridging mode, where the branch disappears.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        Array
            Node update with shape (N, D, C_wide).
        """
        xp = array_api_compat.array_namespace(x)

        # === Step 1. Edge message in the global frame ===
        x_message, _ = self.edge_message(x, edge_cache, radial_feat)
        # (E, D, C_wide)

        # === Step 2. Envelope weighting, with the source gate folded in ===
        edge_weight = edge_cache.edge_env  # (E, 1)
        edge_src_gate = edge_cache.edge_src_gate
        if edge_src_gate is not None:
            edge_weight = edge_weight * xp.astype(edge_src_gate, edge_weight.dtype)
        x_message = x_message * edge_weight[..., None]  # (E, D, C_wide)

        # === Step 3. Destination reduction and degree normalization ===
        compute_dtype = get_xp_precision(xp, self.compute_precision)
        out = xp.zeros(
            x.shape,
            dtype=compute_dtype,
            device=array_api_compat.device(x),
        )
        out = xp_add_at(out, edge_cache.dst, xp.astype(x_message, compute_dtype))
        out = out * xp.astype(edge_cache.inv_sqrt_deg, compute_dtype)
        return xp.astype(out, get_xp_precision(xp, self.precision))

    def forward_attention(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
    ) -> Array:
        """
        Reduce the edge messages with head-wise attention weights.

        Dispatches to one of three backends that share the same contract and
        differ only in how much of the per-edge span their operator absorbs:
        the fused CUDA convolution spans everything from the attention logits to
        the gated aggregate, the fused flash aggregation spans the rotate-back
        and the weighted reduction, and the dense reference materializes every
        stage.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        Array
            Node update with shape (N, D, C_wide).
        """
        # === Step 1. Scalar channels shared by every attention component ===
        xp = array_api_compat.array_namespace(x)
        x_l0_node = xp.reshape(
            x[:, 0, :], (x.shape[0], self.attn_n_focus, self.attn_focus_dim)
        )  # (N, Fa, Ca)

        # === Step 2. Backend dispatch ===
        # The fused CUDA operator computes the attention weights itself, so it
        # does not serve the bridging mode, whose source gate reshapes the
        # softmax normalization.
        training = getattr(self, "training", False)
        run_cuda = (
            self._cuda_conv_fn is not None
            and not training
            and edge_cache.edge_src_gate is None
        )
        run_flash = (
            self._flash_atten_fn is not None
            and (not training or self._flash_atten_trains)
            and not run_cuda
        )
        if run_cuda:
            return self.forward_attention_cuda(x, edge_cache, radial_feat, x_l0_node)
        if run_flash:
            return self.forward_attention_flash(x, edge_cache, radial_feat, x_l0_node)
        return self.forward_attention_dense(x, edge_cache, radial_feat, x_l0_node)

    def forward_attention_cuda(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
        x_l0_node: Array,
    ) -> Array:
        """
        Evaluate the whole per-edge span with the fused CUDA convolution.

        One operator covers the attention logits and their envelope-gated
        segment softmax, the rotation into the edge frame, the radial degree
        mixer, the gated mixing stack, the inverse rotation, the attention
        weighting, the destination reduction and the output-side head gate, so
        neither a per-edge activation nor the ungated node aggregate reaches
        device memory. Only the node-level projections are built here.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).
        x_l0_node : Array
            Node scalar channels with shape (N, Fa, Ca).

        Returns
        -------
        Array
            Node update with shape (N, D, C_wide).
        """
        xp = array_api_compat.array_namespace(x)

        # === Step 1. Projected radial features ===
        rad_feat = self._cuda_conv_fn.radial_features(
            radial_feat
        )  # (E, lmax+1, C_wide)

        # === Step 2. Attention query and key projections ===
        q_node, k_node = self.attention_qk(x_l0_node)  # (N, Fa, Ca) each

        # === Step 3. Output-side head gate ===
        head_gate = self.attention_head_gate(x_l0_node)  # (N, Fa, H)

        # === Step 4. Fused convolution ===
        out = self._cuda_conv_fn(x, edge_cache, rad_feat, q_node, k_node, head_gate)
        return xp.astype(out, get_xp_precision(xp, self.precision))

    def forward_attention_flash(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
        x_l0_node: Array,
    ) -> Array:
        """
        Evaluate the attention path with the fused flash aggregation.

        The SO(2) message stays in the local frame, and one destination-segmented
        kernel folds the block-diagonal rotate-back, the inverse-rotation
        rescale, the per-edge weighting and the destination reduction into a
        single atomic-free pass, so the transient rotate-back message and
        weighted value tensors are never materialized.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).
        x_l0_node : Array
            Node scalar channels with shape (N, Fa, Ca).

        Returns
        -------
        Array
            Node update with shape (N, D, C_wide).
        """
        xp = array_api_compat.array_namespace(x)

        # === Step 1. Local-frame edge message ===
        x_local, rad_feat = self.so2_message(
            x, edge_cache, radial_feat, return_local=True
        )  # (E, F, D_m, Cf), (E, lmax+1, C_wide)

        # === Step 2. Attention weights ===
        attn_alpha = self.attention_weights(
            x_l0_node, edge_cache, rad_feat
        )  # (E, F, H)

        # === Step 3. Output-side head gate ===
        head_gate = self.attention_head_gate(x_l0_node)  # (N, Fa, H)

        # === Step 4. Fused rotate-back and weighted destination reduction ===
        # The destination CSR view is built once per step and shared by every
        # segment consumer of the graph.
        if self._cached_edge_csr_fn is None:
            raise RuntimeError("The fused attention path requires a CSR builder")
        dst = edge_cache.dst
        order, row_ptr = self._cached_edge_csr_fn(edge_cache, "dst", x.shape[0])
        rotation = edge_cache.Dt_full
        if rotation is None:
            rotation = self._cuda_value_train.edge_runs(edge_cache)
        pre_gate = self._flash_atten_fn(
            x_local,
            rotation,
            self.rotate_inv_rescale_full,
            attn_alpha,
            order,
            row_ptr,
            dst,
            self.lmax,
            self.n_atten_head,
        )  # (N, D, C_wide)

        # === Step 5. Output-side head gate, node-level elementwise ===
        gate_full = self.broadcast_head_gate(head_gate)  # (N, C_wide)
        out = pre_gate * gate_full[:, None, :]
        return xp.astype(out, get_xp_precision(xp, self.precision))

    def forward_attention_dense(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
        x_l0_node: Array,
    ) -> Array:
        """
        Evaluate the attention path with dense head-wise aggregation.

        The reference backend: it materializes the per-head weighted value and
        carries the optional value and output projections, which the fused
        backends do not support.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).
        x_l0_node : Array
            Node scalar channels with shape (N, Fa, Ca).

        Returns
        -------
        Array
            Node update with shape (N, D, C_wide).
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        dst = edge_cache.dst
        n_node = x.shape[0]
        compute_dtype = get_xp_precision(xp, self.compute_precision)

        # === Step 1. Global-frame edge message ===
        x_message, rad_feat = self.edge_message(
            x, edge_cache, radial_feat
        )  # (E, D, C_wide), (E, lmax+1, C_wide)
        n_edge = x_message.shape[0]

        # === Step 2. Attention weights ===
        attn_alpha = self.attention_weights(
            x_l0_node, edge_cache, rad_feat
        )  # (E, F, H)

        # === Step 3. Output-side head gate ===
        head_gate = self.attention_head_gate(x_l0_node)  # (N, Fa, H)

        # === Step 4. Value projection ===
        value_focus = xp.astype(
            xp.reshape(
                x_message,
                (n_edge, self.ebed_dim_full, self.attn_n_focus, self.attn_focus_dim),
            ),
            compute_dtype,
        )  # (E, D, Fa, Ca)
        if self.attn_v_proj is not None:
            value_focus = self.attn_v_proj(value_focus)

        # === Step 5. Head-wise weighting and destination reduction ===
        value_heads = xp.reshape(
            value_focus,
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
        )  # (E, D, Fa, H, Ch)
        out_heads = xp.zeros(
            (
                n_node,
                self.ebed_dim_full,
                self.attn_n_focus,
                self.n_atten_head,
                self.head_dim,
            ),
            dtype=compute_dtype,
            device=device,
        )  # (N, D, Fa, H, Ch)
        out_heads = xp_add_at(out_heads, dst, weighted_value)

        # === Step 6. Output-side head gate ===
        out_heads = out_heads * xp.reshape(
            head_gate,
            (n_node, 1, self.attn_n_focus, self.n_atten_head, 1),
        )  # (N, D, Fa, H, Ch)

        # === Step 7. Output projection and head merge ===
        out_focus = xp.reshape(
            out_heads,
            (n_node, self.ebed_dim_full, self.attn_n_focus, self.attn_focus_dim),
        )  # (N, D, Fa, Ca)
        if self.attn_o_proj is not None:
            out_focus = self.attn_o_proj(out_focus)
        out = xp.reshape(out_focus, (n_node, self.ebed_dim_full, self.hidden_channels))
        return xp.astype(out, get_xp_precision(xp, self.precision))

    def attention_qk(self, x_l0_node: Array) -> tuple[Array, Array]:
        """
        Project the normalized scalar channels into queries and keys.

        Parameters
        ----------
        x_l0_node : Array
            Node scalar channels with shape (N, Fa, Ca).

        Returns
        -------
        tuple[Array, Array]
            ``(q_node, k_node)``, each with shape (N, Fa, Ca).
        """
        xp = array_api_compat.array_namespace(x_l0_node)
        qk_input = self.attn_qk_norm(
            xp.astype(x_l0_node, get_xp_precision(xp, self.compute_precision))
        )
        return self.attn_q_proj(qk_input), self.attn_k_proj(qk_input)

    def attention_weights(
        self,
        x_l0_node: Array,
        edge_cache: EdgeCache,
        rad_feat: Array,
    ) -> Array:
        """
        Build envelope-gated attention weights from the scalar channels.

        The softmax takes ``src_weight`` so that the Source Freeze Propagation
        Gate enters both the numerator and the denominator. A muted source
        (``eta_src = 0``) then drops out of the destination's normalization
        entirely, which the frozen-zone invariance requires: post-multiplying the
        weights alone would still leak the muted source through the shared
        denominator.

        Parameters
        ----------
        x_l0_node : Array
            Node scalar channels with shape (N, Fa, Ca).
        edge_cache : EdgeCache
            Precomputed edge cache.
        rad_feat : Array
            Projected radial features with shape (E, lmax+1, C_wide).

        Returns
        -------
        Array
            Attention weights with shape (E, F, H).
        """
        xp = array_api_compat.array_namespace(x_l0_node)
        device = array_api_compat.device(x_l0_node)
        src, dst = edge_cache.src, edge_cache.dst
        n_edge = src.shape[0]
        compute_dtype = get_xp_precision(xp, self.compute_precision)

        # === Step 1. Query-key logits on the edges ===
        q_node, k_node = self.attention_qk(x_l0_node)  # (N, Fa, Ca) each
        q_edge = xp.reshape(
            xp.take(q_node, dst, axis=0),
            (n_edge, self.attn_n_focus, self.n_atten_head, self.head_dim),
        )  # (E, Fa, H, Ch), Ca = H * Ch
        k_edge = xp.reshape(
            xp.take(k_node, src, axis=0),
            (n_edge, self.attn_n_focus, self.n_atten_head, self.head_dim),
        )  # (E, Fa, H, Ch)
        attn_logits = xp.sum(q_edge * k_edge, axis=-1) * (
            self.head_dim**-0.5
        )  # (E, F, H)

        # === Step 2. Radial logit bias ===
        radial_l0 = xp.reshape(
            rad_feat[:, 0, :], (n_edge, self.attn_n_focus, self.attn_focus_dim)
        )  # (E, Fa, Ca)
        radial_bias = xp_einsum(
            "efi,ifo->efo",
            xp.astype(radial_l0, compute_dtype),
            xp_asarray_nodetach(xp, self.adamw_attn_logit_w[...], device=device),
        )  # (E, F, H)
        attn_logits = attn_logits + radial_bias

        # === Step 3. Envelope-gated segment softmax with a null mass ===
        return self._attention_softmax(
            attn_logits, edge_cache, x_l0_node.shape[0]
        )  # (E, F, H)

    def _attention_softmax(
        self,
        attn_logits: Array,
        edge_cache: EdgeCache,
        n_nodes: int,
    ) -> Array:
        """
        Normalize the attention logits over each destination segment.

        The dense reference below materializes the scatter/gather chain of
        the envelope-gated softmax; accelerated backends override this seam
        with a CSR-segmented operator whose backward and second order stay
        in-kernel under a force loss.

        Parameters
        ----------
        attn_logits : Array
            Attention logits with shape (E, F, H).
        edge_cache : EdgeCache
            Precomputed edge cache.
        n_nodes : int
            Number of destination nodes.

        Returns
        -------
        Array
            Attention weights with shape (E, F, H).
        """
        xp = array_api_compat.array_namespace(attn_logits)
        device = array_api_compat.device(attn_logits)
        compute_dtype = get_xp_precision(xp, self.compute_precision)
        edge_src_gate = edge_cache.edge_src_gate
        return segment_envelope_gated_softmax(
            logits=attn_logits,
            edge_env=xp.astype(edge_cache.edge_env, compute_dtype),
            dst=edge_cache.dst,
            n_nodes=n_nodes,
            z_bias_raw=xp_asarray_nodetach(
                xp, self.adamw_attn_z_bias_raw[...], device=device
            ),
            eps=self.eps,
            src_weight=(
                None
                if edge_src_gate is None
                else xp.astype(edge_src_gate, compute_dtype)
            ),
            edge_mask=edge_cache.edge_mask,
        )

    def attention_head_gate(self, x_l0_node: Array) -> Array:
        """
        Build the output-side head gate from the scalar channels.

        Parameters
        ----------
        x_l0_node : Array
            Node scalar channels with shape (N, Fa, Ca).

        Returns
        -------
        Array
            One gate per node, focus stream and head, with shape (N, Fa, H).
        """
        xp = array_api_compat.array_namespace(x_l0_node)
        device = array_api_compat.device(x_l0_node)
        compute_dtype = get_xp_precision(xp, self.compute_precision)
        normalized = self.attn_output_gate_norm(xp.astype(x_l0_node, compute_dtype))
        return xp_sigmoid(
            xp_einsum(
                "nfi,ifo->nfo",
                normalized,
                xp_asarray_nodetach(xp, self.adamw_attn_gate_w[...], device=device),
            )
        )

    def broadcast_head_gate(self, head_gate: Array) -> Array:
        """
        Spread a per-head gate over the channels of its head.

        Parameters
        ----------
        head_gate : Array
            Gate with shape (N, Fa, H).

        Returns
        -------
        Array
            Gate with shape (N, C_wide), laid out as the packed hidden width
            ``c = f * Cf + h * head_dim + ch``.
        """
        xp = array_api_compat.array_namespace(head_gate)
        n_node = head_gate.shape[0]
        gate = xp.reshape(head_gate, (n_node, self.attn_n_focus, self.n_atten_head, 1))
        gate = xp.broadcast_to(
            gate,
            (n_node, self.attn_n_focus, self.n_atten_head, self.head_dim),
        )
        return xp.reshape(gate, (n_node, self.hidden_channels))

    def edge_message(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
    ) -> tuple[Array, Array]:
        """
        Build the edge message in the global frame.

        Dispatches to the Cartesian product, the SO(2) mixing stack, or the
        rotation-free radial message when no local-frame operation is needed.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        tuple[Array, Array]
            ``(x_message, rad_feat)`` with shapes (E, D, C_wide) and
            (E, lmax+1, C_wide).
        """
        # === Step 1. Message construction ===
        if self.edge_cartesian:
            x_message, rad_feat = self.cartesian_message(x, edge_cache, radial_feat)
        elif self.needs_local_frame:
            x_message, rad_feat = self.so2_message(x, edge_cache, radial_feat)
        else:
            x_message, rad_feat = self.radial_message(x, edge_cache, radial_feat)

        # === Step 2. Optional focus mixing for the attention stream ===
        if self.attn_focus_mix is not None:
            x_message = self.attn_focus_mix(x_message[:, :, None, :])[:, :, 0, :]
        return x_message, rad_feat

    def radial_message(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
    ) -> tuple[Array, Array]:
        """
        Build edge messages by rotation-free per-degree radial scaling.

        Used when no local-frame operation is required (``mixing_layers == 0``,
        ``radial_so2_mode == "none"``, and no node-wise grid product). Per-degree
        scalar radial scaling commutes with rotation, so the edge-aligned frame
        is unnecessary and the message reduces to a source gather, an elementwise
        per-degree scale, and the optional cross-focus competition.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        tuple[Array, Array]
            ``(x_message, rad_feat)`` with shapes (E, D, C_wide) and
            (E, lmax+1, C_wide). The ``l=0`` slice of ``rad_feat`` is consumed by
            the attention aggregation.
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        src = edge_cache.src
        n_edge = src.shape[0]

        rad_feat = radial_feat  # (E, lmax+1, C)
        if self.radial_hidden_proj is not None:
            rad_feat = self.radial_hidden_proj(rad_feat)  # (E, lmax+1, C_wide)

        # Broadcast each degree's radial weight over its 2l+1 orders and scale the
        # gathered source feature in the global frame.
        x_src = xp.take(x, src, axis=0)  # (E, D, C_wide)
        rad_packed = xp.take(
            rad_feat,
            xp_asarray_nodetach(xp, self.degree_index_full[...], device=device),
            axis=1,
        )  # (E, D, C_wide)
        x_message = x_src * rad_packed

        # === Cross-focus softmax competition ===
        # Gate on the radial-fused source l=0 scalar, matching the SO(2) path.
        if self.focus_compete and self.n_focus > 1:
            focus_gate_src = xp.reshape(
                x_src[:, 0, :] * rad_feat[:, 0, :],
                (n_edge, self.n_focus, self.so2_focus_dim),
            )  # (E, F, Cf)
            alpha = self._focus_alpha(focus_gate_src)
            x_message = xp.reshape(
                xp.reshape(
                    x_message,
                    (n_edge, self.ebed_dim_full, self.n_focus, self.so2_focus_dim),
                )
                * xp.astype(alpha, x_message.dtype)[:, None, :, None],
                (n_edge, self.ebed_dim_full, self.hidden_channels),
            )
        return x_message, rad_feat

    def so2_message(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
        return_local: bool = False,
    ) -> tuple[Array, Array]:
        """
        Build edge messages by rotate-to-local, SO(2) mixing, and rotate-back.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).
        return_local : bool
            If True, return the pre-rotate-back per-focus local features
            ``(E, F, D_m, Cf)`` instead of the rotated-back message. Used by the
            fused flash-attention aggregation, which folds the rotate-back into
            its own kernel.

        Returns
        -------
        tuple[Array, Array]
            ``(x_message, rad_feat)`` with shapes (E, D, C_wide) and
            (E, D_m, C_wide) by default, or ``(x_local, rad_feat)`` with
            ``x_local`` of shape (E, F, D_m, Cf) when ``return_local`` is True.
            The ``l=0`` slice of ``rad_feat`` is consumed by the attention
            aggregation.
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        src = edge_cache.src
        n_edge = src.shape[0]

        training = getattr(self, "training", False)
        if self._cutile_value_path is not None and not training:
            # === Steps 1-5 (fused cuTile operators). ``rotate_mix`` folds the
            # rotation and the radial degree mixing into one edge-parallel
            # kernel writing the focus-major layout; ``mixing_stack`` runs the
            # whole gated stack, keeping the inter-layer activations and the
            # gated-layer pre-activations off the traced graph entirely. ===
            x_local, rad_feat = self._cutile_value_path(x, edge_cache, radial_feat)
        elif self._cuda_value_train is not None and training:
            # === Steps 1-5 (one CUDA kernel, training). The whole value
            # stream up to the attention aggregation runs in a single launch
            # with analytic backward and second order; only the backward
            # anchors reach device memory. ===
            if self._cached_edge_csr_fn is not None:
                self._cached_edge_csr_fn(edge_cache, "src", x.shape[0])
            x_local, rad_feat = self._cuda_value_train(x, edge_cache, radial_feat)
        elif self._triton_value_path is not None and not training:
            # === Steps 1-5 (fused Triton operators). ``so2_rotate_mix`` folds
            # the rotation and the radial degree mixing into one edge-parallel
            # kernel writing the focus-major layout; ``so2_mixing_stack`` runs
            # the whole gated stack with the competition weight fused into its
            # final store, keeping the inter-layer activations off the traced
            # graph. The rotate-mix backward reduces through the source CSR
            # view, which is built once per step and kept on the edge cache. ===
            if self._cached_edge_csr_fn is not None:
                self._cached_edge_csr_fn(edge_cache, "src", x.shape[0])
            x_local, rad_feat = self._triton_value_path(x, edge_cache, radial_feat)
        else:
            # === Steps 1-3. Rotation, radial mixing and the focus-major cast ===
            x_local, rad_feat = self._rotate_mix(x, edge_cache, radial_feat)

            # The scalar slices the mixing stack needs are shared by every
            # rotate-mix implementation, so they are derived here rather than
            # inside the seam.
            rad_feat_l0_focus = xp.reshape(
                rad_feat[:, 0, :], (n_edge, self.n_focus, self.so2_focus_dim)
            )  # (E, F, Cf)
            focus_gate_src: Array | None = None
            if self.focus_compete and self.n_focus > 1:
                focus_gate_src = x_local[:, :, 0, :]  # (F, E, Cf)

            # === Step 4. Multi-layer SO(2) mixing (pre-norm + residual) ===

            def so2_l0_extractor(v: Array) -> Array:
                """Extract scalar features from the edge-major layout (E, F, D_m, Cf)."""
                return xp.reshape(v[:, :, 0, :], (v.shape[0], self.hidden_channels))

            def apply_bias_correction(
                x_local: Array,
                so2_linear: SO2Linear,
                layer_idx: int,
            ) -> Array:
                if layer_idx != 0 or so2_linear.bias0 is None:
                    return x_local
                if so2_linear.out_channels == self.so2_focus_dim:
                    radial_factor = rad_feat_l0_focus
                elif so2_linear.out_channels == 2 * self.so2_focus_dim:
                    radial_factor = xp.concat(
                        [rad_feat_l0_focus, rad_feat_l0_focus], axis=-1
                    )
                else:
                    raise RuntimeError(
                        "Unexpected SO2Linear output width in bias correction"
                    )
                # Focus-major broadcast: bias0 (F, Cout), the radial l=0 factor
                # (E, F, .) transposed to (F, E, .), the per-edge envelope over the
                # edge axis, applied to the l=0 scalar slice (F, E, Cout).
                bias0 = xp.reshape(
                    xp_asarray_nodetach(xp, so2_linear.bias0[...], device=device),
                    (self.n_focus, so2_linear.out_channels),
                )
                radial_factor = xp.permute_dims(radial_factor, (1, 0, 2))  # (F, E, .)
                bias_correction = bias0[:, None, :] * (
                    radial_factor * xp.reshape(edge_cache.edge_env, (1, -1, 1)) - 1.0
                )
                x_local = xp.concat(
                    [
                        x_local[:, :, :1, :] + bias_correction[:, :, None, :],
                        x_local[:, :, 1:, :],
                    ],
                    axis=2,
                )
                return x_local

            if self.use_so2_attn_res:
                # The depth-attention residual is a per-edge reduction over the
                # layer history (``DepthAttnRes`` batches on axis 0), so the history
                # is kept in the edge-major orientation and each mixing step
                # transposes into the focus-major layout for the linear.
                so2_depth_sources = [
                    xp.permute_dims(x_local, (1, 0, 2, 3)),  # (E, F, D_m, Cf)
                ]
                for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                    zip(
                        self.so2_linears,
                        self.so2_inter_norms,
                        self.non_linearities,
                        strict=True,
                    )
                ):
                    x_edge: Array = self.so2_layer_attn_res[layer_idx](
                        sources=so2_depth_sources,
                        scalar_extractor=so2_l0_extractor,
                        current_x=xp.permute_dims(x_local, (1, 0, 2, 3)),
                    )
                    x_local = xp.permute_dims(x_edge, (1, 0, 2, 3))  # (F, E, D_m, Cf)
                    residual = x_local
                    x_local = inter_norm(x_local)
                    x_local = so2_linear(x_local)
                    x_local = apply_bias_correction(x_local, so2_linear, layer_idx)

                    x_local = non_linear(x_local)

                    if self.layer_scale:
                        scale: Array = xp.reshape(
                            xp_asarray_nodetach(
                                xp,
                                self.adam_so2_layer_scales[layer_idx][...],
                                device=device,
                            ),
                            (self.n_focus, 1, 1, self.so2_focus_dim),
                        )
                        x_local = residual + scale * x_local
                    else:
                        x_local = residual + x_local
                    so2_depth_sources.append(
                        xp.permute_dims(x_local - residual, (1, 0, 2, 3))
                    )
            else:
                for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                    zip(
                        self.so2_linears,
                        self.so2_inter_norms,
                        self.non_linearities,
                        strict=True,
                    )
                ):
                    residual = x_local
                    x_local = inter_norm(x_local)
                    x_local = so2_linear(x_local)
                    x_local = apply_bias_correction(x_local, so2_linear, layer_idx)

                    x_local = non_linear(x_local)

                    if self.layer_scale:
                        scale = xp.reshape(
                            xp_asarray_nodetach(
                                xp,
                                self.adam_so2_layer_scales[layer_idx][...],
                                device=device,
                            ),
                            (self.n_focus, 1, 1, self.so2_focus_dim),
                        )
                        x_local = residual + scale * x_local
                    else:
                        x_local = residual + x_local

            # === Step 5. Cross-focus softmax competition ===
            if self.focus_compete and self.n_focus > 1:
                # ``_focus_alpha`` is shared with the rotation-free radial and Cartesian
                # messages in the edge-major (E, F) orientation; feed it the transposed
                # view of the focus-major scalar and broadcast the weights back over the
                # focus-major activation.
                alpha = self._focus_alpha(
                    xp.permute_dims(focus_gate_src, (1, 0, 2))
                )  # (E, F)
                x_local = (
                    x_local
                    * xp.astype(xp.permute_dims(alpha, (1, 0)), x_local.dtype)[
                        ..., None, None
                    ]
                )

            # === Exit. Restore the (E, F, D_m, Cf) orientation ===
            # Both the fused flash-attention aggregation kernel and the rotate-back
            # consume this orientation through explicit strides, so the focus-major
            # buffer is handed back as a view with no copy.
            x_local = xp.permute_dims(
                x_local, (1, 0, 2, 3)
            )  # (E, F, D_m, Cf), strided view

        # The fused flash-attention aggregation consumes the per-focus
        # ``(E, F, D_m, Cf)`` local layout directly and performs the rotate-back
        # inside its kernel, so return before the standalone rotate-back.
        if return_local:
            return x_local, rad_feat

        # === Step 6. Rotate back to global frame ===
        x_message = self._rotate_back(x_local, edge_cache, n_edge)
        # Reduced layouts keep only 2*mmax+1 orders for l>mmax. Applying the
        # inverse-rotation degree rescale after the global lift restores the
        # full-basis amplitude expected by the block output contract.
        x_message = x_message * xp.reshape(
            xp_asarray_nodetach(xp, self.rotate_inv_rescale_full[...], device=device),
            (1, -1, 1),
        )
        return x_message, rad_feat

    def _rotate_to_local(
        self, x: Array, edge_cache: EdgeCache
    ) -> tuple[Array, Array | None]:
        """
        Rotate node features into the edge-aligned reduced local frame.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.

        Returns
        -------
        tuple[Array, Array | None]
            ``(x_local, x_dst_local)`` with shapes (E, D_m, C_wide). The
            destination-node projection is built only for the node-wise grid
            product and is ``None`` otherwise.
        """
        xp = array_api_compat.array_namespace(x)
        D_full = edge_cache.D_full
        D_m_prime = project_D_to_m(
            D_full=D_full,
            coeff_index_m=self.coeff_index_m,
            ebed_dim_full=self.ebed_dim_full,
            cache=edge_cache.D_to_m_cache,
            key_lmax=self.lmax,
            key_mmax=self.mmax,
        )
        x_src = xp.take(x, edge_cache.src, axis=0)  # (E, D, C_wide)
        x_local = xp.matmul(D_m_prime, x_src)  # (E, D_m, C_wide)
        x_dst_local: Array | None = None
        if self.node_wise_grid_product is not None:
            x_dst = xp.take(x, edge_cache.dst, axis=0)  # (E, D, C_wide)
            x_dst_local = xp.matmul(D_m_prime, x_dst)  # (E, D_m, C_wide)
        return x_local, x_dst_local

    def _rotate_mix(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
    ) -> tuple[Array, Array]:
        """
        Rotate the source features and apply the radial degree mixing.

        This is the entry stage of the SO(2) message: the gathered source
        features are rotated into the edge frame, multiplied (or mixed) by the
        projected radial features, and cast to the focus-major layout the
        mixing stack consumes. Accelerated backends override this seam with a
        single edge-parallel kernel that writes the focus-major layout
        directly, so the degree-expanded global intermediate and its relayout
        never reach device memory.

        The mixing stack runs with the focus stream on the batch axis, the
        native layout of the block-diagonal batched matmul: the per-focus
        linear consumes it with no edge-axis transpose and writes each ``|m|``
        block with no reassembly cost. The returned array is a strided view of
        the reduced global buffer, materialized by the first linear's reshape
        exactly as any reduced-layout view would be.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        tuple[Array, Array]
            The mixing-stack input with shape (F, E, D_m, Cf) and the
            projected radial features with shape (E, D_m, C_wide).
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        n_edge = edge_cache.src.shape[0]

        # === Step 1. Rotate to edge-aligned local frame ===
        x_local, x_dst_local = self._rotate_to_local(x, edge_cache)

        # === Step 2. Select radial/type features for reduced layout ===
        rad_feat = xp.take(
            radial_feat,
            xp_asarray_nodetach(xp, self.degree_index_m[...], device=device),
            axis=1,
        )  # (E, D_m, C)
        if self.radial_hidden_proj is not None:
            rad_feat = self.radial_hidden_proj(rad_feat)
        if self.radial_degree_mixer is None:
            x_local = x_local * rad_feat
        else:
            x_local = self.radial_degree_mixer(x_local, rad_feat)
        if self.node_wise_grid_product is not None:
            x_local = x_local + self.node_wise_grid_product(x_local, x_dst_local)

        # === Step 3. Cast to the focus-major SO(2) mixing layout ===
        x_local = xp.permute_dims(
            xp.reshape(
                x_local,
                (n_edge, self.reduced_dim, self.n_focus, self.so2_focus_dim),
            ),
            (2, 0, 1, 3),
        )  # (F, E, D_m, Cf), strided view
        return x_local, rad_feat

    def _rotate_back(self, x_local: Array, edge_cache: EdgeCache, n_edge: int) -> Array:
        """
        Rotate the SO(2) focus-layout features back to the global frame.

        Parameters
        ----------
        x_local : Array
            Local features with shape (E, F, D_m, Cf) in the SO(2) focus layout
            produced by the SO(2) mixing layers.
        edge_cache : EdgeCache
            Precomputed edge cache.
        n_edge : int
            Number of edges E.

        Returns
        -------
        Array
            Global-frame message with shape (E, D, C_wide), before the
            inverse-rotation degree rescale.
        """
        xp = array_api_compat.array_namespace(x_local)
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
        return xp.matmul(Dt_from_m, x_local)  # (E, D, C_wide)

    def cartesian_message(
        self,
        x: Array,
        edge_cache: EdgeCache,
        radial_feat: Array,
    ) -> tuple[Array, Array]:
        """
        Build edge messages via the global-frame Cartesian rank-2 tensor product.

        Parameters
        ----------
        x : Array
            Node features with shape (N, D, C_wide) after pre-focus mixing.
        edge_cache : EdgeCache
            Precomputed edge cache.
        radial_feat : Array
            Per-edge radial features with shape (E, lmax+1, C).

        Returns
        -------
        tuple[Array, Array]
            ``(x_message, rad_feat)`` with shapes (E, D, C_wide) and
            (E, lmax+1, C_wide). The ``l=0`` slice of ``rad_feat`` is consumed by
            the attention aggregation.
        """
        xp = array_api_compat.array_namespace(x)
        src = edge_cache.src
        n_edge = src.shape[0]

        # === Step 1. Per-degree radial weights projected to the hidden width ===
        rad_feat = radial_feat  # (E, lmax+1, C)
        if self.radial_hidden_proj is not None:
            rad_feat = self.radial_hidden_proj(rad_feat)  # (E, lmax+1, C_wide)

        # === Step 2. Global-frame Cartesian tensor product ===
        x_src = xp.take(x, src, axis=0)  # (E, D, C_wide)
        x_message = self.edge_cartesian_tp(
            x_src, edge_cache.edge_vec, rad_feat
        )  # (E, D, C_wide)

        # === Step 3. Cross-focus softmax competition ===
        # Gate on the radial-fused source l=0 scalar, matching the SO(2) path,
        # whose competition reads the pre-mixing input (its l=0 equals the
        # rotation-invariant source l=0 times the l=0 radial weight).
        if self.focus_compete and self.n_focus > 1:
            focus_gate_src = xp.reshape(
                x_src[:, 0, :] * rad_feat[:, 0, :],
                (n_edge, self.n_focus, self.so2_focus_dim),
            )  # (E, F, Cf)
            alpha = self._focus_alpha(focus_gate_src)
            x_message = xp.reshape(
                xp.reshape(
                    x_message,
                    (n_edge, self.ebed_dim_full, self.n_focus, self.so2_focus_dim),
                )
                * xp.astype(alpha, x_message.dtype)[:, None, :, None],
                (n_edge, self.ebed_dim_full, self.hidden_channels),
            )
        return x_message, rad_feat

    def _focus_alpha(self, focus_gate_src: Array) -> Array:
        """
        Compute per-focus softmax competition weights from l=0 scalars.

        Parameters
        ----------
        focus_gate_src : Array
            Per-edge l=0 scalar features with shape (E, F, Cf).

        Returns
        -------
        Array
            Label-smoothed competition weights with shape (E, F), in the compute
            dtype.
        """
        xp = array_api_compat.array_namespace(focus_gate_src)
        device = array_api_compat.device(focus_gate_src)
        focus_in = xp.astype(
            focus_gate_src, get_xp_precision(xp, self.compute_precision)
        )
        if self.focus_norm:
            focus_in = self.focus_compete_norm(focus_in)
        focus_logits = xp.sum(
            focus_in
            * xp.permute_dims(
                xp_asarray_nodetach(xp, self.adamw_focus_compete_w[...], device=device),
                (1, 0),
            )[None, :, :],
            axis=2,
        )
        if self.mlp_bias:
            focus_logits = (
                focus_logits
                + xp_asarray_nodetach(xp, self.focus_compete_bias[...], device=device)[
                    None, :
                ]
            )
        focus_logits = focus_logits / self.focus_softmax_tau
        alpha = xp.exp(focus_logits - xp.max(focus_logits, axis=1, keepdims=True))
        alpha = alpha / xp.sum(alpha, axis=1, keepdims=True)
        return alpha * (1.0 - self.focus_label_smoothing) + (
            self.focus_label_smoothing / float(self.n_focus)
        )

    def _build_so2_mixing(
        self,
        *,
        seed_so2_stack: int | list[int] | None,
        seed_non_linearities: int | list[int] | None,
        seed_depth_attn: int | list[int] | None,
        trainable: bool,
    ) -> None:
        """
        Build the SO(2) rotation-frame mixing stack.

        Populates the m-major reduced-layout buffers, the multi-layer
        ``SO2Linear`` stack, its intermediate norms and nonlinearities, the
        optional depth-wise attention residuals, and the optional per-layer
        LayerScale. These are the SO(2)-only tensors; they are skipped entirely
        when ``edge_cartesian`` is True.

        Parameters
        ----------
        seed_so2_stack : int | list[int] | None
            Seed for the ``SO2Linear`` layers.
        seed_non_linearities : int | list[int] | None
            Seed for the intermediate nonlinearities.
        seed_depth_attn : int | list[int] | None
            Seed for the depth-wise attention residuals.
        trainable : bool
            Whether parameters are trainable.
        """
        # === Step 1. Precompute coefficient indices for m-major reduced layout ===
        coeff_index_m = build_m_major_index(self.lmax, self.mmax)
        degree_index_m = build_m_major_l_index(self.lmax, self.mmax)
        degree_index_full = map_degree_idx(self.lmax)
        rotate_inv_rescale_full = build_rotate_inv_rescale(
            lmax=self.lmax,
            mmax=self.mmax,
            degree_index=degree_index_full,
        ).astype(PRECISION_DICT[self.compute_precision])
        self.coeff_index_m = coeff_index_m
        self.degree_index_m = degree_index_m
        # Packed (l, m) -> l index, used by the rotation-free radial message to
        # broadcast each degree's radial weight over its orders.
        self.degree_index_full = degree_index_full
        self.rotate_inv_rescale_full = rotate_inv_rescale_full
        self.reduced_dim = int(coeff_index_m.size)

        # === Step 3. Multiple SO2Linear layers ===
        self.so2_linears = [
            SO2Linear(
                lmax=self.lmax,
                mmax=self.mmax,
                in_channels=self.so2_focus_dim,
                out_channels=(
                    2 * self.so2_focus_dim
                    if self.s2_activation and i < self.mixing_layers - 1
                    else self.so2_focus_dim
                ),
                n_focus=self.n_focus,
                precision=self.precision,
                mlp_bias=self.mlp_bias,
                seed=child_seed(seed_so2_stack, i),
                trainable=trainable,
            )
            for i in range(self.mixing_layers)
        ]

        # === Step 4. Intermediate norms (the last layer always uses Identity) ===
        inter_norms: list[NativeOP] = []
        for i in range(self.mixing_layers):
            if self.so2_norm and i < self.mixing_layers - 1:
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
                inter_norms.append(Identity())
        self.so2_inter_norms = inter_norms

        # === Step 5. Intermediate non-linearity (the last layer stays linear) ===
        # Both branches run inside the focus-major SO(2) mixing layout, so they are
        # built with ``layout="fndc"``: the ``S2GridNet`` activation (an S2 or
        # SO(3) grid GLU per its grid configuration) folds the focus-major
        # re-orientation into its coefficient/grid transpose, and the coefficient
        # ``GatedActivation`` projects its per-focus gate in the same layout.
        non_linearities: list[NativeOP] = []
        for i in range(self.mixing_layers):
            if i >= self.mixing_layers - 1:
                non_linearities.append(Identity())
            elif self.s2_activation:
                non_linearities.append(
                    S2GridNet(
                        lmax=self.lmax,
                        mmax=self.mmax,
                        channels=self.so2_focus_dim,
                        n_focus=self.n_focus,
                        mode="self",
                        op_type="glu",
                        precision=self.compute_precision,
                        layout="fndc",
                        grid_resolution_list=self.s2_grid_resolution,
                        coefficient_layout="m_major",
                        grid_method=self.s2_grid_method,
                        mlp_bias=self.mlp_bias,
                        trainable=trainable,
                        seed=child_seed(seed_non_linearities, i),
                    )
                )
            else:
                non_linearities.append(
                    GatedActivation(
                        lmax=self.lmax,
                        mmax=self.mmax,
                        channels=self.so2_focus_dim,
                        n_focus=self.n_focus,
                        precision=self.compute_precision,
                        activation_function=self.activation_function,
                        mlp_bias=self.mlp_bias,
                        layout="fndc",
                        trainable=trainable,
                        seed=child_seed(seed_non_linearities, i),
                    )
                )
        self.non_linearities = non_linearities

        # === Step 6. Optional depth-wise attention residuals across SO(2) layers ===
        if self.use_so2_attn_res:
            self.so2_layer_attn_res: list[DepthAttnRes] | None = [
                DepthAttnRes(
                    channels=self.hidden_channels,
                    input_dependent=self.so2_attn_res_mode == "dependent",
                    eps=self.eps,
                    bias=self.mlp_bias,
                    precision=self.compute_precision,
                    trainable=trainable,
                    seed=child_seed(seed_depth_attn, i),
                )
                for i in range(self.mixing_layers)
            ]
        else:
            self.so2_layer_attn_res = None

        # === Step 7. Optional per-layer LayerScale for SO(2) residual branches ===
        if self.layer_scale:
            self.adam_so2_layer_scales = [
                np.ones(
                    (self.n_focus, self.so2_focus_dim),
                    dtype=PRECISION_DICT[self.precision.lower()],
                )
                * 1e-3
                for _ in range(self.mixing_layers)
            ]
        else:
            self.adam_so2_layer_scales = None

    def _sub_modules(self) -> list[tuple[str, NativeOP]]:
        """Single equivariant sub-modules keyed by their pt ``state_dict`` prefixes."""
        modules: list[tuple[str, NativeOP]] = []
        if self.edge_cartesian:
            modules.append(("edge_cartesian_tp", self.edge_cartesian_tp))
        if self.node_cartesian_tp is not None:
            modules.append(("node_cartesian_tp", self.node_cartesian_tp))
        if self.attn_qk_norm is not None:
            modules.append(("attn_qk_norm", self.attn_qk_norm))
            modules.append(("attn_q_proj", self.attn_q_proj))
            modules.append(("attn_k_proj", self.attn_k_proj))
            if self.attn_focus_mix is not None:
                modules.append(("attn_focus_mix", self.attn_focus_mix))
            if self.attn_v_proj is not None:
                modules.append(("attn_v_proj", self.attn_v_proj))
            if self.attn_o_proj is not None:
                modules.append(("attn_o_proj", self.attn_o_proj))
            modules.append(("attn_output_gate_norm", self.attn_output_gate_norm))
        if self.focus_compete_norm is not None:
            modules.append(("focus_compete_norm", self.focus_compete_norm))
        if self.radial_hidden_proj is not None:
            modules.append(("radial_hidden_proj", self.radial_hidden_proj))
        if self.radial_degree_mixer is not None:
            modules.append(("radial_degree_mixer", self.radial_degree_mixer))
        if self.node_wise_grid_product is not None:
            modules.append(("node_wise_grid_product", self.node_wise_grid_product))
        if self.message_node_grid_product is not None:
            modules.append(
                ("message_node_grid_product", self.message_node_grid_product)
            )
        modules.append(("pre_focus_mix", self.pre_focus_mix))
        modules.append(("post_focus_mix", self.post_focus_mix))
        return modules

    def _variables(self) -> dict[str, Any]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables: dict[str, Any] = {}
        # === Single equivariant sub-modules ===
        for prefix, sub in self._sub_modules():
            for key, value in sub.serialize().get("@variables", {}).items():
                variables[f"{prefix}.{key}"] = value
        # === SO(2) mixing stack (absent under the Cartesian edge core) ===
        if not self.edge_cartesian:
            for attr in (
                "so2_linears",
                "so2_inter_norms",
                "non_linearities",
                "so2_layer_attn_res",
            ):
                sub_list = getattr(self, attr)
                if sub_list is None:
                    continue
                for i, sub in enumerate(sub_list):
                    for key, value in sub.serialize().get("@variables", {}).items():
                        variables[f"{attr}.{i}.{key}"] = value
            if self.adam_so2_layer_scales is not None:
                for i, value in enumerate(self.adam_so2_layer_scales):
                    variables[f"adam_so2_layer_scales.{i}"] = to_numpy_array(value)
        # === Raw attention and cross-focus competition parameters ===
        for name in (
            "adamw_attn_logit_w",
            "adamw_attn_z_bias_raw",
            "adamw_attn_gate_w",
            "adamw_focus_compete_w",
            "focus_compete_bias",
        ):
            value = getattr(self, name)
            if value is not None:
                variables[name] = to_numpy_array(value)
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        compute_prec = PRECISION_DICT[self.compute_precision]
        prec = PRECISION_DICT[self.precision.lower()]
        # === Single equivariant sub-modules ===
        for attr, sub in self._sub_modules():
            prefix = f"{attr}."
            sub_variables = {
                key[len(prefix) :]: value
                for key, value in variables.items()
                if key.startswith(prefix)
            }
            data = sub.serialize()
            data["@variables"] = sub_variables
            setattr(self, attr, type(sub).deserialize(data))
        # === SO(2) mixing stack (absent under the Cartesian edge core) ===
        if not self.edge_cartesian:
            for attr in (
                "so2_linears",
                "so2_inter_norms",
                "non_linearities",
                "so2_layer_attn_res",
            ):
                sub_list = getattr(self, attr)
                if sub_list is None:
                    continue
                for i, sub in enumerate(sub_list):
                    prefix = f"{attr}.{i}."
                    sub_variables = {
                        key[len(prefix) :]: value
                        for key, value in variables.items()
                        if key.startswith(prefix)
                    }
                    data = sub.serialize()
                    data["@variables"] = sub_variables
                    sub_list[i] = type(sub).deserialize(data)
            if self.adam_so2_layer_scales is not None:
                # Rebuild the per-layer scales locally and assign the list once.
                # Under pt_expt ``adam_so2_layer_scales`` is a ParameterList whose
                # elements reject direct numpy assignment; reassigning the whole
                # list lets the backend rebuild the container cleanly while the
                # numeric values stay identical.
                self.adam_so2_layer_scales = [
                    np.asarray(variables[f"adam_so2_layer_scales.{i}"], dtype=prec)
                    for i in range(len(self.adam_so2_layer_scales))
                ]
        # === Raw attention and cross-focus competition parameters ===
        for name in (
            "adamw_attn_logit_w",
            "adamw_attn_z_bias_raw",
            "adamw_attn_gate_w",
            "adamw_focus_compete_w",
            "focus_compete_bias",
        ):
            if name in variables:
                setattr(self, name, np.asarray(variables[name], dtype=compute_prec))

    def serialize(self) -> dict[str, Any]:
        """Serialize the SO2Convolution to a dict."""
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
                "focus_norm": self.focus_norm,
                "so2_norm": self.so2_norm,
                "mixing_layers": self.mixing_layers,
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
                "edge_cartesian": self.edge_cartesian,
                "node_cartesian": self.node_cartesian,
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
