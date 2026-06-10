# SPDX-License-Identifier: LGPL-3.0-or-later
"""
SO(2)-equivariant message-passing layers for SeZM.

This module defines the reduced-layout SO(2) linear operator and the
edge convolution used inside SeZM interaction blocks.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    get_generator,
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
    np_safe,
    nvtx_range,
    safe_numpy_to_tensor,
    use_triton_infer,
)

if TYPE_CHECKING:
    from .edge_cache import (
        EdgeFeatureCache,
    )


class SO2Linear(nn.Module):
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
    by ``_build_so2_weight()``, then applied via a single batched matmul
    over all focus streams: ``einsum("efi,foi->efo")``.

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
    dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        mlp_bias: bool = False,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(self.lmax if mmax is None else mmax)
        if self.mmax < 0:
            raise ValueError("`mmax` must be non-negative")
        if self.mmax > self.lmax:
            raise ValueError("`mmax` must be <= `lmax`")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.mlp_bias = bool(mlp_bias)

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
        self.register_buffer(
            "m0_idx",
            torch.arange(m0_size, device=self.device, dtype=torch.long),
            persistent=True,
        )

        pos_indices_list: list[torch.Tensor] = []
        neg_indices_list: list[torch.Tensor] = []
        # Each entry: (neg_start, pos_start, num_l) for a fixed |m|.
        # These ranges are contiguous in m-major layout.
        m_ranges: list[tuple[int, int, int]] = []

        offset = m0_size
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            neg_start = offset
            pos_start = offset + num_l
            neg_idx = torch.arange(
                neg_start, neg_start + num_l, device=self.device, dtype=torch.long
            )
            pos_idx = torch.arange(
                pos_start, pos_start + num_l, device=self.device, dtype=torch.long
            )
            neg_indices_list.append(neg_idx)
            pos_indices_list.append(pos_idx)
            m_ranges.append((neg_start, pos_start, num_l))
            offset += 2 * num_l

        self.reduced_dim = int(offset)

        if len(pos_indices_list) > 0:
            self.register_buffer(
                "pos_indices", torch.cat(pos_indices_list), persistent=True
            )
            self.register_buffer(
                "neg_indices", torch.cat(neg_indices_list), persistent=True
            )
            self._m_ranges = m_ranges
        else:
            self.register_buffer(
                "pos_indices",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "neg_indices",
                torch.empty(0, device=self.device, dtype=torch.long),
                persistent=True,
            )
            self._m_ranges = []

        # === Step 2. Learnable weight parameters ===
        # weight_m0: folded (num_l*Cin, F*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, num_l*Cout).
        #   Cross-l mixing is allowed because m=0 coefficients are real.
        num_m0 = self.lmax + 1
        num_in_m0 = num_m0 * self.in_channels
        num_out_m0 = num_m0 * self.out_channels
        self.weight_m0 = nn.Parameter(
            torch.empty(
                num_in_m0,
                self.n_focus * num_out_m0,
                device=self.device,
                dtype=self.dtype,
            )
        )
        weight_m0_view = self.weight_m0.view(num_in_m0, self.n_focus, num_out_m0)
        for focus_idx in range(self.n_focus):
            init_trunc_normal_fan_in_out(
                weight_m0_view[:, focus_idx, :], child_seed(seed, 1000 + focus_idx)
            )
        if self.mlp_bias:
            self.bias0: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    self.n_focus * self.out_channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        else:
            self.bias0 = None

        # weight_m[i]: folded (num_l*Cin, F*2*num_l*Cout) storage — (in, out) convention.
        #   Runtime view: (num_l*Cin, F, 2*num_l*Cout).
        #   The factor of 2 comes from storing W_u and W_v concatenated along the
        #   output axis. _build_so2_weight() splits them and fills the 2x2 block.
        #   Scaling by 1/sqrt(2) compensates for the doubled parameter count.
        self.weight_m: nn.ParameterList = nn.ParameterList()
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            num_in = num_l * self.in_channels
            num_out = 2 * num_l * self.out_channels
            weight = nn.Parameter(
                torch.empty(
                    num_in,
                    self.n_focus * num_out,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            weight_view = weight.view(num_in, self.n_focus, num_out)
            for focus_idx in range(self.n_focus):
                init_trunc_normal_fan_in_out(
                    weight_view[:, focus_idx, :],
                    child_seed(seed, 2000 + m * 100 + focus_idx),
                )
            # Apply scaling for SO(2) equivariance
            weight.data.mul_(1.0 / math.sqrt(2.0))
            self.weight_m.append(weight)

        for p in self.parameters():
            p.requires_grad = trainable

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

        # Weight cache: only used in eval + no_grad (inference).
        # Invalidated on train() via overridden method below.
        self._cached_weight: torch.Tensor | None = None

        # Export override for the block-diagonal vs dense matmul branch below.
        # ``None`` keeps the runtime ``x_flat.is_cuda`` dispatch; the freeze sets
        # it so the AOTI graph follows the *target* device, not the CPU trace.
        self._force_block_diag_matmul: bool | None = None

        # The assembled SO(2) weight is block-diagonal over |m| groups; the
        # forward contracts only the diagonal blocks (see _block_diagonal_matmul).
        # Each |m| group occupies a contiguous (in, out) block on the diagonal.
        self._block_diag_slices = self._build_block_diag_slices()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input with shape (E, F, D_m_trunc, Cin), where D_m_trunc is the
            coefficient dimension of the m-major layout truncated by `mmax`.

        Returns
        -------
        torch.Tensor
            Output with shape (E, F, D_m_trunc, Cout), where Cout is output channels.
        """
        # === Step 1. Flatten coefficient + channel axes for matmul ===
        # (E, F, D_m, Cin) -> (E, F, D_m*Cin)
        n_edge = x.shape[0]
        in_dim_total = self.reduced_dim * self.in_channels
        x_flat = x.reshape(n_edge, self.n_focus, in_dim_total)

        # === Step 2. Get block-diagonal weight (cached in eval+no_grad) ===
        if self._cached_weight is not None:
            weight = self._cached_weight
        else:
            weight = self._build_so2_weight()
            # Cache only in eval mode with grad disabled (pure inference).
            if not self.training and not torch.is_grad_enabled():
                self._cached_weight = weight.detach()

        # === Step 3. Block-diagonal matmul over focus streams + reshape back ===
        # The dense einsum is a CPU-only fallback: its block ``torch.cat`` lowering
        # trips an Inductor AVX2 C++ codegen bug, so only CPU needs it. Every other
        # device uses the block-diagonal contraction, which skips the structural
        # off-|m| zeros. ``make_fx`` resolves this Python branch at trace time, so
        # the freeze pins ``_force_block_diag_matmul`` to the AOTI target device
        # (tracing always runs on CPU regardless of where the artifact will run).
        if self._force_block_diag_matmul is None:
            use_block_diag = not x_flat.is_cpu
        else:
            use_block_diag = self._force_block_diag_matmul
        if use_block_diag:
            out_flat = self._block_diagonal_matmul(x_flat, weight)
        else:
            out_flat = torch.einsum("efi,ifo->efo", x_flat, weight)
        out = out_flat.reshape(
            n_edge, self.n_focus, self.reduced_dim, self.out_channels
        )

        # === Step 4. Bias on l=0 scalar index ===
        if self.mlp_bias:
            bias0 = self.bias0.view(self.n_focus, self.out_channels)
            out[:, :, 0, :] = out[:, :, 0, :] + bias0.unsqueeze(0)
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

    def train(self, mode: bool = True) -> SO2Linear:
        """Invalidate weight cache when switching to training mode."""
        self._cached_weight = None
        return super().train(mode)

    def _apply(self, fn: Any) -> SO2Linear:
        """Invalidate weight cache on device or dtype moves."""
        self._cached_weight = None
        return super()._apply(fn)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Invalidate weight cache before loading new weights."""
        self._cached_weight = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _build_so2_weight(self) -> torch.Tensor:
        """
        Assemble the per-focus block-diagonal SO(2) weight matrix.

        The flattened weight has shape ``(D_m*Cin, F, D_m*Cout)`` (in, out)
        where both axes follow the same m-major coefficient ordering.
        Off-diagonal blocks (cross-|m|) are zero, enforcing SO(2) equivariance.

        Returns
        -------
        torch.Tensor
            Weight with shape (D_m*Cin, F, D_m*Cout).
        """
        in_total = self.reduced_dim * self.in_channels
        out_total = self.reduced_dim * self.out_channels
        weight = self.weight_m0.new_zeros(in_total, self.n_focus, out_total)
        num_in_m0 = (self.lmax + 1) * self.in_channels
        num_out_m0 = (self.lmax + 1) * self.out_channels
        weight_m0 = self.weight_m0.view(num_in_m0, self.n_focus, num_out_m0)

        # m=0 block: (Cin_blk, F, Cout_blk) — (in, out) convention.
        weight[: self._m0_in, :, : self._m0_out] = weight_m0

        # |m|>0 blocks: fill the 2x2 SO(2) coupling structure.
        # For each |m|, the learnable param w has shape (in_blk, F, 2*out_blk)
        # which is split into W_u and W_v along the output axis.
        for m_idx, w in enumerate(self.weight_m):
            ni0, ni1, pi0, pi1, no0, no1, po0, po1 = self._block_slices[m_idx]
            ib = ni1 - ni0  # in_block size
            ob = no1 - no0  # out_block size
            w = w.view(ib, self.n_focus, 2 * ob)
            w_u = w[:, :, :ob]  # (in_blk, F, out_blk)
            w_v = w[:, :, ob:]  # (in_blk, F, out_blk)
            # Fill the 2x2 coupling:
            #   Row = input (neg/pos), Col = output (neg/pos).
            #   [ W_u^T, -W_v^T ]^T  =>  row=neg_in: W_u to neg_out, W_v to pos_out
            #   [ W_v^T,  W_u^T ]^T  =>  row=pos_in: -W_v to neg_out, W_u to pos_out
            weight[ni0:ni1, :, no0:no1] = w_u  # neg_in -> neg_out
            weight[ni0:ni1, :, po0:po1] = w_v  # neg_in -> pos_out
            weight[pi0:pi1, :, no0:no1] = -w_v  # pos_in -> neg_out
            weight[pi0:pi1, :, po0:po1] = w_u  # pos_in -> pos_out
        return weight

    def _block_diagonal_matmul(
        self, x_flat: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Contract only the diagonal ``|m|`` blocks of the assembled weight.

        ``weight`` is block-diagonal over ``|m|`` (cross-``|m|`` blocks are
        exactly zero), so concatenating the per-group matmuls reproduces the
        dense ``einsum`` over the full ``(D_m*Cin, D_m*Cout)`` matrix while
        skipping the structural zeros. The result is fp32-equivalent to the
        dense path up to the matmul reduction order.

        Parameters
        ----------
        x_flat : torch.Tensor
            Flattened input with shape ``(E, F, D_m*Cin)``.
        weight : torch.Tensor
            Assembled block-diagonal weight with shape ``(D_m*Cin, F, D_m*Cout)``.

        Returns
        -------
        torch.Tensor
            Flattened output with shape ``(E, F, D_m*Cout)``.
        """
        blocks = [
            torch.einsum(
                "efi,ifo->efo",
                x_flat[:, :, in0:in1],
                weight[in0:in1, :, out0:out1],
            )
            for in0, in1, out0, out1 in self._block_diag_slices
        ]
        return torch.cat(blocks, dim=-1)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
        return {
            "@class": "SO2Linear",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "mmax": self.mmax,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "n_focus": self.n_focus,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "mlp_bias": self.mlp_bias,
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj


class DynamicRadialDegreeMixer(nn.Module):
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
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
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
        self.dtype = dtype
        self.device = env.DEVICE

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

        self.weight = nn.Parameter(
            torch.empty(
                self.input_dim,
                self.proj_out_dim,
                device=self.device,
                dtype=self.dtype,
            )
        )
        init_trunc_normal_fan_in_out(self.weight, child_seed(seed, 0))

        if self.mode == "degree_channel" and self.rank > 0:
            self.channel_basis: nn.Parameter | None = nn.Parameter(
                torch.empty(
                    self.rank,
                    self.channels,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            init_trunc_normal_fan_in_out(self.channel_basis, child_seed(seed, 1))
        else:
            self.channel_basis = None

        compact_idx, dense_idx = self._build_dense_scatter_indices()
        self.register_buffer("kernel_compact_index", compact_idx, persistent=True)
        self.register_buffer("kernel_dense_index", dense_idx, persistent=True)
        for p in self.parameters():
            p.requires_grad = trainable

        # Inference fast path (opt-in via ``DP_TRITON_INFER``): a fused Triton
        # kernel replaces the dense scatter and the tiny batched matmul of the
        # ``degree_channel`` low-rank branch in the ``mmax == 1`` layout.
        self.use_triton_infer = use_triton_infer()
        self._radial_mix_block = None
        if (
            self.use_triton_infer
            and self.mode == "degree_channel"
            and self.rank > 0
            and self.mmax == 1
        ):
            from .triton.radial_mix import (
                radial_mix_block,
            )

            self._radial_mix_block = radial_mix_block

    def _build_dense_scatter_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
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
            torch.tensor(compact_indices, device=self.device, dtype=torch.long),
            torch.tensor(dense_indices, device=self.device, dtype=torch.long),
        )

    def _project_radial(self, radial_feat: torch.Tensor) -> torch.Tensor:
        radial_m0 = radial_feat[:, : self.lmax + 1, :].reshape(
            radial_feat.shape[0], self.input_dim
        )
        return torch.matmul(radial_m0, self.weight)

    def _scatter_degree_kernel(self, compact: torch.Tensor) -> torch.Tensor:
        n_edge = compact.shape[0]
        dense = compact.new_zeros(n_edge, self.reduced_dim * self.reduced_dim)
        source = compact.index_select(1, self.kernel_compact_index)
        dense.index_copy_(1, self.kernel_dense_index, source)
        return dense.view(n_edge, self.reduced_dim, self.reduced_dim)

    def _scatter_rank_kernel(self, compact: torch.Tensor) -> torch.Tensor:
        n_edge = compact.shape[0]
        dense = compact.new_zeros(
            n_edge, self.reduced_dim * self.reduced_dim, self.rank
        )
        source = compact.index_select(1, self.kernel_compact_index)
        dense.index_copy_(1, self.kernel_dense_index, source)
        return dense.view(n_edge, self.reduced_dim, self.reduced_dim, self.rank)

    def _scatter_channel_kernel(self, compact: torch.Tensor) -> torch.Tensor:
        n_edge = compact.shape[0]
        dense = compact.new_zeros(
            n_edge, self.reduced_dim * self.reduced_dim, self.channels
        )
        source = compact.index_select(1, self.kernel_compact_index)
        dense.index_copy_(1, self.kernel_dense_index, source)
        return dense.view(n_edge, self.reduced_dim, self.reduced_dim, self.channels)

    def forward(self, x_local: torch.Tensor, radial_feat: torch.Tensor) -> torch.Tensor:
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

        kernel_flat = self._project_radial(radial_feat)
        if self.mode == "degree":
            kernel = self._scatter_degree_kernel(kernel_flat)
            return torch.bmm(kernel, x_local)

        if self.rank > 0:
            compact = kernel_flat.view(
                x_local.shape[0], self.degree_kernel_size, self.rank
            )
            if self._radial_mix_block is not None and not self.training:
                return self._radial_mix_block(
                    compact, x_local, self.channel_basis, self.lmax
                )
            kernel = self._scatter_rank_kernel(compact)
            mixed = torch.einsum("eoir,eic->eorc", kernel, x_local)
            channel_basis = self.channel_basis.view(1, 1, self.rank, self.channels)
            return (mixed * channel_basis).sum(dim=2)

        compact = kernel_flat.view(
            x_local.shape[0], self.degree_kernel_size, self.channels
        )
        kernel = self._scatter_channel_kernel(compact)
        return torch.einsum("eoic,eic->eoc", kernel, x_local)


class SO2Convolution(nn.Module):
    """
    SO(2)-equivariant edge convolution with cached geometry and rotations.

    This module consumes node features in packed SO(3) layout `(N, D, C)` and
    performs edge message passing in the reduced m-major local layout. The
    operation pipeline is:

    1. `pre_focus_mix`: project node features `(N, D, C)` to the SO(2) hidden width.
    2. rotate global -> local reduced basis with cached `D_to_m`.
    3. radial modulation in reduced layout.
    4. `so2_layers` stacked local mixers:
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
    so2_norm
        If True, apply intermediate ReducedEquivariantRMSNorm as pre-norm before
        each SO(2) mixing layer. The last SO(2) layer always uses Identity.
    so2_layers
        Number of SO2Linear layers per convolution (default: 1).
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
    eps
        Small epsilon for normalization modules.
    dtype
        Parameter dtype.
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
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
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
        self.eps = float(eps)
        self.ebed_dim_full = get_so3_dim_of_lmax(self.lmax)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[dtype]
        self.compute_dtype = get_promoted_dtype(self.dtype)
        # Optional Triton inference kernels for the SO(2) convolution, enabled by
        # ``DP_TRITON_INFER=1`` (default disabled, in which case the dense
        # ``bmm`` rotation is used). The flag is read once at construction so it
        # is a compile-time constant in the traced (``make_fx``) graph, and it
        # only takes effect during inference.
        self.use_triton_infer = use_triton_infer()
        # Triton rotation kernels: block for the mmax == 1 layout, dense otherwise.
        self._rotate_to_local_fn = None
        self._rotate_back_fn = None
        if self.use_triton_infer:
            from .triton.so2_rotation import (
                rotate_back_block_so2,
                rotate_back_dense,
                rotate_to_local_block,
                rotate_to_local_dense,
            )

            if self.mmax == 1:
                self._rotate_to_local_fn = lambda x, src, wigner: rotate_to_local_block(
                    x, src, wigner, self.lmax
                )
                # The block kernel reads the (E, F, D_m, Cf) focus layout directly,
                # so the rotate-back path passes ``x_local`` before the global
                # reshape and the transpose-back copy is skipped (see Step 7).
                self._rotate_back_fn = lambda x_local, wigner: rotate_back_block_so2(
                    x_local, wigner, self.lmax
                )
            else:
                self._rotate_to_local_fn = lambda x, src, wigner: rotate_to_local_dense(
                    x, src, wigner, self.coeff_index_m, self.ebed_dim_full
                )
                self._rotate_back_fn = lambda x_local, wigner: rotate_back_dense(
                    x_local, wigner, self.coeff_index_m, self.ebed_dim_full
                )

        # === Step 1. Precompute coefficient indices for m-major reduced layout ===
        coeff_index_m = build_m_major_index(self.lmax, self.mmax, device=self.device)
        degree_index_m = build_m_major_l_index(self.lmax, self.mmax, device=self.device)
        degree_index_full = map_degree_idx(self.lmax, device=self.device)
        rotate_inv_rescale_full = build_rotate_inv_rescale(
            lmax=self.lmax,
            mmax=self.mmax,
            degree_index=degree_index_full,
            device=self.device,
            dtype=self.dtype,
        )
        self.register_buffer("coeff_index_m", coeff_index_m, persistent=True)
        self.register_buffer("degree_index_m", degree_index_m, persistent=True)
        self.register_buffer(
            "rotate_inv_rescale_full", rotate_inv_rescale_full, persistent=True
        )
        self.reduced_dim = int(coeff_index_m.numel())

        # === Step 2. Split deterministic seeds at the module top-level ===
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

        # === Step 3. Multiple SO2Linear layers ===
        self.so2_linears = nn.ModuleList(
            [
                SO2Linear(
                    lmax=self.lmax,
                    mmax=self.mmax,
                    in_channels=self.so2_focus_dim,
                    out_channels=(
                        2 * self.so2_focus_dim
                        if self.s2_activation and i < self.so2_layers - 1
                        else self.so2_focus_dim
                    ),
                    n_focus=self.n_focus,
                    dtype=self.dtype,
                    mlp_bias=self.mlp_bias,
                    seed=child_seed(seed_so2_stack, i),
                    trainable=trainable,
                )
                for i in range(self.so2_layers)
            ]
        )

        # === Step 4. Intermediate norms (Optional) ===
        inter_norms: list[nn.Module] = []
        if self.so2_norm:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(
                    ReducedEquivariantRMSNorm(
                        lmax=self.lmax,
                        mmax=self.mmax,
                        channels=self.so2_focus_dim,
                        degree_index_m=self.degree_index_m,
                        n_focus=self.n_focus,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                    )
                )
        else:
            for _ in range(max(0, self.so2_layers - 1)):
                inter_norms.append(nn.Identity())
        inter_norms.append(nn.Identity())
        self.so2_inter_norms = nn.ModuleList(inter_norms)

        # === Step 5. Intermediate non-linearity ===
        non_linearities: list[nn.Module] = []
        for i in range(max(0, self.so2_layers - 1)):
            if self.s2_activation:
                non_linearities.append(
                    S2GridNet(
                        lmax=self.lmax,
                        mmax=self.mmax,
                        channels=self.so2_focus_dim,
                        n_focus=self.n_focus,
                        mode="self",
                        op_type="glu",
                        dtype=self.compute_dtype,
                        layout="nfdc",
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
                        dtype=self.compute_dtype,
                        activation_function=self.activation_function,
                        mlp_bias=self.mlp_bias,
                        layout="nfdc",
                        trainable=trainable,
                        seed=child_seed(seed_non_linearities, i),
                    )
                )
        non_linearities.append(nn.Identity())
        self.non_linearities = nn.ModuleList(non_linearities)

        # === Step 5.5. Optional depth-wise attention residuals across SO(2) layers ===
        if self.use_so2_attn_res:
            self.so2_layer_attn_res: nn.ModuleList | None = nn.ModuleList(
                [
                    DepthAttnRes(
                        channels=self.hidden_channels,
                        input_dependent=self.so2_attn_res_mode == "dependent",
                        eps=self.eps,
                        bias=self.mlp_bias,
                        dtype=self.compute_dtype,
                        trainable=trainable,
                        seed=child_seed(seed_depth_attn, i),
                    )
                    for i in range(self.so2_layers)
                ]
            )
        else:
            self.so2_layer_attn_res = None

        # === Step 6. Optional per-layer LayerScale for SO(2) residual branches ===
        if self.layer_scale:
            self.adam_so2_layer_scales = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(
                            self.n_focus,
                            self.so2_focus_dim,
                            dtype=self.dtype,
                            device=self.device,
                        )
                        * 1e-3,
                        requires_grad=trainable,
                    )
                    for _ in range(self.so2_layers)
                ]
            )
        else:
            self.adam_so2_layer_scales = None

        # === Step 7. Optional attention projections (n_atten_head > 0) ===
        self.attn_qk_norm: ScalarRMSNorm | None = None
        self.attn_q_proj: FocusLinear | None = None
        self.attn_k_proj: FocusLinear | None = None
        self.attn_focus_mix: SO3Linear | None = None
        self.attn_v_proj: SO3Linear | None = None
        self.attn_o_proj: SO3Linear | None = None
        self.adamw_attn_logit_w: nn.Parameter | None = None
        self.adamw_attn_z_bias_raw: nn.Parameter | None = None
        self.attn_output_gate_norm: ScalarRMSNorm | None = None
        self.adamw_attn_gate_w: nn.Parameter | None = None
        if self.n_atten_head > 0:
            self.attn_qk_norm = ScalarRMSNorm(
                channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.attn_q_proj = FocusLinear(
                in_channels=self.attn_focus_dim,
                out_channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                dtype=self.compute_dtype,
                bias=False,
                seed=child_seed(seed_gate, 0),
                trainable=trainable,
            )
            self.attn_k_proj = FocusLinear(
                in_channels=self.attn_focus_dim,
                out_channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                dtype=self.compute_dtype,
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
                    dtype=self.compute_dtype,
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
                    dtype=self.compute_dtype,
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
                    dtype=self.compute_dtype,
                    mlp_bias=False,
                    seed=child_seed(seed_gate, 21),
                    trainable=trainable,
                )
            self.adamw_attn_logit_w = nn.Parameter(
                torch.empty(
                    self.attn_focus_dim,
                    self.attn_n_focus,
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_attn_logit_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 2)),
            )
            # softplus(0.5413) ~= 1.0 provides balanced initial competition.
            self.adamw_attn_z_bias_raw = nn.Parameter(
                torch.full(
                    (self.attn_n_focus, self.n_atten_head),
                    0.5413,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            self.attn_output_gate_norm = ScalarRMSNorm(
                channels=self.attn_focus_dim,
                n_focus=self.attn_n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.adamw_attn_gate_w = nn.Parameter(
                torch.empty(
                    self.attn_focus_dim,
                    self.attn_n_focus,
                    self.n_atten_head,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_attn_gate_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 3)),
            )

        # === Step 7.5. Optional cross-focus competition ===
        self.focus_compete_norm: ScalarRMSNorm | None = None
        self.adamw_focus_compete_w: nn.Parameter | None = None
        self.focus_compete_bias: nn.Parameter | None = None
        if self.focus_compete and self.n_focus > 1:
            self.focus_compete_norm = ScalarRMSNorm(
                channels=self.so2_focus_dim,
                n_focus=self.n_focus,
                eps=self.eps,
                dtype=self.compute_dtype,
                trainable=trainable,
            )
            self.adamw_focus_compete_w = nn.Parameter(
                torch.empty(
                    self.so2_focus_dim,
                    self.n_focus,
                    dtype=self.compute_dtype,
                    device=self.device,
                ),
                requires_grad=trainable,
            )
            nn.init.normal_(
                self.adamw_focus_compete_w,
                mean=0.0,
                std=0.01,
                generator=get_generator(child_seed(seed_gate, 4)),
            )
            if self.mlp_bias:
                self.focus_compete_bias = nn.Parameter(
                    torch.zeros(
                        self.n_focus,
                        dtype=self.compute_dtype,
                        device=self.device,
                    ),
                    requires_grad=trainable,
                )

        # === Step 8. Optional radial hidden projection ===
        self.radial_hidden_proj: ChannelLinear | None = None
        if self.use_hidden_projection:
            self.radial_hidden_proj = ChannelLinear(
                in_channels=self.channels,
                out_channels=self.hidden_channels,
                dtype=self.dtype,
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
                dtype=self.dtype,
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
                    dtype=self.compute_dtype,
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
                    dtype=self.compute_dtype,
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
                    dtype=self.compute_dtype,
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
                    dtype=self.compute_dtype,
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
            dtype=dtype,
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
            dtype=dtype,
            mlp_bias=self.mlp_bias,
            trainable=trainable,
            seed=seed_so3_post,
            init_std=0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        radial_feat: torch.Tensor,
    ) -> torch.Tensor:
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
        torch.Tensor
            Message updates with shape (N, D, C).
        """
        src, dst = edge_cache.src, edge_cache.dst
        n_node = x.shape[0]
        n_edge = src.numel()

        # === Step 1. Pre-focus channel mixing on full width ===
        with nvtx_range("SO2Conv/pre_focus_mix"):
            # (N, D, C_wide), C_wide = F * Cf
            x = self.pre_focus_mix(x.unsqueeze(2)).squeeze(2)

        # === Step 2. Rotate to edge-aligned local frame ===
        with nvtx_range("SO2Conv/rotate_to_local"):
            D_full = edge_cache.D_full
            x_dst_local: torch.Tensor | None = None
            if self.use_triton_infer and not self.training:
                # ``self._rotate_to_local_fn`` was bound in ``__init__`` (the
                # block kernel for the m-major ``mmax == 1`` layout, dense
                # otherwise).
                x_local = self._rotate_to_local_fn(x, src, D_full)  # (E, D_m, C_wide)
                if self.node_wise_grid_product is not None:
                    x_dst_local = self._rotate_to_local_fn(
                        x, dst, D_full
                    )  # (E, D_m, C_wide)
            else:
                D_m_prime = project_D_to_m(
                    D_full=D_full,
                    coeff_index_m=self.coeff_index_m,
                    ebed_dim_full=self.ebed_dim_full,
                    cache=edge_cache.D_to_m_cache,
                    key_lmax=self.lmax,
                    key_mmax=self.mmax,
                )
                x_src = x.index_select(0, src)  # (E, D, C_wide)
                x_local = torch.bmm(D_m_prime, x_src)  # (E, D_m, C_wide)
                if self.node_wise_grid_product is not None:
                    x_dst = x.index_select(0, dst)  # (E, D, C_wide)
                    x_dst_local = torch.bmm(D_m_prime, x_dst)  # (E, D_m, C_wide)

        # === Step 3. Select radial/type features for reduced layout ===
        with nvtx_range("SO2Conv/radial_fuse"):
            rad_feat = radial_feat[:, self.degree_index_m, :]  # (E, D_m, C)
            if self.radial_hidden_proj is not None:
                rad_feat = self.radial_hidden_proj(rad_feat)
            if self.radial_degree_mixer is None:
                x_local.mul_(rad_feat)
            else:
                x_local = self.radial_degree_mixer(x_local, rad_feat)
            if self.node_wise_grid_product is not None:
                x_local = x_local + self.node_wise_grid_product(
                    x_local,
                    x_dst_local,
                )
            rad_feat_l0_focus = rad_feat[:, 0, :].reshape(
                n_edge, self.n_focus, self.so2_focus_dim
            )  # (E, F, Cf)

        # === Step 4. Convert to SO(2) internal focus layout ===
        focus_gate_src: torch.Tensor | None = None
        with nvtx_range("SO2Conv/reshape_for_so2"):
            x_local = x_local.reshape(
                n_edge, self.reduced_dim, self.n_focus, self.so2_focus_dim
            ).transpose(1, 2)  # (E, F, D_m, Cf), strided
            if self.focus_compete and self.n_focus > 1:
                focus_gate_src = x_local[:, :, 0, :]

        # === Step 5. Multi-layer SO(2) mixing (pre-norm + residual) ===
        with nvtx_range("SO2Conv/so2_layers"):

            def so2_l0_extractor(v: torch.Tensor) -> torch.Tensor:
                """Extract scalar features from SO(2) reduced layout."""
                return v[:, :, 0, :].reshape(v.shape[0], self.hidden_channels)

            def apply_bias_correction(
                x_local: torch.Tensor,
                so2_linear: SO2Linear,
                layer_idx: int,
            ) -> None:
                if layer_idx != 0 or so2_linear.bias0 is None:
                    return
                bias0 = so2_linear.bias0.view(
                    self.n_focus, so2_linear.out_channels
                ).unsqueeze(0)
                if so2_linear.out_channels == self.so2_focus_dim:
                    radial_factor = rad_feat_l0_focus
                elif so2_linear.out_channels == 2 * self.so2_focus_dim:
                    radial_factor = torch.cat(
                        [rad_feat_l0_focus, rad_feat_l0_focus], dim=-1
                    )
                else:
                    raise RuntimeError(
                        "Unexpected SO2Linear output width in bias correction"
                    )
                bias_correction = bias0 * (
                    radial_factor * edge_cache.edge_env.reshape(-1, 1, 1) - 1.0
                )
                x_local[:, :, 0, :].add_(bias_correction)

            if self.use_so2_attn_res:
                so2_depth_sources = [x_local]
                for layer_idx, (so2_linear, inter_norm, non_linear) in enumerate(
                    zip(
                        self.so2_linears,
                        self.so2_inter_norms,
                        self.non_linearities,
                        strict=True,
                    )
                ):
                    x_local: torch.Tensor = self.so2_layer_attn_res[layer_idx](
                        sources=so2_depth_sources,
                        scalar_extractor=so2_l0_extractor,
                        current_x=x_local,
                    )
                    residual = x_local
                    x_local = inter_norm(x_local)
                    x_local = so2_linear(x_local)
                    apply_bias_correction(x_local, so2_linear, layer_idx)

                    x_local = non_linear(x_local)

                    if self.layer_scale:
                        scale: torch.Tensor = self.adam_so2_layer_scales[
                            layer_idx
                        ].reshape(1, self.n_focus, 1, self.so2_focus_dim)
                        x_local = residual + scale * x_local
                    else:
                        x_local = residual + x_local
                    so2_depth_sources.append(x_local - residual)
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
                    apply_bias_correction(x_local, so2_linear, layer_idx)

                    x_local = non_linear(x_local)

                    if self.layer_scale:
                        scale = self.adam_so2_layer_scales[layer_idx].reshape(
                            1, self.n_focus, 1, self.so2_focus_dim
                        )
                        x_local = residual + scale * x_local
                    else:
                        x_local = residual + x_local

        # === Step 6. Cross-focus softmax competition ===
        if self.focus_compete and self.n_focus > 1:
            focus_gate_src = focus_gate_src.to(dtype=self.compute_dtype)
            focus_logits = torch.einsum(
                "efi,if->ef",
                self.focus_compete_norm(focus_gate_src),
                self.adamw_focus_compete_w,
            )
            if self.mlp_bias:
                focus_logits = focus_logits + self.focus_compete_bias.unsqueeze(0)
            alpha = torch.softmax(focus_logits / self.focus_softmax_tau, dim=1).to(
                dtype=x_local.dtype
            )
            alpha = alpha * (1.0 - self.focus_label_smoothing) + (
                self.focus_label_smoothing / float(self.n_focus)
            )
            x_local = x_local * alpha.unsqueeze(-1).unsqueeze(-1)

        # === Step 7. Rotate back to global frame ===
        with nvtx_range("SO2Conv/rotate_back"):
            Dt_full = edge_cache.Dt_full
            if self.use_triton_infer and self.mmax == 1 and not self.training:
                # The block kernel consumes the (E, F, D_m, Cf) focus layout in
                # place, folding the inverse transpose into its channel addressing.
                x_message = self._rotate_back_fn(x_local, Dt_full)  # (E, D, C_wide)
            else:
                # Restore reduced global layout (E, D_m, C_wide) for inverse rotation.
                x_local = (
                    x_local.transpose(1, 2)
                    .contiguous()
                    .reshape(n_edge, self.reduced_dim, self.hidden_channels)
                )
                if self.use_triton_infer and not self.training:
                    x_message = self._rotate_back_fn(x_local, Dt_full)  # (E, D, C_wide)
                else:
                    Dt_from_m = project_Dt_from_m(
                        Dt_full=Dt_full,
                        coeff_index_m=self.coeff_index_m,
                        ebed_dim_full=self.ebed_dim_full,
                        cache=edge_cache.Dt_from_m_cache,
                        key_lmax=self.lmax,
                        key_mmax=self.mmax,
                    )
                    x_message = torch.bmm(Dt_from_m, x_local)  # (E, D, C_wide)
            # Reduced layouts keep only 2*mmax+1 orders for l>mmax. Applying the
            # inverse-rotation degree rescale after the global lift restores the
            # full-basis amplitude expected by the block output contract.
            x_message = x_message * self.rotate_inv_rescale_full.view(1, -1, 1)
            if self.attn_focus_mix is not None:
                x_message = self.attn_focus_mix(x_message.unsqueeze(2)).squeeze(2)

        # === Step 8. Aggregate with optional head-wise gating ===
        with nvtx_range("SO2Conv/aggregate"):
            # Source Freeze Propagation Gate: broadcast the per-edge scalar
            # eta[src] to the edge message before destination aggregation.
            # ``edge_src_gate`` is ``None`` outside bridging mode, in which
            # case this branch disappears and the baseline / attention paths
            # run unchanged.
            edge_src_gate = edge_cache.edge_src_gate
            if self.n_atten_head == 0:
                # Baseline path: fused envelope-weighted scatter add -> degree norm.
                # Folding edge_src_gate into the scalar envelope keeps the
                # op count unchanged.
                edge_weight = edge_cache.edge_env  # (E, 1)
                if edge_src_gate is not None:
                    edge_weight = edge_weight * edge_src_gate.to(
                        dtype=edge_weight.dtype
                    )
                x_message = x_message * edge_weight.unsqueeze(-1)
                out = x.new_zeros(x.shape, dtype=self.compute_dtype)
                out.index_add_(0, dst, x_message.to(dtype=self.compute_dtype))
                out.mul_(edge_cache.inv_sqrt_deg.to(dtype=self.compute_dtype))
                out = out.to(dtype=self.dtype)  # (N, D, C_wide)
            else:
                # === Step 8.1. Build attention logits from scalar channels ===
                compute_dtype = self.compute_dtype
                x_l0_node = x[:, 0, :].reshape(
                    n_node, self.attn_n_focus, self.attn_focus_dim
                )  # (N, Fa, Ca)
                qk_input = self.attn_qk_norm(x_l0_node.to(dtype=compute_dtype))
                q_node = self.attn_q_proj(qk_input)  # (N, Fa, Ca)
                k_node = self.attn_k_proj(qk_input)  # (N, Fa, Ca)
                q_edge = q_node.index_select(0, dst).reshape(
                    n_edge, self.attn_n_focus, self.n_atten_head, self.head_dim
                )  # (E, Fa, H, Ch), Ca = H * Ch
                k_edge = k_node.index_select(0, src).reshape(
                    n_edge, self.attn_n_focus, self.n_atten_head, self.head_dim
                )  # (E, Fa, H, Ch)
                radial_l0 = rad_feat[:, 0, :].reshape(
                    n_edge, self.attn_n_focus, self.attn_focus_dim
                )  # (E, Fa, Ca)
                radial_bias = torch.einsum(
                    "efi,ifo->efo",
                    radial_l0.to(dtype=compute_dtype),
                    self.adamw_attn_logit_w,
                )  # (E, F, H)
                attn_logits: torch.Tensor = (q_edge * k_edge).sum(-1) * (
                    self.head_dim**-0.5
                )
                attn_logits = attn_logits + radial_bias

                # === Step 8.2. Destination-wise stable envelope-gated softmax ===
                # ``src_weight=edge_src_gate`` folds SFPG into both the
                # numerator and the denominator of the softmax. A muted
                # source (``eta_src = 0``) therefore drops out of the
                # destination's attention normalization entirely, which
                # is required for the attention path to honor the
                # frozen-zone invariance: a post-multiplication on
                # ``attn_alpha`` alone would still leave the muted
                # source leaking through the shared denominator.
                attn_alpha = segment_envelope_gated_softmax(
                    logits=attn_logits,
                    edge_env=edge_cache.edge_env.to(dtype=compute_dtype),
                    dst=dst,
                    n_nodes=n_node,
                    z_bias_raw=self.adamw_attn_z_bias_raw,
                    eps=self.eps,
                    src_weight=(
                        None
                        if edge_src_gate is None
                        else edge_src_gate.to(dtype=compute_dtype)
                    ),
                )  # (E, F, H)

                # === Step 8.3. Value projection and head-wise aggregation ===
                value_focus = x_message.reshape(
                    n_edge,
                    self.ebed_dim_full,
                    self.attn_n_focus,
                    self.attn_focus_dim,
                ).to(dtype=compute_dtype)  # (E, D, Fa, Ca)
                if self.attn_v_proj is not None:
                    value_focus = self.attn_v_proj(value_focus)
                value_heads = value_focus.reshape(
                    n_edge,
                    self.ebed_dim_full,
                    self.attn_n_focus,
                    self.n_atten_head,
                    self.head_dim,
                )  # (E, D, Fa, H, Ch)
                weighted_value = value_heads * attn_alpha.reshape(
                    n_edge, 1, self.attn_n_focus, self.n_atten_head, 1
                )
                out_heads = torch.zeros(
                    n_node,
                    self.ebed_dim_full,
                    self.attn_n_focus,
                    self.n_atten_head,
                    self.head_dim,
                    device=x.device,
                    dtype=compute_dtype,
                )  # (N, D, Fa, H, Ch)
                out_heads.index_add_(0, dst, weighted_value)

                # === Step 8.4. Output-side head gate ===
                attn_output_gate = torch.sigmoid(
                    torch.einsum(
                        "nfi,ifo->nfo",
                        self.attn_output_gate_norm(x_l0_node.to(dtype=compute_dtype)),
                        self.adamw_attn_gate_w,
                    )
                )  # (N, F, H)
                out_heads = out_heads * attn_output_gate.reshape(
                    n_node, 1, self.attn_n_focus, self.n_atten_head, 1
                )  # (N, D, Fa, H, Ch)

                # === Step 8.5. Output projection and merge heads ===
                out_focus = out_heads.reshape(
                    n_node,
                    self.ebed_dim_full,
                    self.attn_n_focus,
                    self.attn_focus_dim,
                )  # (N, D, Fa, Ca)
                if self.attn_o_proj is not None:
                    out_focus = self.attn_o_proj(out_focus)
                out = out_focus.reshape(
                    n_node, self.ebed_dim_full, self.hidden_channels
                ).to(dtype=self.dtype)  # (N, D, C_wide)

        # === Step 9. Optional message-node grid product ===
        if self.message_node_grid_product is not None:
            with nvtx_range("SO2Conv/message_node_grid"):
                out = out + self.message_node_grid_product(out, x)

        # === Step 10. Final channel mixing ===
        with nvtx_range("SO2Conv/post_focus_mix"):
            out = self.post_focus_mix(out.unsqueeze(2)).squeeze(2)
        return out  # (N, D, C)

    def serialize(self) -> dict[str, Any]:
        trainable = all(p.requires_grad for p in self.parameters())
        state = self.state_dict()
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
                "precision": RESERVED_PRECISION_DICT[self.dtype],
                "trainable": trainable,
                "seed": None,
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SO2Convolution:
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "SO2Convolution":
            raise ValueError(f"Invalid class for SO2Convolution: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        variables = data.pop("@variables")
        precision = config.pop("precision")
        config["dtype"] = PRECISION_DICT[precision]
        obj = cls(**config)
        template = obj.state_dict()
        state = {
            key: safe_numpy_to_tensor(
                value, device=template[key].device, dtype=template[key].dtype
            )
            for key, value in variables.items()
        }
        obj.load_state_dict(state)
        return obj
