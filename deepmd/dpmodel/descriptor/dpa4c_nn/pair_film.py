# SPDX-License-Identifier: LGPL-3.0-or-later
"""Ordered type-pair FiLM cache for DPA4C."""

from __future__ import (
    annotations,
)

import math
from typing import (
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

from ..dpa4_nn.mlp import (
    SwiGLUMLP,
    resolve_swiglu_hidden_width,
)


class OrderedPairFiLM(NativeOP):
    r"""Map type embeddings to the ordered finite type-pair cache.

    For center type :math:`a` and neighbor type :math:`b`, the network
    evaluates

    .. math::

       z_{ab}&=[T_a\Vert T_b],\\
       h_{ab}&=\operatorname{SwiGLU}(z_{ab}W_{\rm in}),\\
       [s_{ab},d_{ab},u_{ab},p_{ab},q_{ab}]&=0.1\,h_{ab}W_{\rm out},\\
       \gamma_{ab}&=1+\tanh(s_{ab}),\\
       \beta_{ab}&=T_a+T_b+\tanh(d_{ab}),\\
       U_{ab}&=\tanh(u_{ab}),\\
       \gamma^{s}_{ab}&=\tanh(a^{\gamma}+p_{ab}),\qquad
       \beta^{s}_{ab}=\tanh(a^{\beta}+q_{ab}).

    The network is evaluated over the finite type table rather than over graph
    edges, so compressed inference stores only the resulting tables. Bounding
    every output keeps the cache well conditioned in ``float32``:
    :math:`\gamma` stays in :math:`(0,2)`, and the residual parts of
    :math:`\beta`, :math:`U`, :math:`\gamma^{s}` and :math:`\beta^{s}` stay in
    :math:`(-1,1)`.

    The spin scale :math:`\gamma^{s}` is signed, unlike its geometric
    counterpart. The exchange interaction of an ordered pair may be either
    ferromagnetic or antiferromagnetic, and the radial map is shared across
    pairs, so the sign has to be available in the pair cache.

    That sign freedom rules out the structural anchor the geometric heads use
    -- the constant one for :math:`\gamma`, the type embedding for
    :math:`\beta` -- and the SwiGLU trunk is bias free, so without an anchor
    of their own the two spin heads would emerge from a small centred product
    and start several orders of magnitude below the geometric tables. The
    descriptor calibration would then freeze a preconditioner at a scale the
    first optimizer steps immediately leave. Both heads are therefore anchored
    on a learned per-channel offset :math:`a`, initialized to
    :math:`\pm\operatorname{artanh}` of :attr:`_SPIN_ANCHOR_MAGNITUDE` with an
    independent random sign per channel. Every spin channel therefore starts
    at that magnitude exactly, for any channel width and any seed, with its
    sign left free.

    Parameters
    ----------
    channels
        Type-embedding and FiLM channel width :math:`C`.
    radial_modes
        Number :math:`R` of shared radial mode profiles each ordered pair
        mixes. Zero omits the mixing table.
    spin_channels
        Number :math:`C_s` of ordered spin scale and shift channels. Zero
        omits the spin tables.
    precision
        Parameter precision.
    trainable
        Whether the pair-encoder weights are trainable.
    seed
        Random seed.

    Raises
    ------
    ValueError
        If ``channels`` is not positive, or if ``radial_modes`` or
        ``spin_channels`` is negative.
    """

    _OUTPUT_SCALE = 0.1

    #: Initial magnitude of both ordered spin tables. It is the same order as
    #: the geometric scale and shift and stays well clear of the saturated
    #: region of the bounding nonlinearity, whose derivative there is 0.75.
    _SPIN_ANCHOR_MAGNITUDE = 0.5

    def __init__(
        self,
        channels: int,
        radial_modes: int = 0,
        spin_channels: int = 0,
        *,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if channels <= 0:
            raise ValueError(f"`channels` must be positive, got {channels}")
        if radial_modes < 0:
            raise ValueError(f"`radial_modes` must be non-negative, got {radial_modes}")
        if spin_channels < 0:
            raise ValueError(
                f"`spin_channels` must be non-negative, got {spin_channels}"
            )
        self.channels = int(channels)
        self.radial_modes = int(radial_modes)
        self.spin_channels = int(spin_channels)
        self.precision = str(precision)
        self.trainable = bool(trainable)
        input_dim = 2 * self.channels
        self.hidden_dim = resolve_swiglu_hidden_width(input_dim)
        output_dim = self.channels * (2 + self.radial_modes) + 2 * self.spin_channels
        self.network = SwiGLUMLP(
            [input_dim, self.hidden_dim, output_dim],
            output_scale=self._OUTPUT_SCALE,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, 0),
        )
        if self.spin_channels == 0:
            # The anchors exist only alongside the spin head they bias.
            self.adam_spin_scale_anchor = None
            self.adam_spin_shift_anchor = None
        else:
            rng = np.random.default_rng(child_seed(seed, 1))
            precision_dtype = PRECISION_DICT[self.precision.lower()]
            offset = math.atanh(self._SPIN_ANCHOR_MAGNITUDE)
            signs = rng.integers(0, 2, size=(2, self.spin_channels)) * 2.0 - 1.0
            self.adam_spin_scale_anchor = (offset * signs[0]).astype(precision_dtype)
            self.adam_spin_shift_anchor = (offset * signs[1]).astype(precision_dtype)

    @property
    def pair_hidden_width(self) -> int:
        """Return the width of the encoder's hidden pre-activation."""
        return 2 * self.hidden_dim

    def pair_latent(self, type_embedding: Any) -> tuple[Any, Any]:
        """Build the condition-independent ordered-pair state.

        Both returned tables are functions of the ordered type pair alone.
        Splitting them off lets a frame-level condition enter as an additive
        pre-activation bias, which one shared projection over the finite type
        table then serves for every frame.

        Parameters
        ----------
        type_embedding
            Complete type table with shape ``(T + 1, channels)``, where the
            trailing row is the zero padding type.

        Returns
        -------
        pre_activation
            Hidden affine pre-activation with shape
            ``((T + 1) ** 2, pair_hidden_width)``.
        base_shift
            Structural shift anchor :math:`T_a + T_b` with shape
            ``((T + 1) ** 2, channels)``.
        """
        xp = array_api_compat.array_namespace(type_embedding)
        ntypes = type_embedding.shape[0]
        pair_shape = (ntypes, ntypes, self.channels)
        pair_input = xp.reshape(
            xp.concat(
                [
                    xp.broadcast_to(type_embedding[:, None, :], pair_shape),
                    xp.broadcast_to(type_embedding[None, :, :], pair_shape),
                ],
                axis=-1,
            ),
            (-1, 2 * self.channels),
        )
        return (
            self.network.call_hidden_affine(pair_input),
            xp.reshape(
                type_embedding[:, None, :] + type_embedding[None, :, :],
                (-1, self.channels),
            ),
        )

    def heads(
        self, pre_activation: Any, base_shift: Any
    ) -> tuple[Any, Any, Any | None, Any | None, Any | None]:
        """Finish the conditioning tables from a hidden pre-activation.

        Every operation acts on the trailing axis, so the same head applies
        to the finite ordered-pair table and to a per-edge expansion of it.

        Parameters
        ----------
        pre_activation
            Hidden affine pre-activation with shape
            ``(..., pair_hidden_width)``.
        base_shift
            Structural shift anchor with shape ``(..., channels)``.

        Returns
        -------
        scale
            Radial scales with shape ``(..., channels)``.
        shift
            Radial shifts with shape ``(..., channels)``.
        mixing
            Mode-mixing matrices with shape
            ``(..., channels, radial_modes)``, or ``None`` when
            ``radial_modes`` is zero.
        spin_scale
            Spin scales with shape ``(..., spin_channels)``, or ``None`` when
            ``spin_channels`` is zero.
        spin_shift
            Spin shifts with the same shape as ``spin_scale``.
        """
        xp = array_api_compat.array_namespace(pre_activation)
        device = array_api_compat.device(pre_activation)
        logits = self.network.call_from_hidden_affine(pre_activation)

        # The output splits into the scale, the shift residual, the flattened
        # mixing matrix, and the two spin tables, in that order.
        shift_end = 2 * self.channels
        mixing_end = shift_end + self.channels * self.radial_modes
        spin_scale_end = mixing_end + self.spin_channels

        def anchored(anchor: Any, block: Any) -> Any:
            """Bound one spin head around its learned per-channel offset."""
            return xp.tanh(
                block
                + xp_asarray_nodetach(
                    xp,
                    anchor,
                    dtype=block.dtype,
                    device=device,
                )
            )

        return (
            1.0 + xp.tanh(logits[..., : self.channels]),
            base_shift + xp.tanh(logits[..., self.channels : shift_end]),
            None
            if self.radial_modes == 0
            else xp.reshape(
                xp.tanh(logits[..., shift_end:mixing_end]),
                (-1, self.channels, self.radial_modes),
            ),
            None
            if self.spin_channels == 0
            else anchored(
                self.adam_spin_scale_anchor,
                logits[..., mixing_end:spin_scale_end],
            ),
            None
            if self.spin_channels == 0
            else anchored(self.adam_spin_shift_anchor, logits[..., spin_scale_end:]),
        )

    def call(
        self, type_embedding: Any, hidden_bias: Any = None
    ) -> tuple[Any, Any, Any | None, Any | None, Any | None]:
        """Build the ordered scale, shift, mixing, and spin tables.

        Parameters
        ----------
        type_embedding
            Complete type table with shape ``(T + 1, channels)``, where the
            trailing row is the zero padding type.
        hidden_bias
            Optional frame-condition bias with shape
            ``(pair_hidden_width,)``, added to the hidden pre-activation of
            every ordered pair. ``None`` leaves the cache unconditioned.

        Returns
        -------
        scale
            Ordered radial scales with shape ``((T + 1) ** 2, channels)``.
        shift
            Ordered radial shifts with shape ``((T + 1) ** 2, channels)``.
        mixing
            Ordered mode-mixing matrices with shape
            ``((T + 1) ** 2, channels, radial_modes)``, or ``None`` when
            ``radial_modes`` is zero.
        spin_scale
            Ordered spin scales with shape
            ``((T + 1) ** 2, spin_channels)``, or ``None`` when
            ``spin_channels`` is zero.
        spin_shift
            Ordered spin shifts with the same shape as ``spin_scale``.
        """
        pre_activation, base_shift = self.pair_latent(type_embedding)
        if hidden_bias is not None:
            pre_activation = pre_activation + hidden_bias
        return self.heads(pre_activation, base_shift)

    def serialize(self) -> dict[str, Any]:
        """Serialize the ordered pair encoder.

        Returns
        -------
        dict[str, Any]
            Versioned configuration, pair-encoder parameters, and the two spin
            anchors, which are present only for a spin-conditioned cache.
        """
        return {
            "@class": "OrderedPairFiLM",
            "@version": 1,
            "channels": self.channels,
            "radial_modes": self.radial_modes,
            "spin_channels": self.spin_channels,
            "precision": self.precision,
            "trainable": self.trainable,
            "network": self.network.serialize(),
            "@variables": {}
            if self.spin_channels == 0
            else {
                "adam_spin_scale_anchor": to_numpy_array(self.adam_spin_scale_anchor),
                "adam_spin_shift_anchor": to_numpy_array(self.adam_spin_shift_anchor),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OrderedPairFiLM:
        """Deserialize an :class:`OrderedPairFiLM`.

        Parameters
        ----------
        data
            Versioned dictionary produced by :meth:`serialize`.

        Returns
        -------
        OrderedPairFiLM
            Reconstructed ordered type-pair module.

        Raises
        ------
        ValueError
            If the payload does not describe an :class:`OrderedPairFiLM`.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        if data.pop("@class") != "OrderedPairFiLM":
            raise ValueError("Invalid serialized class for OrderedPairFiLM")
        network = data.pop("network")
        variables = data.pop("@variables")
        obj = cls(**data)
        obj.network = SwiGLUMLP.deserialize(network)
        precision_dtype = PRECISION_DICT[obj.precision.lower()]
        for name, value in variables.items():
            setattr(obj, name, np.asarray(value, dtype=precision_dtype))
        return obj
