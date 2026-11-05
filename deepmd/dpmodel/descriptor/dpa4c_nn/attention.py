# SPDX-License-Identifier: LGPL-3.0-or-later
"""Attention of the further DPA4C message-passing blocks."""

from __future__ import (
    annotations,
)

import itertools
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
    xp_sigmoid,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    segment_sum,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    softplus_t,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from ..dpa4_nn.mlp import (
    SwiGLUMLP,
)
from .geometry import (
    degree_offsets,
)

#: Single sub-layers of a block, serialized by name.
_SINGLE_LAYERS = (
    "hidden_state",
    "hidden_field",
    "hidden_amplitude",
    "hidden_output",
    "transport_radial",
    "null_mass",
)
#: Per-degree sub-layer lists of a block, serialized by name.
_LAYER_LISTS = ("key_projections", "transport_outputs")


def soft_saturate(value: Any, knee: float) -> Any:
    r"""Map every entry through :math:`x/\sqrt{1+(x/x_0)^2}`.

    The map is the identity to first order and bounds its output by the knee
    :math:`x_0`; its slope decays as :math:`|x|^{-3}` far beyond it.
    """
    xp = array_api_compat.array_namespace(value)
    ratio = value / knee
    return value / xp.sqrt(1.0 + ratio * ratio)


class BlockAttention(NativeOP):
    r"""Attention of one further message-passing block.

    Block :math:`k` reads the moments :math:`X^{(k-1)}` and the calibrated
    invariants :math:`D^{(k-1)}` of the previous block. Every node carries
    saturated equivariant keys, one per head :math:`h` and non-scalar degree,

    .. math::

       K_{jh}^{(\ell)}=X_j^{(\ell)}A_h^{(\ell)}\Big/
       \sqrt{1+|X_j^{(\ell)}A_h^{(\ell)}|^2/\kappa^2},

    and every edge :math:`j\to i` reads the fields of its source along the
    bond towards the center, :math:`F_{ij\ell h}=B^{(\ell)}(\hat u_{ji})\cdot
    K_{jh}^{(\ell)}`, which measure how the environment of the neighbor is
    oriented relative to the bond. The logit of amplitude channel :math:`c`
    sums three terms of the neighbor,

    .. math::

       a_{ijc}=\sum_\ell g_{\ell h}F_{ij\ell h}+s_{jc}
       +\bigl[\mathrm{SiLU}\bigl(W_D\tilde D_j+W_FF_{ij}
       +W_\phi\tilde\phi_{ij}\bigr)W_O\bigr]_c,\qquad h=c\bmod H:

    a field per head, the encoded state :math:`s_j` of its saturated
    invariants :math:`\tilde D_j`, and a hidden layer that reads the state,
    every field and the saturated first-block amplitude
    :math:`\tilde\phi_{ij}` together. The hidden layer lets the effect of the
    neighbor on a channel depend on the distance and the type pair of the
    edge, which a factor of the neighbor alone cannot express. Every
    first-block amplitude is rescaled by
    :math:`m_{ijc}=1+\chi_k(r_{ij})\tanh a_{ijc}`, which lies in
    :math:`(0,2)`, so a neighbor reweights its message but cannot invert it.

    The keys of the neighbors also enter the non-scalar moments of the
    center, which receive the directions of the second shell through an
    envelope-weighted softmax of the capped logits of the leading :math:`H`
    channels with a null mass :math:`\zeta_{ih}` that the center sets:

    .. math::

       \Delta X_i^{(\ell)}=\sum_h\frac{\sum_j\mu_{ijh}\rho_{ijh}K_{jh}^{(\ell)}}
       {\zeta_{ih}+\sum_j\mu_{ijh}}O_h^{(\ell)},\qquad
       \mu_{ijh}=\chi_k^2\chi_{ij}^2e^{c\tanh(a_{ijh}/c)},

    with a radial weight :math:`\rho_{ijh}` of the saturated amplitude. Every
    learned map reads saturated inputs, so close contacts, which drive the
    moments, the invariants and the amplitude far outside the sampled range,
    cannot drive the maps outside the range they were trained on.

    Every output map starts at zero, so a block starts as the identity, and
    every term vanishes with the C³ envelope :math:`\chi_k` of the block at
    its cutoff. Under zone bridging the first-block envelope :math:`\chi_{ij}`
    carries the source gate, which removes a frozen atom from the numerator
    and the denominator of the softmax alike. Source features are formed per
    node by :meth:`source_features` and center features by
    :meth:`center_features`; a distributed caller refreshes the ghost rows of
    the source features in between.

    Parameters
    ----------
    degree_channels
        Channel widths of degrees zero through ``lmax``.
    descriptor_dim
        Width of the calibrated invariants that the encoders read.
    precision
        Parameter precision.
    trainable
        Whether the parameters are trainable.
    seed
        Random seed reserved for the block.
    """

    #: Number :math:`H` of heads, each with its own keys, field gains and
    #: softmax; amplitude channel :math:`c` follows head :math:`c \bmod H`.
    HEADS = 2
    #: Hidden width of the state encoder.
    STATE_WIDTH = 16
    #: Knee of the saturation of the calibrated invariants.
    STATE_KNEE = 2.0
    #: Knee :math:`\kappa` of the saturation of the keys.
    KEY_KNEE = 1.5
    #: Knee of the saturation of the amplitude that the learned maps read.
    AMPLITUDE_KNEE = 1.5
    #: Cap :math:`c` of the softmax logit.
    LOGIT_CAP = 2.0
    #: Floor of the null mass of the softmax.
    NULL_FLOOR = 1.0e-6
    #: Sign tables that the constructor rebuilds from the configuration, kept
    #: out of the state dictionary.
    CONFIG_DERIVED_ARRAYS = ("head_index", "reflection_sign")

    def __init__(
        self,
        degree_channels: list[int],
        descriptor_dim: int,
        *,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        self.degree_channels = [int(width) for width in degree_channels]
        self.descriptor_dim = int(descriptor_dim)
        self.precision = str(precision)
        self.trainable = bool(trainable)
        self.degree_offsets = degree_offsets(self.degree_channels)
        dtype = PRECISION_DICT[self.precision]
        channels = self.degree_channels[0]
        n_degree = len(self.degree_channels) - 1
        heads = self.HEADS
        seeds = itertools.count()

        def layer(n_in: int, n_out: int, zero: bool) -> NativeLayer:
            built = NativeLayer(
                n_in,
                n_out,
                bias=False,
                precision=self.precision,
                trainable=self.trainable,
                seed=child_seed(seed, next(seeds)),
            )
            if zero:
                built.w = np.zeros((n_in, n_out), dtype=dtype)
            return built

        # === Step 1. Keys with orthonormal columns where the width allows ===
        self.key_projections = []
        for degree in range(1, n_degree + 1):
            width = self.degree_channels[degree]
            projection = layer(width, heads, zero=False)
            draw = np.random.default_rng(child_seed(seed, next(seeds))).normal(
                size=(width, heads)
            )
            if width >= heads:
                draw = np.linalg.qr(draw)[0]
            else:
                draw /= np.linalg.norm(draw, axis=0, keepdims=True)
            projection.w = draw.astype(dtype)
            self.key_projections.append(projection)

        # === Step 2. Logit terms of the neighbor ===
        self.adam_field_gain = np.zeros((n_degree, heads), dtype=dtype)
        self.state_encoder = SwiGLUMLP(
            [self.descriptor_dim, self.STATE_WIDTH, channels],
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, next(seeds)),
        )
        self.state_encoder.layers[-1].w = np.zeros(
            (self.STATE_WIDTH, channels), dtype=dtype
        )
        self.hidden_state = layer(self.descriptor_dim, channels, zero=False)
        self.hidden_field = layer(n_degree * heads, channels, zero=False)
        self.hidden_amplitude = layer(channels, channels, zero=False)
        self.hidden_output = layer(channels, channels, zero=True)

        # === Step 3. Transport of the neighbor keys ===
        self.transport_radial = layer(channels, heads, zero=False)
        self.transport_outputs = [
            layer(heads, self.degree_channels[degree], zero=True)
            for degree in range(1, n_degree + 1)
        ]
        self.null_mass = NativeLayer(
            self.descriptor_dim,
            heads,
            bias=True,
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, next(seeds)),
        )
        self.null_mass.w = np.zeros((self.descriptor_dim, heads), dtype=dtype)
        # softplus(0.5413) = 1 gives the null the mass of one full edge.
        self.null_mass.b = np.full(heads, 0.5413, dtype=dtype)

        # === Step 4. Head assignment and reflection signs ===
        # Amplitude channel c follows head c mod H. The fields read the bond
        # towards the center, and B_l(-u) = (-1)^l B_l(u).
        self.head_index = np.arange(channels, dtype=np.int64) % heads
        self.reflection_sign = np.concatenate(
            [
                np.full(2 * degree + 1, (-1.0) ** degree, dtype=dtype)
                for degree in range(1, n_degree + 1)
            ]
        )

    @property
    def harmonic_width(self) -> int:
        """Return the number of non-scalar harmonic components ``(L + 1)^2 - 1``."""
        return len(self.degree_channels) ** 2 - 1

    @property
    def key_width(self) -> int:
        """Return the width of the keys in the source features."""
        return self.harmonic_width * self.HEADS

    # === Node features ===

    def source_features(self, moments: Any, invariants: Any) -> Any:
        """Build the per-node features that every edge reads from its source.

        Parameters
        ----------
        moments
            Normalized moments of the previous block with shape ``(N, S)``;
            columns past the geometric degrees are not read.
        invariants
            Calibrated invariants of the previous block with shape
            ``(N, descriptor_dim)``.

        Returns
        -------
        Any
            Source features with shape ``(N, key_width + 2 * C_0)``: the keys,
            harmonic-major with the heads innermost, the state and the
            neighbor term of the hidden layer.
        """
        xp = array_api_compat.array_namespace(moments)
        n_node = moments.shape[0]
        blocks = []
        for degree, projection in enumerate(self.key_projections, start=1):
            start, end = self.degree_offsets[degree : degree + 2]
            block = projection.call(
                xp.reshape(
                    moments[:, start:end],
                    (n_node, 2 * degree + 1, self.degree_channels[degree]),
                )
            )
            # The saturation acts on the norm of every head, which keeps its
            # direction and therefore its transformation law.
            norm_squared = xp.sum(block * block, axis=1, keepdims=True)
            blocks.append(
                block / xp.sqrt(1.0 + norm_squared / (self.KEY_KNEE * self.KEY_KNEE))
            )
        keys = xp.reshape(xp.concat(blocks, axis=1), (n_node, self.key_width))
        saturated = soft_saturate(invariants, self.STATE_KNEE)
        return xp.concat(
            [
                keys,
                self.state_encoder.call(saturated),
                self.hidden_state.call(saturated),
            ],
            axis=1,
        )

    def center_features(self, invariants: Any) -> Any:
        """Return the null-mass logits of every center with shape ``(N, H)``."""
        return self.null_mass.call(soft_saturate(invariants, self.STATE_KNEE))

    # === Edge evaluation ===

    def call(
        self,
        sources: Any,
        centers: Any,
        amplitude: Any,
        basis: Any,
        envelope: Any,
        block_envelope: Any,
        src: Any,
        dst: Any,
    ) -> tuple[Any, Any]:
        """Evaluate the rescaled amplitudes and the transported moments.

        Parameters
        ----------
        sources
            Source features of every node with shape ``(N, key_width + 2 * C_0)``.
        centers
            Center features of every node with shape ``(N, H)``.
        amplitude
            Enveloped first-block edge amplitudes with shape ``(E, C_0)``.
        basis
            Cartesian harmonics of the edge directions with shape
            ``(E, (lmax + 1) ** 2)``.
        envelope
            Masked C³ envelope of the first block with shape ``(E,)``,
            including the bridging switch and the source gate.
        block_envelope
            C³ envelope at the cutoff of this block with shape ``(E,)``.
        src, dst
            Source and destination node of every edge with shape ``(E,)``.

        Returns
        -------
        amplitude
            Edge amplitudes of this block with shape ``(E, C_0)``.
        transport
            Transported non-scalar moments with shape ``(N, S_geo - C_0)``.
        """
        xp = array_api_compat.array_namespace(sources)
        device = array_api_compat.device(sources)
        heads = self.HEADS
        channels = self.degree_channels[0]
        # Each part of the source features is gathered on its own, so every
        # backward scatters narrow cotangents instead of one zero-padded row
        # of the full width.
        n_edge = src.shape[0]
        key_end = self.key_width
        state_end = key_end + channels
        keys = xp.reshape(
            xp.take(sources[:, :key_end], src, axis=0),
            (n_edge, self.harmonic_width, heads),
        )
        state = xp.take(sources[:, key_end:state_end], src, axis=0)
        hidden = xp.take(sources[:, state_end:], src, axis=0)
        saturated = soft_saturate(amplitude, self.AMPLITUDE_KNEE)

        # === Step 1. Fields of the neighbor along the reversed bond ===
        sign = xp_asarray_nodetach(xp, self.reflection_sign, device=device)
        signed = (basis[:, 1:] * sign)[:, :, None] * keys
        fields = []
        start = 0
        for degree in range(1, len(self.degree_channels)):
            size = 2 * degree + 1
            fields.append(xp.sum(signed[:, start : start + size, :], axis=1))
            start += size
        fields = xp.stack(fields, axis=1)  # (E, L, H)

        # === Step 2. Logit of every amplitude channel ===
        gain = xp_asarray_nodetach(xp, self.adam_field_gain, device=device)
        head_index = xp_asarray_nodetach(xp, self.head_index, device=device)
        flat_fields = xp.reshape(
            fields, (n_edge, (len(self.degree_channels) - 1) * heads)
        )
        hidden = (
            hidden
            + self.hidden_field.call(flat_fields)
            + self.hidden_amplitude.call(saturated)
        )
        logit = (
            xp.take(xp.sum(fields * gain[None], axis=1), head_index, axis=1)
            + state
            + self.hidden_output.call(hidden * xp_sigmoid(hidden))
        )
        rescaled = amplitude * (1.0 + block_envelope[:, None] * xp.tanh(logit))

        # === Step 3. Softmax transport of the neighbor keys ===
        n_node = sources.shape[0]
        weight = envelope * block_envelope
        capped = self.LOGIT_CAP * xp.tanh(logit[:, :heads] / self.LOGIT_CAP)
        mass = (weight * weight)[:, None] * xp.exp(capped)  # (E, H)
        denominator = (
            segment_sum(mass, dst, n_node) + softplus_t(centers) + self.NULL_FLOOR
        )
        weighted = (mass * self.transport_radial.call(saturated))[:, None, :] * keys
        reduced = segment_sum(
            xp.reshape(weighted, (n_edge, self.key_width)), dst, n_node
        )
        reduced = (
            xp.reshape(reduced, (n_node, self.harmonic_width, heads))
            / denominator[:, None, :]
        )
        transported = []
        start = 0
        for degree, output in enumerate(self.transport_outputs, start=1):
            size = 2 * degree + 1
            transported.append(
                xp.reshape(
                    output.call(reduced[:, start : start + size, :]),
                    (n_node, size * self.degree_channels[degree]),
                )
            )
            start += size
        return rescaled, xp.concat(transported, axis=1)

    def serialize(self) -> dict[str, Any]:
        """Serialize the block attention.

        Returns
        -------
        dict[str, Any]
            Versioned configuration and parameters.
        """
        layers: dict[str, Any] = {
            name: [layer.serialize() for layer in getattr(self, name)]
            for name in _LAYER_LISTS
        }
        for name in _SINGLE_LAYERS:
            layers[name] = getattr(self, name).serialize()
        layers["state_encoder"] = self.state_encoder.serialize()
        return {
            "@class": "BlockAttention",
            "@version": 1,
            "degree_channels": list(self.degree_channels),
            "descriptor_dim": self.descriptor_dim,
            "precision": self.precision,
            "trainable": self.trainable,
            "layers": layers,
            "@variables": {"adam_field_gain": to_numpy_array(self.adam_field_gain)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> BlockAttention:
        """Deserialize a :class:`BlockAttention`.

        Parameters
        ----------
        data
            Versioned dictionary produced by :meth:`serialize`.

        Returns
        -------
        BlockAttention
            Reconstructed block with restored parameters.

        Raises
        ------
        ValueError
            If the payload does not describe a :class:`BlockAttention`.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        if data.pop("@class") != "BlockAttention":
            raise ValueError("Invalid serialized class for BlockAttention")
        layers = data.pop("layers")
        variables = data.pop("@variables")
        obj = cls(**data)
        for name in _LAYER_LISTS:
            records = layers[name]
            if len(records) != len(getattr(obj, name)):
                raise ValueError(
                    f"Serialized `{name}` holds {len(records)} layers, expected "
                    f"{len(getattr(obj, name))}."
                )
            setattr(obj, name, [NativeLayer.deserialize(record) for record in records])
        for name in _SINGLE_LAYERS:
            setattr(obj, name, NativeLayer.deserialize(layers[name]))
        obj.state_encoder = SwiGLUMLP.deserialize(layers["state_encoder"])
        obj.adam_field_gain = np.asarray(
            variables["adam_field_gain"], dtype=PRECISION_DICT[obj.precision]
        )
        return obj
