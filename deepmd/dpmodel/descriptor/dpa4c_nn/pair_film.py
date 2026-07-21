# SPDX-License-Identifier: LGPL-3.0-or-later
"""Ordered type-pair FiLM cache for DPA4C."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    NativeOP,
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
       [s_{ab},d_{ab},u_{ab}]&=0.1\,h_{ab}W_{\rm out},\\
       \gamma_{ab}&=1+\tanh(s_{ab}),\\
       \beta_{ab}&=T_a+T_b+\tanh(d_{ab}),\\
       U_{ab}&=\tanh(u_{ab}).

    The network is evaluated over the finite type table rather than over graph
    edges, so compressed inference stores only the three resulting tables.
    Bounding every output keeps the cache well conditioned in ``float32``:
    :math:`\gamma` stays in :math:`(0,2)`, and the residual parts of
    :math:`\beta` and :math:`U` stay in :math:`(-1,1)`.

    Parameters
    ----------
    channels
        Type-embedding and FiLM channel width :math:`C`.
    radial_modes
        Number :math:`R` of shared radial mode profiles each ordered pair
        mixes. Zero omits the mixing table.
    precision
        Parameter precision.
    trainable
        Whether the pair-encoder weights are trainable.
    seed
        Random seed.

    Raises
    ------
    ValueError
        If ``channels`` is not positive or ``radial_modes`` is negative.
    """

    _OUTPUT_SCALE = 0.1

    def __init__(
        self,
        channels: int,
        radial_modes: int = 0,
        *,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if channels <= 0:
            raise ValueError(f"`channels` must be positive, got {channels}")
        if radial_modes < 0:
            raise ValueError(f"`radial_modes` must be non-negative, got {radial_modes}")
        self.channels = int(channels)
        self.radial_modes = int(radial_modes)
        self.precision = str(precision)
        self.trainable = bool(trainable)
        input_dim = 2 * self.channels
        self.hidden_dim = resolve_swiglu_hidden_width(input_dim)
        output_dim = self.channels * (2 + self.radial_modes)
        self.network = SwiGLUMLP(
            [input_dim, self.hidden_dim, output_dim],
            output_scale=self._OUTPUT_SCALE,
            precision=self.precision,
            trainable=self.trainable,
            seed=seed,
        )

    def call(self, type_embedding: Any) -> tuple[Any, Any, Any | None]:
        """Build the ordered scale, shift, and mixing tables.

        Parameters
        ----------
        type_embedding
            Complete type table with shape ``(T + 1, channels)``, where the
            trailing row is the zero padding type.

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
        logits = self.network.call(pair_input)

        # The output splits into the scale, the shift residual, and the
        # flattened mixing matrix, in that order.
        shift_end = 2 * self.channels
        base_shift = xp.reshape(
            type_embedding[:, None, :] + type_embedding[None, :, :],
            (-1, self.channels),
        )
        return (
            1.0 + xp.tanh(logits[:, : self.channels]),
            base_shift + xp.tanh(logits[:, self.channels : shift_end]),
            None
            if self.radial_modes == 0
            else xp.reshape(
                xp.tanh(logits[:, shift_end:]),
                (-1, self.channels, self.radial_modes),
            ),
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the ordered pair encoder.

        Returns
        -------
        dict[str, Any]
            Versioned configuration and pair-encoder parameters.
        """
        return {
            "@class": "OrderedPairFiLM",
            "@version": 1,
            "channels": self.channels,
            "radial_modes": self.radial_modes,
            "precision": self.precision,
            "trainable": self.trainable,
            "network": self.network.serialize(),
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
        obj = cls(**data)
        obj.network = SwiGLUMLP.deserialize(network)
        return obj
