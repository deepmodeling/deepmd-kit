# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Frame-level charge and spin-multiplicity conditioning for DPA4C.

The same nuclear geometry can be a cation, a neutral or an anion, and can be
a singlet or a triplet, with genuinely different energies and forces. The
descriptor therefore accepts one integer pair per frame, the total charge
:math:`Q_f` in units of the elementary charge and the spin multiplicity
:math:`M_f`, and turns it into two vectors that reach the two places where
the type embedding enters DPA4C:

- :math:`\mathbf w_f`, added to the centre type embedding, which the
  descriptor emits as its trailing output block;
- :math:`\mathbf y_f`, added to the hidden pre-activation of the ordered
  type-pair encoder, which conditions the scale, shift, mode-mixing and spin
  tables of every ordered pair.

The second route is what makes the conditioning a property of the descriptor
rather than of the fitting network: it changes how a given geometry maps to
the degree-wise moments. It enters as a pre-activation bias rather than as a
shift of the encoder input because the first affine map of that encoder is
linear and bias free, so the two are equivalent while only the bias form
lets one shared projection over the finite type table serve every frame.

Both routes are functions of the ordered type pair and of the condition, and
of nothing that depends on the interatomic distance. Compressed inference can
therefore fold them into its frozen type and ordered-pair tables without
changing a single table shape, at the cost of specializing the snapshot to
one charge state.

This module is unrelated to :mod:`~deepmd.dpmodel.descriptor.dpa4c_nn.spin`,
which carries a per-atom magnetic moment. The two are independent inputs and
may be used together.
"""

from __future__ import (
    annotations,
)

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
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.charge_state import (
    CHARGE_OFFSET,
    CHARGE_TABLE_ROWS,
    MULTIPLICITY_TABLE_ROWS,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from ..dpa4_nn.embedding import (
    SeZMTypeEmbedding,
)
from ..dpa4_nn.mlp import (
    SwiGLUMLP,
    resolve_swiglu_hidden_width,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )


class ChargeStateEmbedding(NativeOP):
    r"""Embed the frame charge and spin multiplicity into two condition vectors.

    The two integers are embedded independently and mixed by one bias-free
    SwiGLU trunk whose single output head is split into the two routes:

    .. math::

       [\mathbf w_f \Vert \mathbf y_f]
       = \operatorname{SwiGLU}\bigl(
           [\,E^{Q}_{Q_f}\Vert E^{M}_{M_f}\,]W_{\rm in}
         \bigr)W_{\rm out}.

    Charge and multiplicity share one nonlinear pathway rather than
    contributing two additive embeddings. They are not independent degrees of
    freedom: the number of unpaired electrons has the parity of the electron
    count, so changing the charge by one flips it, and the structural
    response to a spin-state change depends on the oxidation state it happens
    in. An additive decomposition can represent neither coupling.

    The output projection is zero initialized, so an untrained descriptor is
    independent of the condition for every value of it. This keeps the fixed
    output calibration, which is measured once before training, free of a
    random condition offset.

    Parameters
    ----------
    channels
        Width :math:`C_0` of the centre type embedding, and of each of the
        two integer embedding tables.
    pair_hidden_width
        Width :math:`2H_{\rm pair}` of the ordered pair encoder's hidden
        pre-activation.
    precision
        Parameter precision.
    trainable
        Whether the condition parameters receive optimizer updates.
    seed
        Random seed.

    Raises
    ------
    ValueError
        If ``channels`` or ``pair_hidden_width`` is not positive.
    """

    def __init__(
        self,
        channels: int,
        pair_hidden_width: int,
        *,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if channels <= 0:
            raise ValueError(f"`channels` must be positive, got {channels}")
        if pair_hidden_width <= 0:
            raise ValueError(
                f"`pair_hidden_width` must be positive, got {pair_hidden_width}"
            )
        self.channels = int(channels)
        self.pair_hidden_width = int(pair_hidden_width)
        self.precision = str(precision)
        self.trainable = bool(trainable)

        self.charge_embedding = SeZMTypeEmbedding(
            ntypes=CHARGE_TABLE_ROWS,
            embed_dim=self.channels,
            precision=self.precision,
            seed=child_seed(seed, 0),
            trainable=self.trainable,
            padding=False,
        )
        self.spin_embedding = SeZMTypeEmbedding(
            ntypes=MULTIPLICITY_TABLE_ROWS,
            embed_dim=self.channels,
            precision=self.precision,
            seed=child_seed(seed, 1),
            trainable=self.trainable,
            padding=False,
        )
        hidden_width = resolve_swiglu_hidden_width(self.channels)
        output_width = self.channels + self.pair_hidden_width
        self.network = SwiGLUMLP(
            [2 * self.channels, hidden_width, output_width],
            precision=self.precision,
            trainable=self.trainable,
            seed=child_seed(seed, 2),
        )
        # ``NativeLayer`` has no zero initializer; replicate it by overwriting
        # the output projection, which leaves the condition inert until the
        # optimizer moves it.
        self.network.layers[-1].w = np.zeros(
            (hidden_width, output_width),
            dtype=PRECISION_DICT[self.precision.lower()],
        )

    def call(self, charge_spin: Array) -> tuple[Array, Array]:
        """Embed the frame conditions into the two condition vectors.

        Parameters
        ----------
        charge_spin
            Frame conditions with shape ``(nf, 2)``, holding the total charge
            and the spin multiplicity as exactly representable integers.

        Returns
        -------
        type_shift
            Centre type-embedding shift with shape ``(nf, channels)``.
        pair_hidden_bias
            Ordered pair encoder pre-activation bias with shape
            ``(nf, pair_hidden_width)``.
        """
        xp = array_api_compat.array_namespace(charge_spin)
        charge = xp.astype(charge_spin[:, 0], xp.int64) + CHARGE_OFFSET
        multiplicity = xp.astype(charge_spin[:, 1], xp.int64)
        logits = self.network.call(
            xp.concat(
                (self.charge_embedding(charge), self.spin_embedding(multiplicity)),
                axis=-1,
            )
        )
        return logits[:, : self.channels], logits[:, self.channels :]

    def serialize(self) -> dict[str, Any]:
        """Serialize the condition embedding.

        Returns
        -------
        dict[str, Any]
            Versioned configuration and the nested embedding tables and trunk.
        """
        return {
            "@class": "ChargeStateEmbedding",
            "@version": 1,
            "channels": self.channels,
            "pair_hidden_width": self.pair_hidden_width,
            "precision": self.precision,
            "trainable": self.trainable,
            "charge_embedding": self.charge_embedding.serialize(),
            "spin_embedding": self.spin_embedding.serialize(),
            "network": self.network.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> ChargeStateEmbedding:
        """Deserialize a :class:`ChargeStateEmbedding`.

        Parameters
        ----------
        data
            Versioned dictionary produced by :meth:`serialize`.

        Returns
        -------
        ChargeStateEmbedding
            Reconstructed condition embedding.

        Raises
        ------
        ValueError
            If the payload does not describe a :class:`ChargeStateEmbedding`.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        if data.pop("@class") != "ChargeStateEmbedding":
            raise ValueError("Invalid serialized class for ChargeStateEmbedding")
        charge_embedding = data.pop("charge_embedding")
        spin_embedding = data.pop("spin_embedding")
        network = data.pop("network")
        obj = cls(**data)
        obj.charge_embedding = SeZMTypeEmbedding.deserialize(charge_embedding)
        obj.spin_embedding = SeZMTypeEmbedding.deserialize(spin_embedding)
        obj.network = SwiGLUMLP.deserialize(network)
        return obj


def canonicalize_charge_spin(
    charge_spin: Array | None,
    default: list[float] | None,
    *,
    nf: int,
    ref: Array,
) -> Array:
    """Bring a frame-condition argument to the canonical ``(nf, 2)`` form.

    Parameters
    ----------
    charge_spin
        Frame conditions supplied by the caller, or ``None`` to fall back to
        ``default``. A single pair is broadcast over the frame axis.
    default
        Configured fallback ``[charge, multiplicity]``, or ``None`` when the
        descriptor requires an explicit condition.
    nf
        Number of frames.
    ref
        Reference array of the caller's compute context. The array namespace,
        dtype and device are taken from it; deriving the namespace from the
        NumPy ``default`` instead would break every non-NumPy backend.

    Returns
    -------
    Array
        Frame conditions with shape ``(nf, 2)``.

    Raises
    ------
    ValueError
        If no condition is available, or if the supplied condition does not
        have shape ``(nf, 2)`` or a shape broadcastable to it.
    """
    xp = array_api_compat.array_namespace(ref)
    if charge_spin is None:
        if default is None:
            raise ValueError(
                "A charge-conditioned DPA4C requires a frame `charge_spin`. "
                "Set `default_chg_spin` to supply a fallback."
            )
        charge_spin = xp.reshape(
            xp.asarray(
                np.asarray(default),
                dtype=ref.dtype,
                device=array_api_compat.device(ref),
            ),
            (1, 2),
        )
    else:
        charge_spin = xp.astype(xp.reshape(charge_spin, (-1, 2)), ref.dtype)
    if charge_spin.shape[0] == 1 and nf != 1:
        return xp.broadcast_to(charge_spin, (nf, 2))
    if charge_spin.shape[0] != nf:
        raise ValueError(
            f"`charge_spin` must hold one [charge, multiplicity] pair per "
            f"frame, expected {nf} rows, got {charge_spin.shape[0]}"
        )
    return charge_spin
