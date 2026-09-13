# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol pretraining heads that read a DPA backbone.

Uni-Mol's own heads read a pair representation, which its transformer carries
and a DPA backbone does not: DPA4 attends per edge with a scatter softmax and
returns no pair axis at all. These heads read what DPA4 does expose instead --
the l=0 descriptor, and the equivariant state behind it.
"""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
    SO3Linear,
)


def l1_to_cartesian(l1: Array) -> Array:
    r"""Turn an l=1 block in SeZM's packed basis into a Cartesian vector.

    The three rows of the l=1 block are not :math:`(x, y, z)`. Reading them as
    such gives a quantity that does not rotate with the input, which is the
    whole point of taking them. The mapping below is the one that does:

    .. math::

        (x, y, z) = (-c_2,\; c_0,\; c_1)

    It was established by rotating the input by a random :math:`R \in SO(3)` and
    asking which of the 48 signed permutations of the three rows satisfies
    :math:`v(Rx) = R\,v(x)`. Exactly one does, to 1.7e-14 in float64; the next
    best is wrong by 0.64, so there is no ambiguity. An overall sign is not
    determined by that test and does not matter, because the projection feeding
    this is learned and absorbs it.

    Parameters
    ----------
    l1 : Array
        The l=1 rows, shape ``(..., 3)``.

    Returns
    -------
    Array
        Cartesian vectors, shape ``(..., 3)``.
    """
    xp = array_api_compat.array_namespace(l1)
    return xp.stack([-l1[..., 2], l1[..., 0], l1[..., 1]], axis=-1)


class EquivariantCoordHead(NativeOP):
    r"""Predict a per-atom coordinate update from a DPA backbone's l=1 state.

    Uni-Mol predicts the update from its pair channel, weighting each
    :math:`x_j - x_i` by a learned scalar. A DPA backbone has no pair channel,
    so the update is read from the l=1 part of the equivariant state instead,
    which is a per-atom vector by construction. The projection is degree-wise
    and shares its weights across the three :math:`m` components, so the result
    rotates with the input rather than merely being three numbers.

    Parameters
    ----------
    lmax : int
        Degree of the state being read; ``node_readout_lmax`` of the descriptor.
    channels : int
        Channel width of the state.
    precision : str
        Floating-point precision of the parameters.
    seed : int, optional
        Random seed for initialization.
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        if lmax < 1:
            raise ValueError(
                f"a coordinate head needs the l=1 block, so the backbone must "
                f"read out at least degree 1; got lmax={lmax}"
            )
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.precision = precision
        self.proj = SO3Linear(
            lmax=self.lmax,
            in_channels=self.channels,
            out_channels=1,
            n_focus=1,
            precision=precision,
            seed=seed,
        )

    def call(self, latent: Array) -> Array:
        """Read one Cartesian vector per atom.

        Parameters
        ----------
        latent : Array
            The descriptor's equivariant state, shape
            ``(nf * nloc, (lmax + 1) ** 2, 1, channels)``.

        Returns
        -------
        Array
            Per-atom update, shape ``(nf * nloc, 3)``.
        """
        projected = self.proj(latent)
        return l1_to_cartesian(projected[:, 1:4, 0, 0])

    def serialize(self) -> dict:
        """Serialize the head."""
        return {
            "@class": "EquivariantCoordHead",
            "@version": 1,
            "lmax": self.lmax,
            "channels": self.channels,
            "precision": self.precision,
            "proj": self.proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "EquivariantCoordHead":
        """Deserialize the head."""
        data = data.copy()
        data.pop("@class", None)
        data.pop("@version", None)
        proj = data.pop("proj")
        obj = cls(**data)
        obj.proj = SO3Linear.deserialize(proj)
        return obj
