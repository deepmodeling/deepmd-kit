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
from deepmd.dpmodel.utils.network import (
    LayerNorm,
    NativeLayer,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
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


class PairDistanceHead(NativeOP):
    r"""Predict clean pairwise distances without a pair representation.

    Uni-Mol reads this from its pair channel, which carries a vector per ordered
    atom pair. DPA4 has no such channel, so the pair is described by its two
    endpoints instead -- representations that have already exchanged information
    through the backbone's message passing. The combination is symmetric in the
    two endpoints, because a distance is:

    .. math::

        p_{ij} = \big[\, h_i + h_j \;\|\; h_i \odot h_j \,\big]

    Which pairs are covered is a choice, and it is the one place this objective
    departs from Uni-Mol by construction:

    ``neighbour``
        Only pairs inside the backbone's neighbour list. Cheap, and it reuses
        the structure the backbone already built, which is how the rest of
        deepmd trains. On drug-like molecules a 6 Angstrom cut-off holds about
        half of all pairs, so the objective sees less than Uni-Mol's.
    ``all_pairs``
        Every ordered pair, which is Uni-Mol's own coverage and what to use to
        reproduce its training. Costs O(nloc^2) and ignores the neighbour list,
        so it does not share the backbone's locality.

    Parameters
    ----------
    dim_descrpt : int
        Width of the per-atom representation.
    hidden : int
        Width of the head's hidden layer.
    coverage : str
        ``neighbour`` or ``all_pairs``.
    activation_function : str
        Activation of the head.
    precision : str
        Floating-point precision of the parameters.
    seed : int, optional
        Random seed for initialization.
    """

    COVERAGES = ("neighbour", "all_pairs")

    def __init__(
        self,
        dim_descrpt: int,
        hidden: int = 64,
        coverage: str = "neighbour",
        activation_function: str = "gelu_erf",
        precision: str = "float64",
        seed: int | list[int] | None = None,
    ) -> None:
        if coverage not in self.COVERAGES:
            raise ValueError(
                f"unknown coverage {coverage!r}; it must be one of "
                f"{', '.join(map(repr, self.COVERAGES))}"
            )
        self.dim_descrpt = int(dim_descrpt)
        self.hidden = int(hidden)
        self.coverage = coverage
        self.activation_function = activation_function
        self.precision = precision
        self.dense = NativeLayer(
            2 * self.dim_descrpt,
            self.hidden,
            activation_function=activation_function,
            precision=precision,
            seed=child_seed(seed, 0),
        )
        self.layer_norm = LayerNorm(
            self.hidden, precision=precision, seed=child_seed(seed, 1)
        )
        self.out_proj = NativeLayer(
            self.hidden, 1, bias=True, precision=precision, seed=child_seed(seed, 2)
        )

    def call(self, node_ebd: Array, nlist: Array) -> tuple[Array, Array]:
        """Predict a distance per covered pair.

        Parameters
        ----------
        node_ebd : Array
            Per-atom representation, shape ``(nf, nloc, dim_descrpt)``.
        nlist : Array
            Neighbour list, shape ``(nf, nloc, nnei)``. Negative entries are
            padding. Read only when the coverage is ``neighbour``.

        Returns
        -------
        pair_dist : Array
            Predicted distances, shape ``(nf, nloc, nloc)``.
        pair_mask : Array
            1 where the pair is covered and is not the diagonal, else 0, same
            shape. The objective averages over these entries only.
        """
        xp = array_api_compat.array_namespace(node_ebd)
        nf, nloc = node_ebd.shape[0], node_ebd.shape[1]
        dev = array_api_compat.device(node_ebd)

        summed = node_ebd[:, :, None, :] + node_ebd[:, None, :, :]
        product = node_ebd[:, :, None, :] * node_ebd[:, None, :, :]
        pair = xp.concat([summed, product], axis=-1)
        predicted = self.out_proj(self.layer_norm(self.dense(pair)))
        predicted = xp.reshape(predicted, (nf, nloc, nloc))

        eye = xp.eye(nloc, dtype=node_ebd.dtype, device=dev)[None, :, :]
        mask = xp.ones((nf, nloc, nloc), dtype=node_ebd.dtype, device=dev) - eye
        if self.coverage == "neighbour":
            mask = mask * _neighbour_mask(nlist, nloc, node_ebd.dtype)
        return predicted, mask

    def serialize(self) -> dict:
        """Serialize the head."""
        return {
            "@class": "PairDistanceHead",
            "@version": 1,
            "dim_descrpt": self.dim_descrpt,
            "hidden": self.hidden,
            "coverage": self.coverage,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "dense": self.dense.serialize(),
            "layer_norm": self.layer_norm.serialize(),
            "out_proj": self.out_proj.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "PairDistanceHead":
        """Deserialize the head."""
        data = data.copy()
        data.pop("@class", None)
        data.pop("@version", None)
        parts = {k: data.pop(k) for k in ("dense", "layer_norm", "out_proj")}
        obj = cls(**data)
        obj.dense = NativeLayer.deserialize(parts["dense"])
        obj.layer_norm = LayerNorm.deserialize(parts["layer_norm"])
        obj.out_proj = NativeLayer.deserialize(parts["out_proj"])
        return obj


def _neighbour_mask(nlist: Array, nloc: int, dtype) -> Array:  # noqa: ANN001
    """Scatter a padded neighbour list into a dense ``(nf, nloc, nloc)`` mask."""
    xp = array_api_compat.array_namespace(nlist)
    dev = array_api_compat.device(nlist)
    # Padding is negative and would wrap when used as an index, so it is sent
    # to a scratch column that is dropped again below.
    safe = xp.where(nlist >= 0, nlist, xp.zeros_like(nlist) + nloc)
    onehot = xp.astype(
        xp.arange(nloc + 1, dtype=nlist.dtype, device=dev)[None, None, None, :]
        == safe[..., None],
        dtype,
    )
    return xp.astype(xp.sum(onehot, axis=2)[:, :, :nloc] > 0, dtype)
