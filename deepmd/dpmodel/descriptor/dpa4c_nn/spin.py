# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Native per-atom spin channels for DPA4C.

The magnetic moment :math:`\mathbf s` is an *axial* vector: under spatial
inversion the edge direction flips while the spin does not, and under time
reversal the spin flips while the geometry does not. Labelling every channel
by its angular degree :math:`\ell` and by its spin order :math:`\sigma`
(modulo two), and restricting every angular coupling to the genuine Gaunt
couplings the descriptor already uses (:math:`\ell_1+\ell_2+\ell_3` even),
gives

.. math::

   p=(-1)^{\ell}(-1)^{\sigma},\qquad t=(-1)^{\sigma},

so the two Z2 conditions decouple into the even-degree rule DPA4C already
enforces and one new rule: **only contractions of even total spin order may be
emitted**. The channel families below are laid out so that this rule is
realized by the block structure itself rather than by a per-entry filter.

Five families are accumulated.

============  ============  ==============  ====================================
Family        :math:`\ell`  :math:`\sigma`  Content
============  ============  ==============  ====================================
``V``         1             1               centre and neighbour spin vectors
``P``         1             1               bond-projected neighbour spins
``Q``         2             0               centre and neighbour spin quadrupoles
``M0``        0             0               neighbour moment magnitude
``Mw``        0             0               magnetic effective coordination
============  ============  ==============  ====================================

``P`` is the one family that reads the edge direction. Each of its channels
accumulates :math:`(\hat{\mathbf s}_j\cdot\hat{\mathbf u}_{ij})
\hat{\mathbf u}_{ij}`, a Cartesian vector carrying one moment and two factors
of the unit bond direction. Spatial inversion flips both direction factors and
leaves the moment alone, so the product is even; time reversal flips the
moment alone, so it is odd. That is exactly the grading of ``V``, which is why
the two occupy one block on the channel axis and the Gram of that block emits
every resulting invariant with no filtering rule of its own. Because ``P``
depends on where a neighbour lies, the spin branch carries an angular
cotangent and its coordinate gradient flows through the direction as well as
through the distance.

Their admissible contractions are the Gram of the joint ``V``/``P`` block
(Heisenberg exchange, symmetric anisotropic two-ion exchange, and the
collective anisotropies), the Gram of ``Q`` (biquadratic exchange), the cross
Gram of ``Q`` against the geometric degree-two moments (single-ion
anisotropy), and ``M0``/``Mw`` emitted directly. The cross Gram of the vector
block against the geometric degree-one moments has odd spin order and is
deliberately absent.

Antisymmetric spin bilinears, and with them the Dzyaloshinskii-Moriya
interaction, remain unreachable at every order: the readout contracts channels
only through symmetric Grams and symmetric Gaunt couplings, so no
antisymmetric pairing of two moments is ever formed.

``Mw`` is the one family that does not read the spin *value*: it is the
radially weighted count of neighbours that carry a magnetic degree of freedom
at all, and it is therefore nonzero even at vanishing spin. It is retained
because every other spin family vanishes with the moments, so without it the
readout cannot distinguish a neighbourhood with no magnetic species from a
magnetic neighbourhood that happens to be demagnetized -- the distinction that
separates a paramagnetic configuration from a non-magnetic one. Being
independent of the spin value, it contributes nothing to the magnetic force.
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
)
from deepmd.dpmodel.common import (
    get_xp_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .geometry import (
    build_angular_basis,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )

#: Number of quadrupole channels reduced over neighbours. One channel is
#: enough for the biquadratic exchange it enables, and each further channel
#: costs five moment accumulators against the three of a vector channel.
NEIGHBOR_QUADRUPOLE_CHANNELS = 1


def derive_spin_channels(degree_channels: Sequence[int]) -> int:
    """Derive the neighbour spin channel width.

    The spin channels index independent radial shapes of the effective
    exchange function, and they read the leading channels of the same shared
    radial map as the geometric degrees. The width is tied to the degree-two
    width, which is the narrowest multi-channel degree of every supported
    profile: that keeps the spin Gram, which grows quadratically, in
    proportion to the rest of the profile, and it guarantees that every spin
    channel addresses an already-evaluated radial channel.

    Parameters
    ----------
    degree_channels
        Channel widths for degrees zero through ``lmax``. The profile must
        reach degree two, which every supported ``lmax`` does.

    Returns
    -------
    int
        Neighbour spin channel width :math:`C_s`.

    Raises
    ------
    ValueError
        If the degree profile does not reach degree two.
    """
    if len(degree_channels) < 3:
        raise ValueError(
            "DPA4C native spin requires a degree profile reaching degree two, "
            f"got {list(degree_channels)}"
        )
    return int(degree_channels[2])


class SpinChannels(NativeOP):
    r"""Accumulate and contract the native spin channels of DPA4C.

    The per-atom spin enters through one conditioned node quantity

    .. math::

       \hat{\mathbf s}_i=\frac{m_{a_i}}{s^{\mathrm{ref}}_{a_i}}\,\mathbf s_i,

    where :math:`m` is the per-type spin mask and :math:`s^{\mathrm{ref}}` the
    per-type reference magnitude. The mask is multiplicative and is applied
    once, so :math:`\partial E/\partial\mathbf s_i` vanishes identically -- at
    every derivative order -- for a type that carries no magnetic degree of
    freedom. Relying on the dataset convention :math:`\mathbf s=0` would not
    achieve this, because the force loss differentiates the magnetic force
    again and therefore probes the spin direction even where the value is
    zero.

    Neighbour contributions share the radial map of the geometric branch and
    add one ordered type-pair cache of their own:

    .. math::

       \phi^{s}_{ij,c}=\chi_{ij}^2\bigl(
         \gamma^{s}_{ab,c}g_c(\rho_{ij})+\beta^{s}_{ab,c}\bigr).

    The squared envelope matches the weight of every non-scalar geometric
    moment, so the neighbour spin families share the existing normalizer
    :math:`n^{(+)}` and need no neighbourhood mass of their own. Unlike the
    geometric scale, :math:`\gamma^{s}` is signed: the exchange interaction of
    an ordered pair may have either sign, and the shared radial map cannot
    supply that sign per pair.

    Parameters
    ----------
    ntypes
        Number of real atom types.
    degree_channels
        Channel widths for degrees zero through ``lmax``.
    use_spin
        Per-type flags marking which atom types carry a magnetic moment. Every
        weight is sized by the type count rather than by the magnetic subset,
        and the flags only build the per-type gate, which is derived from the
        configuration rather than serialized. An all-false mask is therefore a
        supported configuration: it contributes an identical zero and keeps the
        full set of spin parameters, which is what a spin-free pretrain needs in
        order to declare its magnetic types at fine-tune time.
    precision
        Parameter precision.
    trainable
        Whether the on-site weights receive optimizer updates.
    seed
        Random seed.

    Raises
    ------
    ValueError
        If ``use_spin`` does not have one entry per real atom type.
    """

    CONFIG_DERIVED_ARRAYS = ("spin_mask",)

    def __init__(
        self,
        ntypes: int,
        degree_channels: Sequence[int],
        use_spin: Sequence[bool],
        *,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if len(use_spin) != int(ntypes):
            raise ValueError(
                f"`use_spin` must contain {int(ntypes)} entries, got {len(use_spin)}"
            )
        self.ntypes = int(ntypes)
        self.degree_channels = [int(width) for width in degree_channels]
        self.use_spin = [bool(flag) for flag in use_spin]
        self.precision = str(precision)
        self.trainable = bool(trainable)
        self.spin_channels = derive_spin_channels(self.degree_channels)
        precision_dtype = PRECISION_DICT[self.precision.lower()]

        # === Per-type spin gate ===
        # Deterministic from the configuration (hence ``CONFIG_DERIVED_ARRAYS``),
        # so it is rebuilt rather than serialized. The trailing row is the
        # padding type.
        self.spin_mask = np.asarray(
            [1.0 if flag else 0.0 for flag in self.use_spin] + [0.0],
            dtype=precision_dtype,
        )
        # Per-type reference magnitude in the units of the dataset. Measured
        # by the descriptor calibration and therefore persistent; a unit
        # reference leaves the raw spin untouched.
        self.spin_reference = np.ones(self.ntypes + 1, dtype=precision_dtype)

        # === On-site weights ===
        # One channel each, so a per-type scalar. The invariants they enter
        # are linear in this weight, and the fitting network already receives
        # the centre type embedding, so additional on-site channels would only
        # rescale the same quantities.
        rng = np.random.default_rng(child_seed(seed, 0))
        self.adam_spin_vector_weight = rng.normal(
            0.0, 1.0, size=(self.ntypes + 1,)
        ).astype(precision_dtype)
        self.adam_spin_quadrupole_weight = rng.normal(
            0.0, 1.0, size=(self.ntypes + 1,)
        ).astype(precision_dtype)

        # === Branch gate ===
        # One scalar on the whole branch, the only place all of it passes
        # through: the families reach the fitting network by several routes
        # and at two spin orders, so no weight inside the branch gates all of
        # them, and a factor applied to the conditioned moment instead would
        # enter the invariants quadratically and leave zero a stationary
        # point. The descriptor applies it to the CALIBRATED block (see
        # ``DescrptDPA4C._readout``), so a closed gate contributes exactly
        # zero whatever the measured calibration is, and the compressed path
        # carries it as a factor on the inverse deviation alone.
        self.spin_gate = np.zeros((1,), dtype=precision_dtype)

        # Isometric half-vectorization of the two spin Grams. The quadrupole
        # block drops entry zero, its on-site self-term: the harmonic blocks
        # are homogeneous, so |B_2(s)|^2 = |s|^4 exactly and that entry is a
        # per-type constant times the square of the on-site self-term the
        # vector block already emits.
        self.vector_gram_index, self.vector_gram_scale = _half_gram_layout(
            self.vector_width,
            precision_dtype,
        )
        quadrupole_index, quadrupole_scale = _half_gram_layout(
            self.quadrupole_width,
            precision_dtype,
        )
        self.quadrupole_gram_index = quadrupole_index[1:]
        self.quadrupole_gram_scale = quadrupole_scale[1:]

    def call(self, spin_moments: Array, degree_two: Array) -> Array:
        r"""Contract the spin moments into invariants of even spin order.

        Parameters
        ----------
        spin_moments
            Flat spin moments with shape ``(N, moment_width)``, normalized and
            concatenated by the descriptor.
        degree_two
            Geometric degree-two moments with shape ``(N, 5, C_2)``.

        Returns
        -------
        Array
            Invariant spin features with shape ``(N, get_dim_out())``.
        """
        xp = array_api_compat.array_namespace(spin_moments)
        magnitude, coordination, vector, quadrupole = self.split(spin_moments, xp)
        return xp.concat(
            [
                _half_gram(
                    vector,
                    self.vector_gram_index,
                    self.vector_gram_scale,
                    xp,
                ),
                _half_gram(
                    quadrupole,
                    self.quadrupole_gram_index,
                    self.quadrupole_gram_scale,
                    xp,
                ),
                # Cross Gram against the geometric degree-two moments. Both
                # factors have even spin order, so the product is admissible;
                # with a unit direction it evaluates to the single-ion
                # anisotropy sum over neighbours.
                xp.reshape(
                    xp.matmul(xp.permute_dims(quadrupole, (0, 2, 1)), degree_two),
                    (
                        spin_moments.shape[0],
                        self.quadrupole_width * self.degree_channels[2],
                    ),
                ),
                magnitude,
                coordination,
            ],
            axis=-1,
        )

    def split(self, spin_moments: Array, xp: Any) -> tuple[Array, Array, Array, Array]:
        """Split the flat spin moments into their five families.

        Parameters
        ----------
        spin_moments
            Flat spin moments with shape ``(N, moment_width)``.
        xp
            Array namespace associated with ``spin_moments``.

        Returns
        -------
        magnitude
            Neighbour moment magnitudes with shape ``(N, spin_channels)``.
        coordination
            Magnetic effective coordination with shape ``(N, spin_channels)``.
        vector
            Joint degree-one spin block with shape ``(N, 3, vector_width)``,
            holding the on-site moment, the ``V`` channels and the ``P``
            channels in that order.
        quadrupole
            Degree-two spin block with shape ``(N, 5, quadrupole_width)``.
        """
        nodes = spin_moments.shape[0]
        channels = self.spin_channels
        neighbor_quadrupole = NEIGHBOR_QUADRUPOLE_CHANNELS
        offset = 0
        magnitude = spin_moments[:, offset : offset + channels]
        offset += channels
        coordination = spin_moments[:, offset : offset + channels]
        offset += channels
        neighbor_vector = xp.reshape(
            spin_moments[:, offset : offset + 3 * channels],
            (nodes, 3, channels),
        )
        offset += 3 * channels
        neighbor_bond = xp.reshape(
            spin_moments[:, offset : offset + 3 * channels],
            (nodes, 3, channels),
        )
        offset += 3 * channels
        neighbor_tensor = xp.reshape(
            spin_moments[:, offset : offset + 5 * neighbor_quadrupole],
            (nodes, 5, neighbor_quadrupole),
        )
        offset += 5 * neighbor_quadrupole
        onsite_vector = spin_moments[:, offset : offset + 3]
        offset += 3
        onsite_tensor = spin_moments[:, offset : offset + 5]
        # The on-site channel leads each block so that its Gram entries
        # against the neighbour channels, which carry the two-body physics,
        # occupy the first row of the upper triangle.
        return (
            magnitude,
            coordination,
            xp.concat(
                [onsite_vector[:, :, None], neighbor_vector, neighbor_bond],
                axis=-1,
            ),
            xp.concat([onsite_tensor[:, :, None], neighbor_tensor], axis=-1),
        )

    def conditioned_spin(self, spin: Array, atype: Array) -> Array:
        r"""Apply the per-type spin mask and reference magnitude.

        Parameters
        ----------
        spin
            Per-node spin vectors with shape ``(N, 3)``.
        atype
            Flat node types with shape ``(N,)``. Padding nodes use type index
            ``ntypes`` and select the zero row.

        Returns
        -------
        Array
            Conditioned spin :math:`\hat{\mathbf s}` with shape
            ``(N, 3)``, exactly zero for non-magnetic and padding types.
        """
        xp = array_api_compat.array_namespace(spin)
        device = array_api_compat.device(spin)
        dtype = get_xp_precision(xp, self.precision)
        # The gate and the reference collapse into one multiplicative table,
        # which costs one gather instead of two. Every reference entry is
        # strictly positive -- the estimator seeds the table with ones and
        # overwrites only strictly positive measurements, and both setters
        # enforce it -- so the quotient is finite for every type, including
        # the non-magnetic and padding rows whose gate is zero.
        gate = xp_asarray_nodetach(xp, self.spin_mask[...], device=device)
        reference = xp_asarray_nodetach(xp, self.spin_reference[...], device=device)
        weight = xp.astype(gate / reference, dtype)
        index = xp.astype(atype, xp.int64)
        return xp.astype(spin, dtype) * xp.take(weight, index, axis=0)[:, None]

    def onsite_payload(self, conditioned_spin: Array, atype: Array) -> Array:
        r"""Build the node-local on-site spin moments.

        The centre spin enters as the degree-one vector
        :math:`\lambda_{a}\hat{\mathbf s}_i` and the degree-two quadrupole
        :math:`\mu_{a}B_2(\hat{\mathbf s}_i)`. Both are polynomial in the
        spin, hence smooth at :math:`\hat{\mathbf s}=0`, and both are node
        local: they are written after the destination reduction and carry no
        neighbourhood normalizer, so the invariants they enter contain exactly
        one factor of :math:`n^{(+)}` from their neighbour partner.

        Parameters
        ----------
        conditioned_spin
            Conditioned spin with shape ``(N, 3)``.
        atype
            Flat node types with shape ``(N,)``.

        Returns
        -------
        Array
            On-site payload with shape ``(N, node_width)``.
        """
        xp = array_api_compat.array_namespace(conditioned_spin)
        device = array_api_compat.device(conditioned_spin)
        index = xp.astype(atype, xp.int64)
        vector_weight = xp.take(
            xp_asarray_nodetach(xp, self.adam_spin_vector_weight, device=device),
            index,
            axis=0,
        )
        quadrupole_weight = xp.take(
            xp_asarray_nodetach(xp, self.adam_spin_quadrupole_weight, device=device),
            index,
            axis=0,
        )
        quadrupole = build_angular_basis(conditioned_spin, 2)[:, 4:9]
        return xp.concat(
            [
                conditioned_spin * vector_weight[:, None],
                quadrupole * quadrupole_weight[:, None],
            ],
            axis=-1,
        )

    def edge_payload(
        self,
        conditioned_spin: Array,
        atype: Array,
        source: Array,
        direction: Array,
        radial: Array,
        envelope: Array,
        pair_scale: Array,
        pair_shift: Array,
    ) -> Array:
        r"""Build the per-edge spin payload reduced over neighbours.

        Every family shares one edge amplitude, the spin counterpart of the
        geometric :math:`\phi_{ij}`

        .. math::

           \phi^{s}_{ij,c}=\chi_{ij}^2\bigl(
             \gamma^{s}_{ab,c}g_c(\rho_{ij})+\beta^{s}_{ab,c}\bigr),

        so the radial table is read once for the whole spin branch. The
        bond-projected family additionally contracts the unit edge direction,

        .. math::

           \mathbf P_{ij,c}=\phi^{s}_{ij,c}
           (\hat{\mathbf s}_j\cdot\hat{\mathbf u}_{ij})\hat{\mathbf u}_{ij},

        which is what makes the symmetric anisotropic exchange representable.

        Parameters
        ----------
        conditioned_spin
            Conditioned spin with shape ``(N, 3)``.
        atype
            Flat node types with shape ``(N,)``, read for the neighbour spin
            gate of the magnetic-coordination family.
        source
            Source node index of each edge with shape ``(E,)``.
        direction
            Regularized unit edge directions with shape ``(E, 3)``.
        radial
            Shared radial map with shape ``(E, channels)``; the leading
            ``spin_channels`` columns are read.
        envelope
            Masked C3 envelope with shape ``(E,)``.
        pair_scale
            Per-edge ordered spin scales with shape ``(E, spin_channels)``.
        pair_shift
            Per-edge ordered spin shifts with the same shape.

        Returns
        -------
        Array
            Edge payload with shape ``(E, edge_width)``.
        """
        xp = array_api_compat.array_namespace(conditioned_spin)
        device = array_api_compat.device(conditioned_spin)
        channels = self.spin_channels
        neighbor_spin = xp.take(conditioned_spin, source, axis=0)  # (E, 3)
        spin_amplitude = (radial[:, :channels] * pair_scale + pair_shift) * (
            envelope * envelope
        )[:, None]
        neighbor_gate = xp.take(
            xp.take(
                xp_asarray_nodetach(xp, self.spin_mask, device=device),
                xp.astype(atype, xp.int64),
                axis=0,
            ),
            source,
            axis=0,
        )[:, None]  # (E, 1)

        magnitude = xp.sum(neighbor_spin * neighbor_spin, axis=-1, keepdims=True)
        # Component of the neighbour moment along the bond, carried back as a
        # vector so that the block Gram turns it into the bond-resolved
        # invariants. The masked envelope already zeroes excluded edges, so the
        # unmasked direction cannot leak through the amplitude.
        bond_spin = direction * xp.sum(
            neighbor_spin * direction,
            axis=-1,
            keepdims=True,
        )  # (E, 3)
        quadrupole = build_angular_basis(neighbor_spin, 2)[:, 4:9]  # (E, 5)
        return xp.concat(
            [
                spin_amplitude * magnitude,
                spin_amplitude * neighbor_gate,
                xp.reshape(
                    neighbor_spin[:, :, None] * spin_amplitude[:, None, :],
                    (-1, 3 * channels),
                ),
                xp.reshape(
                    bond_spin[:, :, None] * spin_amplitude[:, None, :],
                    (-1, 3 * channels),
                ),
                xp.reshape(
                    quadrupole[:, :, None]
                    * spin_amplitude[:, None, :NEIGHBOR_QUADRUPOLE_CHANNELS],
                    (-1, 5 * NEIGHBOR_QUADRUPOLE_CHANNELS),
                ),
            ],
            axis=-1,
        )

    @property
    def vector_width(self) -> int:
        r"""Return the channel width of the joint degree-one spin block.

        The block holds the on-site moment, the :math:`C_s` isotropic
        neighbour channels ``V`` and the :math:`C_s` bond-projected neighbour
        channels ``P``. ``P`` is given the full spin width rather than a
        narrower one because the readout is linear in the emitted Gram
        entries: the effective radial profile of an interaction is the span of
        the channel amplitudes the fitting network mixes, so a narrower ``P``
        would confine the symmetric anisotropic exchange to a smaller function
        space than the isotropic exchange beside it and than the single-ion
        anisotropy, which already reaches :math:`C_2=C_s` geometric channels
        through its cross Gram. Reusing the leading :math:`C_s` amplitudes
        costs no extra radial evaluation.
        """
        return 1 + 2 * self.spin_channels

    @property
    def quadrupole_width(self) -> int:
        """Return the channel width of the degree-two spin block."""
        return 1 + NEIGHBOR_QUADRUPOLE_CHANNELS

    @property
    def edge_width(self) -> int:
        """Return the width of the per-edge payload reduced over neighbours.

        The layout is ``[M0, Mw, V_neighbor, P_neighbor, Q_neighbor]`` with
        the harmonic component as the outer axis of each non-scalar family,
        matching the geometric moment convention.
        """
        return (
            2 * self.spin_channels
            + 6 * self.spin_channels
            + 5 * NEIGHBOR_QUADRUPOLE_CHANNELS
        )

    @property
    def node_width(self) -> int:
        """Return the width of the node-local on-site payload ``[V_o, Q_o]``."""
        return 3 + 5

    @property
    def moment_width(self) -> int:
        """Return the total spin moment width appended to the flat state."""
        return self.edge_width + self.node_width

    def get_dim_out(self) -> int:
        """Return the invariant width contributed by the spin channels.

        Returns
        -------
        int
            Upper triangles of the two spin Grams, the full cross Gram against
            the geometric degree-two moments, and the two scalar families.
        """
        return (
            int(self.vector_gram_index.shape[0])
            + int(self.quadrupole_gram_index.shape[0])
            + self.quadrupole_width * self.degree_channels[2]
            + 2 * self.spin_channels
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize the spin channels.

        Returns
        -------
        dict[str, Any]
            Versioned configuration and persistent arrays. The per-type mask
            is deterministic from ``use_spin`` and is therefore rebuilt.
        """
        return {
            "@class": "SpinChannels",
            "@version": 1,
            "ntypes": self.ntypes,
            "degree_channels": list(self.degree_channels),
            "use_spin": list(self.use_spin),
            "precision": self.precision,
            "trainable": self.trainable,
            "@variables": {
                "spin_reference": to_numpy_array(self.spin_reference),
                "adam_spin_vector_weight": to_numpy_array(self.adam_spin_vector_weight),
                "adam_spin_quadrupole_weight": to_numpy_array(
                    self.adam_spin_quadrupole_weight
                ),
                "spin_gate": to_numpy_array(self.spin_gate),
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> SpinChannels:
        """Deserialize a :class:`SpinChannels`.

        Parameters
        ----------
        data
            Versioned dictionary produced by :meth:`serialize`.

        Returns
        -------
        SpinChannels
            Reconstructed spin channels.

        Raises
        ------
        ValueError
            If the payload does not describe a :class:`SpinChannels`.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        if data.pop("@class") != "SpinChannels":
            raise ValueError("Invalid serialized class for SpinChannels")
        variables = dict(data.pop("@variables"))
        obj = cls(**data)
        obj.set_variables(variables)
        return obj

    def set_variables(self, variables: dict[str, Any]) -> None:
        """Restore the persistent arrays.

        Parameters
        ----------
        variables
            Mapping produced by the ``@variables`` block of :meth:`serialize`.
        """
        precision_dtype = PRECISION_DICT[self.precision.lower()]
        # Routed through the setter so a restored checkpoint cannot weaken the
        # strictly-positive reference invariant that ``conditioned_spin`` relies on.
        self.set_spin_reference(variables["spin_reference"])
        self.adam_spin_vector_weight = np.asarray(
            variables["adam_spin_vector_weight"], dtype=precision_dtype
        )
        self.adam_spin_quadrupole_weight = np.asarray(
            variables["adam_spin_quadrupole_weight"], dtype=precision_dtype
        )
        self.spin_gate = np.asarray(variables["spin_gate"], dtype=precision_dtype)

    def set_spin_reference(self, reference: np.ndarray) -> None:
        """Store the per-type reference magnitudes.

        Parameters
        ----------
        reference
            Reference magnitudes with shape ``(ntypes + 1,)`` in the units of
            the dataset spin. Every entry must be strictly positive, including
            the non-magnetic and padding rows, whose reference is unused but
            still divides the zero gate.

        Raises
        ------
        ValueError
            If the shape is wrong or any entry is not strictly positive.
        """
        reference = np.asarray(to_numpy_array(reference), dtype=np.float64)
        if reference.shape != (self.ntypes + 1,):
            raise ValueError(
                "DPA4C spin reference must have shape "
                f"{(self.ntypes + 1,)}, got {reference.shape}"
            )
        if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
            raise ValueError(
                "DPA4C spin reference must be finite and strictly positive, "
                f"got {reference.tolist()}"
            )
        self.spin_reference = reference.astype(PRECISION_DICT[self.precision.lower()])


def _half_gram_layout(
    width: int,
    precision_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the isometric upper-triangular Gram index and scale.

    The entries are ordered row by row, so entry zero is always the self-term
    of the leading channel and a caller that does not emit it drops the first
    element of both arrays.

    Parameters
    ----------
    width
        Channel width of the block.
    precision_dtype
        Element type of the emitted scale.

    Returns
    -------
    index
        Flattened Gram coordinates of the upper triangle.
    scale
        Matching isometric scales, ``sqrt(2)`` off the diagonal.
    """
    row, column = np.triu_indices(int(width))
    return (
        (row * int(width) + column).astype(np.int64),
        np.where(row == column, 1.0, math.sqrt(2.0)).astype(precision_dtype),
    )


def _half_gram(
    block: Array,
    index: np.ndarray,
    scale: np.ndarray,
    xp: Any,
) -> Array:
    """Return the Frobenius-isometric upper-triangular channel Gram."""
    device = array_api_compat.device(block)
    width = block.shape[-1]
    gram = xp.reshape(
        xp.matmul(xp.permute_dims(block, (0, 2, 1)), block),
        (block.shape[0], width * width),
    )
    return (
        xp.take(
            gram,
            xp_asarray_nodetach(xp, index, device=device),
            axis=1,
        )
        * xp_asarray_nodetach(xp, scale, device=device)[None, :]
    )
