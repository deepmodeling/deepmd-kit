# SPDX-License-Identifier: LGPL-3.0-or-later
"""O(3)-invariant Cartesian bispectrum coupling tables."""

from __future__ import (
    annotations,
)

import functools
import itertools
import math
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np

from deepmd.dpmodel.utils.lebedev import (
    load_lebedev_rule,
)

from .geometry import (
    MAX_ANGULAR_DEGREE,
    build_angular_basis,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

DegreeTriple = tuple[int, int, int]


@dataclass(frozen=True)
class BispectrumLayout:
    """Store flattened coupling and independent probe-index tables."""

    degree_triples: tuple[DegreeTriple, ...]
    coupling: np.ndarray
    coupling_offsets: tuple[int, ...]
    probe_index: np.ndarray
    probe_scale: np.ndarray
    probe_offsets: tuple[int, ...]

    @property
    def dim_out(self) -> int:
        """Return the number of independent probe contractions.

        Returns
        -------
        int
            Total bispectrum feature count.
        """
        return int(self.probe_index.size)


def enumerate_degree_triples(lmax: int) -> tuple[DegreeTriple, ...]:
    """Enumerate non-scalar O(3)-even bispectrum degree triples.

    Parameters
    ----------
    lmax
        Maximum angular degree. Supported values are zero through four.

    Returns
    -------
    tuple[DegreeTriple, ...]
        Sorted triples satisfying the triangle and even-parity conditions.

    Raises
    ------
    ValueError
        If ``lmax`` is outside the supported range.
    """
    if lmax < 0 or lmax > MAX_ANGULAR_DEGREE:
        raise ValueError(
            f"`lmax` must be between 0 and {MAX_ANGULAR_DEGREE}, got {lmax}"
        )
    triples = []
    for degree_1 in range(1, lmax + 1):
        for degree_2 in range(degree_1, lmax + 1):
            for degree_3 in range(degree_2, lmax + 1):
                if degree_3 > degree_1 + degree_2:
                    continue
                if (degree_1 + degree_2 + degree_3) % 2 != 0:
                    continue
                triples.append((degree_1, degree_2, degree_3))
    return tuple(triples)


@functools.lru_cache(maxsize=5)
def _coupling_tables(
    lmax: int,
) -> tuple[tuple[DegreeTriple, ...], tuple[np.ndarray, ...]]:
    """Build unit-norm real Cartesian Gaunt tensors."""
    triples = enumerate_degree_triples(lmax)
    if not triples:
        return triples, ()

    points, weights = load_lebedev_rule(13)
    basis = np.asarray(build_angular_basis(points, lmax), dtype=np.float64)
    couplings = []
    for degree_1, degree_2, degree_3 in triples:
        block_1 = basis[:, degree_1**2 : (degree_1 + 1) ** 2]
        block_2 = basis[:, degree_2**2 : (degree_2 + 1) ** 2]
        block_3 = basis[:, degree_3**2 : (degree_3 + 1) ** 2]
        coupling = np.einsum(
            "n,ni,nj,nk->ijk",
            weights,
            block_1,
            block_2,
            block_3,
            optimize=True,
        )
        coupling = _symmetrize_equal_degrees(
            coupling,
            (degree_1, degree_2, degree_3),
        )
        norm = float(np.linalg.norm(coupling))
        if not math.isfinite(norm) or norm <= 1.0e-14:
            raise ValueError(
                "Degenerate bispectrum coupling for degree triple "
                f"{(degree_1, degree_2, degree_3)}"
            )
        coupling /= norm
        significant = np.flatnonzero(np.abs(coupling) > 1.0e-14)
        if significant.size > 0 and coupling.flat[significant[0]] < 0.0:
            coupling = -coupling
        couplings.append(coupling)
    return triples, tuple(couplings)


def _symmetrize_equal_degrees(
    coupling: np.ndarray,
    degrees: DegreeTriple,
) -> np.ndarray:
    """Enforce permutation symmetry for equal-degree tensor axes."""
    if degrees[0] == degrees[2]:
        permutations = tuple(itertools.permutations((0, 1, 2)))
    elif degrees[0] == degrees[1]:
        permutations = ((0, 1, 2), (1, 0, 2))
    elif degrees[1] == degrees[2]:
        permutations = ((0, 1, 2), (0, 2, 1))
    else:
        return coupling
    return sum(np.transpose(coupling, axes) for axes in permutations) / len(
        permutations
    )


def _probe_entries(
    degrees: DegreeTriple,
    ranks: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build independent flattened probe indices and isometric scales.

    A symmetrized coupling tensor makes the contraction invariant under any
    permutation of the axes that carry equal degrees, so only one
    representative of each orbit is emitted. Scaling that representative by
    the square root of its orbit size keeps the reduced feature vector
    isometric to the full one.
    """
    entries: list[tuple[int, int, int]] = []
    scales: list[float] = []
    rank_1, rank_2, rank_3 = (int(ranks[degree - 1]) for degree in degrees)

    if degrees[0] == degrees[2]:
        for entry in itertools.combinations_with_replacement(
            range(rank_1),
            3,
        ):
            entries.append(entry)
            counts = [entry.count(value) for value in set(entry)]
            multiplicity = math.factorial(3)
            for count in counts:
                multiplicity //= math.factorial(count)
            scales.append(math.sqrt(float(multiplicity)))
    elif degrees[0] == degrees[1]:
        for first, second in itertools.combinations_with_replacement(
            range(rank_1),
            2,
        ):
            scale = 1.0 if first == second else math.sqrt(2.0)
            for third in range(rank_3):
                entries.append((first, second, third))
                scales.append(scale)
    elif degrees[1] == degrees[2]:
        for first in range(rank_1):
            for second, third in itertools.combinations_with_replacement(
                range(rank_2),
                2,
            ):
                entries.append((first, second, third))
                scales.append(1.0 if second == third else math.sqrt(2.0))
    else:
        entries.extend(
            itertools.product(
                range(rank_1),
                range(rank_2),
                range(rank_3),
            )
        )
        scales.extend([1.0] * (rank_1 * rank_2 * rank_3))

    index = np.asarray(
        [
            (first * rank_2 + second) * rank_3 + third
            for first, second, third in entries
        ],
        dtype=np.int64,
    )
    return index, np.asarray(scales, dtype=np.float64)


def build_bispectrum_layout(
    lmax: int,
    ranks: Sequence[int],
) -> BispectrumLayout:
    """Build flattened coupling and independent probe-output tables.

    Parameters
    ----------
    lmax
        Maximum angular degree. Supported values are zero through four.
    ranks
        Effective positive probe widths for degrees one through ``lmax``.

    Returns
    -------
    BispectrumLayout
        Immutable layout arrays used by all array backends.

    Raises
    ------
    ValueError
        If the rank profile does not match ``lmax``.
    """
    triples = enumerate_degree_triples(lmax)
    ranks = tuple(int(rank) for rank in ranks)
    if len(ranks) != lmax:
        raise ValueError(f"`ranks` must contain {lmax} entries, got {len(ranks)}")
    if any(rank <= 0 for rank in ranks):
        raise ValueError(f"`ranks` must be positive, got {ranks}")
    if not triples:
        return BispectrumLayout(
            degree_triples=(),
            coupling=np.empty(0, dtype=np.float64),
            coupling_offsets=(0,),
            probe_index=np.empty(0, dtype=np.int64),
            probe_scale=np.empty(0, dtype=np.float64),
            probe_offsets=(0,),
        )

    _, coupling_tables = _coupling_tables(lmax)
    coupling_offsets = [0]
    probe_offsets = [0]
    coupling_parts = []
    probe_index_parts = []
    probe_scale_parts = []
    for degrees, coupling in zip(triples, coupling_tables, strict=True):
        coupling_parts.append(np.reshape(coupling, (-1,)))
        coupling_offsets.append(coupling_offsets[-1] + coupling.size)
        probe_index, probe_scale = _probe_entries(degrees, ranks)
        probe_index_parts.append(probe_index)
        probe_scale_parts.append(probe_scale)
        probe_offsets.append(probe_offsets[-1] + probe_index.size)

    return BispectrumLayout(
        degree_triples=triples,
        coupling=np.concatenate(coupling_parts),
        coupling_offsets=tuple(coupling_offsets),
        probe_index=np.concatenate(probe_index_parts),
        probe_scale=np.concatenate(probe_scale_parts),
        probe_offsets=tuple(probe_offsets),
    )


def derive_bispectrum_ranks(
    degree_channels: Sequence[int],
) -> list[int]:
    """Derive the fixed degree-wise bispectrum probe ranks.

    The exact degree Gram determines a degree-one block up to the physical
    rotation group, but for degree two it determines the packed coefficients
    only up to O(5), of which the physical rotations form a three-parameter
    subgroup. The cubic contractions resolve the remaining orientation for the
    probed channels alone, so ``K_2`` trades that resolution against the width
    of the cubic and quartic blocks. Raising ``K_2`` to ``C_2`` was measured to
    give a small accuracy gain that does not justify the wider invariant
    output, so the compact profile is fixed.

    Parameters
    ----------
    degree_channels
        Channel widths for degrees zero through ``lmax``.

    Returns
    -------
    list[int]
        Probe ranks for degrees one through ``lmax``.

    Raises
    ------
    ValueError
        If the degree profile does not cover a supported ``lmax``.
    """
    if len(degree_channels) not in {3, 4, 5}:
        raise ValueError(
            "`degree_channels` must contain three to five entries, got "
            f"{len(degree_channels)}"
        )
    return [int(degree_channels[2]), 2] + [1] * (len(degree_channels) - 3)
