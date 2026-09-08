# SPDX-License-Identifier: LGPL-3.0-or-later
"""Coefficient layout of the reduced SO(2) basis as the cuTile kernels see it.

The canonical reduced layout is defined once, in
:mod:`deepmd.dpmodel.descriptor.dpa4_nn.indexing`, and is reused here rather
than restated. This module adds the two things the tile programming model needs
on top of it: the structural non-zeros of the reduced rotation, which fix the
coefficient numbering every generated kernel shares, and the power-of-two padded
extents, which every tile must have.

Three layouts appear throughout the package and the kernels must agree on them
exactly:

full basis
    ``(lmax + 1)^2`` spherical-harmonic coefficients ordered by ``(l, m)``, the
    layout of the node features and of the per-edge Wigner-D blocks.
reduced m-major
    ``3 * lmax + 1`` rows at ``mmax == 1``: the ``m = 0`` degrees first, then
    ``m = -1``, then ``m = +1``. The mixing stack operates here, and the layout
    is block diagonal over ``|m|``.
padded groups
    each ``|m|`` group widened to a power-of-two degree count. Padded degrees
    carry exact zeros in both the activation and the weight, so a padded
    contraction equals the exact one.
"""

from __future__ import (
    annotations,
)

import dataclasses

from deepmd.dpmodel.descriptor.dpa4_nn.indexing import (
    build_m_major_index,
    get_so3_dim_of_lmax,
)

from ..common import (
    next_pow2,
)

__all__ = ["SO2TileLayout", "m_major_index", "rotation_pairs"]


def m_major_index(lmax: int, mmax: int = 1) -> list[int]:
    """Return the reduced rows as indices into the full ``(lmax + 1)^2`` basis.

    Parameters
    ----------
    lmax : int
        Maximum spherical-harmonic degree.
    mmax : int
        Maximum retained absolute order.

    Returns
    -------
    list[int]
        Full-basis index of each reduced row, in reduced-row order. Python ints,
        because the generated kernels embed them as literals.
    """
    return [int(index) for index in build_m_major_index(lmax, mmax)]


def rotation_pairs(lmax: int, mmax: int = 1) -> list[tuple[int, int]]:
    """Enumerate the structural non-zeros of the reduced rotation.

    The Wigner-D matrix is block diagonal in the degree, so projecting onto the
    reduced rows couples reduced row ``r`` only to the full-basis rows of its own
    degree. The returned order defines the coefficient numbering used by every
    generated kernel that touches the rotation.

    Parameters
    ----------
    lmax : int
        Maximum spherical-harmonic degree.
    mmax : int
        Maximum retained absolute order.

    Returns
    -------
    list[tuple[int, int]]
        ``(reduced_row, full_row)`` pairs, ``sum_r (2 * l_r + 1)`` of them.
    """
    pairs = []
    for reduced, full in enumerate(m_major_index(lmax, mmax)):
        degree = int(full**0.5)
        pairs.extend(
            (reduced, full_row)
            for full_row in range(degree * degree, (degree + 1) ** 2)
        )
    return pairs


@dataclasses.dataclass(frozen=True)
class SO2TileLayout:
    """Every extent the generated kernels of one block layout need.

    Attributes
    ----------
    lmax, focus_dim, n_layers
        Configuration of the convolution block.
    """

    lmax: int
    focus_dim: int
    n_layers: int

    @property
    def key(self) -> tuple[int, int]:
        """Shape key for the launch-configuration tables."""
        return (self.lmax, self.focus_dim)

    @property
    def n_m0(self) -> int:
        """Real degree count of the ``m = 0`` group."""
        return self.lmax + 1

    @property
    def n_m1(self) -> int:
        """Real degree count of the ``|m| = 1`` group."""
        return 2 * self.lmax

    @property
    def pad_m0(self) -> int:
        """``m = 0`` degree count rounded up to a power of two."""
        return next_pow2(self.n_m0)

    @property
    def pad_m1(self) -> int:
        """``|m| = 1`` degree count rounded up to a power of two."""
        return next_pow2(self.n_m1)

    @property
    def width_m0(self) -> int:
        """Padded ``m = 0`` group width in channels, a tile extent of the stack."""
        return self.pad_m0 * self.focus_dim

    @property
    def width_m1(self) -> int:
        """Padded ``|m| = 1`` group width in channels, a tile extent of the stack."""
        return self.pad_m1 * self.focus_dim

    @property
    def dim(self) -> int:
        """Full-basis coefficient count."""
        return get_so3_dim_of_lmax(self.lmax)

    @property
    def row(self) -> int:
        """Reduced-layout width in channels, the stride of the activation."""
        return (3 * self.lmax + 1) * self.focus_dim

    @property
    def kernel_size(self) -> int:
        """Compact per-edge degree-mixing kernel size at rank one."""
        return self.n_m0 * self.n_m0 + self.lmax * self.lmax

    @property
    def n_gated(self) -> int:
        """Number of gated layers; the final layer is the identity layer."""
        return self.n_layers - 1
