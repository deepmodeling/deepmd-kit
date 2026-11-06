# SPDX-License-Identifier: LGPL-3.0-or-later
"""Structural widths shared by compressed DPA4C operators."""

from dataclasses import (
    dataclass,
)


@dataclass(frozen=True)
class DescriptorProfile:
    r"""Describe every width the compiled operator derives from its inputs.

    Attributes
    ----------
    channels
        Scalar degree-zero width :math:`C_0`.
    lmax
        Maximum angular degree.
    degree_channels
        Channel width of each degree from zero through ``lmax``.
    ranks
        Bispectrum probe rank of each degree from one through ``lmax``.
    moment_width
        Flat moment width :math:`S=\\sum_\\ell(2\\ell+1)C_\\ell`.
    output_width
        Invariant descriptor width consumed by the fitting network.
    gram_base
        Descriptor coordinate of the first degree-one Gram entry.
    bispectrum_base
        Descriptor coordinate of the first bispectrum entry.
    spin_width
        Width of the trailing native-spin invariant block.
    basis
        Width of the low-rank expansion of the scalar-only channels, zero for
        a profile that admits no expansion.
    """

    channels: int
    lmax: int
    degree_channels: tuple[int, ...]
    ranks: tuple[int, ...]
    moment_width: int
    output_width: int
    gram_base: int
    bispectrum_base: int
    spin_channels: int
    spin_width: int
    basis: int

    @property
    def state_width(self) -> int:
        """Return the saved-state width: the moments plus both normalizers."""
        return self.moment_width + 2

    @property
    def has_spin(self) -> bool:
        """Return whether the compiled profile carries the spin families."""
        return self.spin_channels > 0

    @property
    def spin_slice(self) -> slice:
        """Return the descriptor columns holding the spin invariants.

        The layout closes with the spin invariants, the two moment divisors
        and the center type tail, so the block is addressed from the end and
        is empty for a spin-free profile.
        """
        stop = self.output_width - 2 - self.degree_channels[0]
        return slice(stop - self.spin_width, stop)
