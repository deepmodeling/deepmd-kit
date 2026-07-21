# SPDX-License-Identifier: LGPL-3.0-or-later
"""Neural and geometric building blocks for DPA4C."""

from .bispectrum import (
    BispectrumLayout,
    build_bispectrum_layout,
    derive_bispectrum_ranks,
    enumerate_degree_triples,
)
from .geometry import (
    MAX_ANGULAR_DEGREE,
    build_angular_basis,
    build_moment_indices,
    degree_offsets,
    derive_degree_channels,
    packed_l2_to_stf,
)
from .pair_film import (
    OrderedPairFiLM,
)
from .readout import (
    InvariantReadout,
)

__all__ = [
    "MAX_ANGULAR_DEGREE",
    "BispectrumLayout",
    "InvariantReadout",
    "OrderedPairFiLM",
    "build_angular_basis",
    "build_bispectrum_layout",
    "build_moment_indices",
    "degree_offsets",
    "derive_bispectrum_ranks",
    "derive_degree_channels",
    "enumerate_degree_triples",
    "packed_l2_to_stf",
]
