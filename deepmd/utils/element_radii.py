# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-element covalent radii that set the length scale of each atom pair."""

import numpy as np

from deepmd.utils.model_preset_data import (
    PERIODIC_TABLE,
)

# fmt: off
COVALENT_RADII: tuple[float, ...] = (
    0.32, 0.46, 1.33, 1.02, 0.85, 0.75, 0.71, 0.63, 0.64, 0.67, 1.55, 1.39, 1.26,
    1.16, 1.11, 1.03, 0.99, 0.96, 1.96, 1.71, 1.48, 1.36, 1.34, 1.22, 1.19, 1.16,
    1.11, 1.10, 1.12, 1.18, 1.24, 1.21, 1.21, 1.16, 1.14, 1.17, 2.10, 1.85, 1.63,
    1.54, 1.47, 1.38, 1.28, 1.25, 1.25, 1.20, 1.28, 1.36, 1.42, 1.40, 1.40, 1.36,
    1.33, 1.31, 2.32, 1.96, 1.80, 1.63, 1.76, 1.74, 1.73, 1.72, 1.68, 1.69, 1.68,
    1.67, 1.66, 1.65, 1.64, 1.70, 1.62, 1.52, 1.46, 1.37, 1.31, 1.29, 1.22, 1.23,
    1.24, 1.33, 1.44, 1.44, 1.51, 1.45, 1.47, 1.42, 2.23, 2.01, 1.86, 1.75, 1.69,
    1.70, 1.71, 1.72, 1.66, 1.66, 1.68, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61, 1.57,
    1.49, 1.43, 1.41, 1.34, 1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65,
    1.57,
)
"""Molecular single-bond covalent radii in Å, indexed by atomic number minus one.

Pyykkö and Atsumi, *Chem. Eur. J.* **15**, 186 (2009), the one published set that
covers all 118 elements. The sum of two of these radii estimates the covalent
bond length of the corresponding pair, the length scale that pair-wise lengths
expressed as a fraction of the pair's own size are measured against.
"""
# fmt: on

_SYMBOL_TO_RADIUS: dict[str, float] = dict(
    zip(PERIODIC_TABLE, COVALENT_RADII, strict=True)
)


def covalent_radii_from_type_map(type_map: list[str]) -> np.ndarray:
    """
    Look up the covalent radius of every type in a type map.

    Parameters
    ----------
    type_map : list[str]
        Element symbols; the index in this list is the ``atype`` value.

    Returns
    -------
    np.ndarray
        Radii in Å with shape ``(ntypes,)`` in float64.

    Raises
    ------
    ValueError
        If a symbol is not an element of the periodic table.
    """
    unknown = [symbol for symbol in type_map if symbol not in _SYMBOL_TO_RADIUS]
    if unknown:
        raise ValueError(
            f"No covalent radius is tabulated for {unknown}; every type must be "
            "an element symbol."
        )
    return np.asarray(
        [_SYMBOL_TO_RADIUS[symbol] for symbol in type_map], dtype=np.float64
    )
