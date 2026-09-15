# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conditions of the isolated neutral ground-state atom of every type.

The isolated-atom energy reference (``vacuum_ref``) pins the energy of an
atom without neighbors to its preset bias. The reference atom is the neutral
atom in its ground state: zero charge, the ground-state spin multiplicity as
the charge/spin condition of the descriptor, and a spin vector whose magnitude
is the number of unpaired electrons for the native-spin descriptors. Every
quantity derives from the spin-resolved orbital occupation table of the
elements.
"""

import numpy as np

from deepmd.utils.econf_embd import (
    electronic_configuration_embedding,
)


def unpaired_electrons(type_map: list[str]) -> np.ndarray:
    """Number of unpaired electrons of the neutral ground-state atom of every type.

    The occupation table encodes every orbital as a pair of entries: ``[1, 1]``
    doubly occupied, ``[-1, 1]`` singly occupied and ``[-1, -1]`` empty. The
    singly occupied orbitals are counted.

    Parameters
    ----------
    type_map : list[str]
        Element symbol of every type.

    Returns
    -------
    np.ndarray
        The unpaired-electron count of every type with shape (ntypes,).

    Raises
    ------
    ValueError
        If a type name is not an element symbol.
    """
    unknown = [
        name for name in type_map if name not in electronic_configuration_embedding
    ]
    if unknown:
        raise ValueError(
            "the isolated-atom reference requires element symbols as type names; "
            f"unknown names: {unknown}"
        )
    counts = []
    for name in type_map:
        occupation = electronic_configuration_embedding[name].reshape(-1, 2)
        counts.append(np.sum((occupation[:, 0] == -1) & (occupation[:, 1] == 1)))
    return np.array(counts, dtype=np.int64)


def reference_charge_spin(type_map: list[str]) -> np.ndarray:
    """Charge and spin condition of the isolated neutral ground-state atom of every type.

    Parameters
    ----------
    type_map : list[str]
        Element symbol of every type.

    Returns
    -------
    np.ndarray
        ``[charge, multiplicity]`` of every type with shape (ntypes, 2): the
        charge is zero and the multiplicity is the unpaired-electron count
        plus one.
    """
    unpaired = unpaired_electrons(type_map)
    return np.stack([np.zeros_like(unpaired), unpaired + 1], axis=-1).astype(np.float64)


def reference_spin(type_map: list[str]) -> np.ndarray:
    """Spin vector of the isolated neutral ground-state atom of every type.

    Parameters
    ----------
    type_map : list[str]
        Element symbol of every type.

    Returns
    -------
    np.ndarray
        Spin vectors with shape (ntypes, 3) in Bohr magnetons, of magnitude
        equal to the unpaired-electron count and directed along ``z``; the
        energy of an isolated atom does not depend on the direction.
    """
    spin = np.zeros((len(type_map), 3), dtype=np.float64)
    spin[:, 2] = unpaired_electrons(type_map)
    return spin
