# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conditions of the isolated neutral ground-state atom of every type.

The isolated-atom energy reference (``vacuum_ref``) pins the energy of an
atom without neighbors to its preset bias. The reference atom is the neutral
atom in its ground state: zero charge, the ground-state spin multiplicity as
the charge/spin condition of the descriptor, and a spin vector for the
native-spin descriptors. The magnitude of that vector is the spin moment the
preset names with the energy of the type, and the number of unpaired
electrons otherwise. The ground-state quantities derive from the
spin-resolved orbital occupation table of the elements.
"""

from typing import (
    Any,
)

import numpy as np

from deepmd.utils.econf_embd import (
    electronic_configuration_embedding,
)
from deepmd.utils.preset_out_bias import (
    REFERENCE_SPIN_KEY,
    preset_assigns,
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


def reference_spin(
    type_map: list[str], preset_out_bias: dict[str, Any] | None = None
) -> np.ndarray:
    """Spin vector of the isolated atom that every type is referenced to.

    The energy of an isolated atom depends on its spin moment, so the
    reference atom carries the moment its preset energy belongs to.

    Parameters
    ----------
    type_map : list[str]
        Name of every type.
    preset_out_bias : dict[str, Any], optional
        Normalized preset bias of the atomic model. A type whose entry names
        a spin moment takes that magnitude, in the units of the spin input;
        every other type takes the unpaired-electron count of the neutral
        ground-state atom, in Bohr magnetons, and is named by its element
        symbol.

    Returns
    -------
    np.ndarray
        Spin vectors with shape (ntypes, 3), directed along ``z``; the energy
        of an isolated atom does not depend on the direction.

    Raises
    ------
    ValueError
        If a type without a named moment is not named by an element symbol.
    """
    moments: list[float | None] = (preset_out_bias or {}).get(
        REFERENCE_SPIN_KEY, [None] * len(type_map)
    )
    unnamed = [
        name for name, moment in zip(type_map, moments, strict=True) if moment is None
    ]
    ground = dict(zip(unnamed, unpaired_electrons(unnamed).tolist(), strict=True))
    spin = np.zeros((len(type_map), 3), dtype=np.float64)
    spin[:, 2] = [
        ground[name] if moment is None else moment
        for name, moment in zip(type_map, moments, strict=True)
    ]
    return spin


def resolve_vacuum_ref(
    fitting: Any, preset_out_bias: dict[str, list[list | None]] | None
) -> None:
    """Keep the vacuum reference of a fitting only on an output whose bias the preset fixes.

    The reference makes an atom without neighbors contribute exactly its bias,
    which is the isolated-atom value only when ``preset_out_bias`` fixes it. An
    output whose bias is fitted from the data keeps the plain network output:
    in multi-task training a branch without a table then differs from the
    referenced branches that share its network by a constant per type, which
    the case conditioning represents, instead of by an offset that would have
    to vanish on the isolated atoms.

    Parameters
    ----------
    fitting
        A fitting network; its ``vacuum_ref`` option is switched off in place
        when the preset does not fix the bias of its output ``var_name``.
    preset_out_bias
        Normalized preset bias of the atomic model, or None.
    """
    if fitting.vacuum_ref and not preset_assigns(preset_out_bias, fitting.var_name):
        fitting.vacuum_ref = False
