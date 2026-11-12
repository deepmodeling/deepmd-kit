# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conditions of the isolated atom that every type is referenced to.

The isolated-atom energy reference (``vacuum_ref``) subtracts from the network
output of every atom the output of an isolated atom of its type, the reference
atom, so that an atom without neighbors under the conditions of its reference
atom contributes its preset bias.

The charge and spin multiplicity condition a whole frame, and the preset
energy of a type was computed for one isolated atom of that type. A table
whose energies belong to one electronic state of every element names that
state per element (``charge`` and ``multiplicity`` of the entry), as the
bundled table of neutral ground-state atoms does; the energy of another state
is then a learned difference from the preset. A table computed under one
common condition names none, and every type is referenced to the default
condition of the model, which is the condition every frame of such data takes,
so every atom without neighbors has exactly its preset bias.

The native spin of an atom is its own moment, on which the energy of an
isolated atom depends. The reference atom carries the moment its preset energy
belongs to: the one the preset names with the energy of the type, and the
number of unpaired electrons of the neutral ground-state atom otherwise. The
ground-state quantities derive from the spin-resolved orbital occupation table
of the elements.
"""

from typing import (
    Any,
)

import numpy as np

from deepmd.utils.econf_embd import (
    electronic_configuration_embedding,
)
from deepmd.utils.preset_out_bias import (
    REFERENCE_CHARGE_SPIN_KEY,
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


def reference_charge_spin(
    type_map: list[str],
    preset_out_bias: dict[str, Any] | None = None,
    default: Any | None = None,
    output: str = "energy",
) -> np.ndarray:
    """Charge/spin condition of the isolated atom that every type is referenced to.

    The condition belongs to the isolated atom the preset energy of a type
    was computed for. A table that computes its energies from one electronic
    state of every element names that state per element; a table computed
    under one common condition leaves it to the model, which then takes
    ``default``. A type without a preset energy is referenced to nothing, so
    it needs no condition; its row is a placeholder that no forward reads.

    Parameters
    ----------
    type_map : list[str]
        Name of every type.
    preset_out_bias : dict[str, Any], optional
        Normalized preset bias of the atomic model. A type whose entry names
        a condition keeps it.
    default : array-like, optional
        The condition ``[charge, multiplicity]`` of a type whose entry names
        none.
    output : str, default="energy"
        Name of the assigned output whose entries give the types their
        preset.

    Returns
    -------
    np.ndarray
        ``[charge, multiplicity]`` of every type with shape (ntypes, 2).

    Raises
    ------
    ValueError
        If a type has a preset energy but names no condition while there is
        no default condition.
    """
    conditions: list[list[float] | None] = (preset_out_bias or {}).get(
        REFERENCE_CHARGE_SPIN_KEY, [None] * len(type_map)
    )
    assigned = (preset_out_bias or {}).get(output, [None] * len(type_map))
    if len(conditions) != len(type_map) or len(assigned) != len(type_map):
        raise ValueError(
            "the preset must hold one entry per type of the type_map "
            f"({len(type_map)}), got {len(assigned)} values and "
            f"{len(conditions)} conditions"
        )
    unnamed = [
        name
        for name, condition, entry in zip(type_map, conditions, assigned, strict=True)
        if entry is not None and condition is None
    ]
    if unnamed and default is None:
        raise ValueError(
            f"the preset assigns no charge/spin condition to the types {unnamed}, "
            "and the descriptor has no `default_chg_spin` to reference them to; "
            "name the `charge` and `multiplicity` of their isolated atoms in "
            "`preset_out_bias` or set `default_chg_spin`"
        )
    fallback = [0.0, 1.0] if default is None else [float(v) for v in default]
    return np.array(
        [fallback if condition is None else condition for condition in conditions],
        dtype=np.float64,
    ).reshape(len(type_map), 2)


def store_reference_charge_spin(
    type_map: list[str],
    preset_out_bias: dict[str, Any] | None,
    conditions: np.ndarray,
    output: str = "energy",
) -> None:
    """Record the resolved charge/spin condition of every assigned type.

    A type without a preset energy has no reference atom, so it names no
    condition. Storing only the types the preset assigns keeps a row that no
    forward reads from reading as a declared condition later, when the model
    is fine-tuned on a preset that assigns the type without naming one: that
    type then takes the default condition of the new model instead of the
    placeholder of the old one.

    Parameters
    ----------
    type_map : list[str]
        Name of every type.
    preset_out_bias : dict[str, Any], optional
        Normalized preset bias of the atomic model; the condition list is
        written into it. None leaves it unset.
    conditions : np.ndarray
        ``[charge, multiplicity]`` of every type with shape (ntypes, 2), as
        returned by :func:`reference_charge_spin`.
    output : str, default="energy"
        Name of the assigned output whose entries decide which types carry a
        reference.

    Raises
    ------
    ValueError
        If the lengths of ``type_map``, ``conditions`` and the preset differ.
    """
    if preset_out_bias is None:
        return
    assigned = preset_out_bias.get(output, [None] * len(type_map))
    if len(assigned) != len(type_map) or len(conditions) != len(type_map):
        raise ValueError(
            "the preset must hold one entry per type of the type_map "
            f"({len(type_map)}), got {len(assigned)} values and "
            f"{len(conditions)} conditions"
        )
    preset_out_bias[REFERENCE_CHARGE_SPIN_KEY] = [
        None if entry is None else [float(charge), float(multiplicity)]
        for entry, (charge, multiplicity) in zip(assigned, conditions, strict=True)
    ]


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
