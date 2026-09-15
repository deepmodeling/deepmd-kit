# SPDX-License-Identifier: LGPL-3.0-or-later
"""Preset output bias of atomic models.

The ``preset_out_bias`` model option assigns the output bias of selected atom
types, typically their energy in vacuum, while the remaining types are fitted
from the data. The helpers of this module turn the configured form into the
canonical per-type form stored on the atomic model, carry it through type-map
changes, and express it in the frame of the output statistics that consume it.

The canonical form holds nested Python lists rather than arrays: it is written
to the serialized model as it is, and no array data lives on the atomic model
outside its registered variables, which the module wrappers of some backends
require.
"""

from typing import (
    Any,
)

import numpy as np


def normalize_preset_out_bias(
    preset_out_bias: dict[str, Any] | None,
    type_map: list[str],
) -> dict[str, list[list | None]] | None:
    """Normalize the ``preset_out_bias`` model option into per-type entries.

    Parameters
    ----------
    preset_out_bias
        The preset bias of the atomic outputs, keyed by output name. Every value
        is either a sequence with one entry per type of ``type_map``, where None
        leaves the type to the statistics, or a dict keyed by element name that
        assigns the listed elements only. An entry is a number, or a nested
        list or array of the output shape. The normalized form is accepted as
        well, so the function is idempotent.
    type_map
        Element names of the model, indexed by type.

    Returns
    -------
    dict[str, list[list | None]] or None
        For every output, one entry per type of ``type_map``: None for a type
        without preset, otherwise the preset as a nested list of floats with
        at least one dimension. None if no preset is given.

    Raises
    ------
    ValueError
        If a value is neither a sequence with one entry per type nor a dict,
        if a dict names an element outside ``type_map``, or if an entry is not
        entirely finite and numeric.
    """
    if preset_out_bias is None:
        return None
    normalized = {}
    for key, spec in preset_out_bias.items():
        if isinstance(spec, dict):
            unknown = [name for name in spec if name not in type_map]
            if unknown:
                raise ValueError(
                    f"preset_out_bias['{key}'] assigns elements {unknown} "
                    f"that are not in the type_map {type_map}"
                )
            entries = [spec.get(name) for name in type_map]
        elif isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == len(type_map):
            entries = list(spec)
        else:
            raise ValueError(
                f"preset_out_bias['{key}'] must be a list with one entry per type "
                f"of the type_map ({len(type_map)}) or a dict keyed by element name, "
                f"got {spec!r}"
            )
        values = []
        for entry in entries:
            if entry is None:
                values.append(None)
                continue
            try:
                value = np.atleast_1d(np.array(entry, dtype=np.float64))
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"unsupported value {entry!r} in preset_out_bias['{key}']: "
                    "expected a number or a nested list of numbers"
                ) from err
            if not np.isfinite(value).all():
                raise ValueError(
                    f"preset_out_bias['{key}'] entry {entry!r} must be assigned "
                    "completely with finite values; a type is either preset "
                    "or left to the statistics"
                )
            values.append(value.tolist())
        normalized[key] = values
    return normalized


def remap_preset_out_bias(
    preset_out_bias: dict[str, list[list | None]] | None,
    remap_index: list[int],
) -> dict[str, list[list | None]] | None:
    """Reorder the per-type entries of a normalized preset bias onto a new type map.

    Parameters
    ----------
    preset_out_bias
        Normalized preset bias on the old type map.
    remap_index
        For every new type, the index of the type in the old type map, or a
        negative index for a type absent from the old type map, as returned by
        :func:`deepmd.utils.finetune.get_index_between_two_maps`.

    Returns
    -------
    dict[str, list[list | None]] or None
        The preset bias on the new type map; a new type is left to the
        statistics.
    """
    if preset_out_bias is None:
        return None
    padding: list[list | None] = [None] * len(remap_index)
    return {
        key: [(entries + padding)[ii] for ii in remap_index]
        for key, entries in preset_out_bias.items()
    }


def check_preset_out_bias(
    preset_out_bias: dict[str, list[list | None]] | None,
    keys: list[str],
    distinguish_types: bool | None = None,
) -> None:
    """Validate a normalized preset bias against the outputs of a model.

    Parameters
    ----------
    preset_out_bias
        Normalized preset bias, or None.
    keys
        Names of the model outputs that carry a bias.
    distinguish_types
        Whether the output statistics of the model resolve atom types, or None
        while this is not yet known. A preset assigns the bias of individual
        types and therefore requires type-resolved statistics.

    Raises
    ------
    ValueError
        If the preset names an output the model does not produce, or assigns
        a type while the statistics do not distinguish types.
    """
    if preset_out_bias is None:
        return
    unknown = sorted(set(preset_out_bias) - set(keys))
    if unknown:
        raise ValueError(
            f"preset_out_bias names the outputs {unknown} which the model does "
            f"not produce; its outputs are {keys}"
        )
    assigned = any(
        entry is not None for entries in preset_out_bias.values() for entry in entries
    )
    if assigned and distinguish_types is False:
        raise ValueError(
            "preset_out_bias assigns the bias of individual atom types, but the "
            "output statistics of this model do not distinguish atom types"
        )


def make_preset_out_bias(
    ntypes: int,
    ibias: list[list | np.ndarray | None] | np.ndarray,
) -> np.ndarray | None:
    """Assemble the preset bias of one output into a per-type array.

    Parameters
    ----------
    ntypes
        The number of atom types.
    ibias
        One entry per type: None for a type without preset, otherwise a nested
        list or array of the output shape. An already assembled array of shape
        (ntypes, ...) whose unassigned rows hold NaN is returned unchanged.

    Returns
    -------
    np.ndarray or None
        Array of shape (ntypes, *(odim0, odim1, ...)) with NaN for the types
        without preset, or None if no type is assigned.

    Raises
    ------
    ValueError
        If ``ibias`` does not have one entry per type.
    """
    if len(ibias) != ntypes:
        raise ValueError("the length of preset bias list should be ntypes")
    if all(ii is None for ii in ibias):
        return None
    for refb in ibias:
        if refb is not None:
            break
    refb = np.array(refb)
    nbias = [
        np.full_like(refb, np.nan, dtype=np.float64) if ii is None else ii
        for ii in ibias
    ]
    return np.array(nbias)


def apply_preset_out_bias(
    preset_out_bias: dict[str, list[list | None]],
    out_bias: np.ndarray,
    keys: list[str],
    sizes: list[int],
) -> np.ndarray:
    """Set assigned bias rows before evaluating a residual-statistics model.

    Parameters
    ----------
    preset_out_bias
        Normalized preset bias.
    out_bias
        Stored bias with shape (n_out, ntypes, max_size).
    keys
        Output names in the order of the first bias axis.
    sizes
        Flattened size of every output.

    Returns
    -------
    np.ndarray
        A copy with assigned rows replaced by their finite presets. Unassigned
        rows are unchanged. Pinning before prediction also handles non-finite
        stored values, for which an additive shift cannot enforce a preset.
    """
    result = np.array(out_bias, copy=True)
    for idx, (key, size) in enumerate(zip(keys, sizes, strict=True)):
        if key in preset_out_bias:
            result[idx, :, :size] = override_assigned_bias(
                result[idx, :, :size],
                make_preset_out_bias(result.shape[1], preset_out_bias[key]),
            )
    return result


def preset_out_bias_shift(
    preset_out_bias: dict[str, list[list | None]] | None,
    out_bias: np.ndarray,
    keys: list[str],
    sizes: list[int],
) -> dict[str, np.ndarray] | None:
    """Express a preset bias as shifts of the stored output bias.

    The statistics fitted in ``change-by-statistic`` mode are added to the
    stored bias, so an assigned type enters the fit as ``preset - stored`` and
    its stored bias ends at exactly the preset value.

    Parameters
    ----------
    preset_out_bias
        Normalized preset bias, or None.
    out_bias
        Stored output bias with shape (n_out, ntypes, max_size); the output
        ``keys[i]`` occupies ``out_bias[i, :, :sizes[i]]``.
    keys
        Output names in the order of the first axis of ``out_bias``.
    sizes
        Flattened size of every output.

    Returns
    -------
    dict[str, np.ndarray] or None
        For every output with at least one assigned type, the shifts with
        shape (ntypes, size) where unassigned types hold NaN. None if no
        preset is configured.
    """
    if preset_out_bias is None:
        return None
    ntypes = out_bias.shape[1]
    shift = {}
    for idx, (key, size) in enumerate(zip(keys, sizes, strict=True)):
        if key not in preset_out_bias:
            continue
        preset = make_preset_out_bias(ntypes, preset_out_bias[key])
        if preset is not None:
            shift[key] = preset.reshape(ntypes, size) - out_bias[idx, :, :size]
    return shift


def override_assigned_bias(
    bias: np.ndarray,
    assigned_bias: np.ndarray | None,
) -> np.ndarray:
    """Replace the rows of the assigned types by their assigned values.

    Parameters
    ----------
    bias
        Computed bias with shape (ntypes, ...).
    assigned_bias
        Assigned bias with the same number of elements, where the rows of the
        unassigned types hold NaN, or None.

    Returns
    -------
    np.ndarray
        The bias with the assigned rows overridden, in the shape of ``bias``.
    """
    if assigned_bias is None:
        return bias
    ntypes = bias.shape[0]
    assigned = np.asarray(assigned_bias).reshape(ntypes, -1)
    mask = ~np.isnan(assigned).any(axis=1)
    result = bias.reshape(ntypes, -1).copy()
    result[mask] = assigned[mask]
    return result.reshape(bias.shape)
