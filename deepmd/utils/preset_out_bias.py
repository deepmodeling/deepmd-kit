# SPDX-License-Identifier: LGPL-3.0-or-later
"""Preset output bias of atomic models.

The ``preset_out_bias`` model option assigns the output bias of selected atom
types, typically their energy in vacuum. An assigned output is fixed by the
preset alone: every type that occurs in the training data must be assigned,
the assigned types take the preset value, and no statistics are computed for
the output. A string entry names a bundled table of isolated-atom energies or
the JSON file holding a table. The helpers of this module turn the configured
form into the canonical per-type form stored on the atomic model, carry it
through type-map changes, and expand it into the per-type rows written to the
model.

The canonical form holds nested Python lists rather than arrays: it is written
to the serialized model as it is, and no array data lives on the atomic model
outside its registered variables, which the module wrappers of some backends
require.
"""

import json
from collections.abc import (
    Iterable,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

BUNDLED_TABLES_FILE = Path(__file__).with_name("preset_out_bias_tables.json")


def bundled_preset_out_bias_tables() -> dict[str, dict[str, Any]]:
    """Preset tables shipped with the package, keyed by table name.

    The database file holds one entry per table name with the ``source`` of
    the table and its ``values`` keyed by element symbol; the bundled entries
    are tables of isolated-atom energies in eV.

    Returns
    -------
    dict[str, dict[str, Any]]
        For every table name, its values keyed by element symbol.
    """
    with open(BUNDLED_TABLES_FILE) as stream:
        database = json.load(stream)
    return {name: entry["values"] for name, entry in database.items()}


def load_preset_out_bias_table(spec: str) -> dict[str, Any]:
    """Look up a preset table by its bundled name or read it from a JSON file.

    Parameters
    ----------
    spec
        The name of a bundled table, or the path of a JSON file keyed by
        element name, relative to the working directory or absolute. A
        bundled name takes precedence.

    Returns
    -------
    dict[str, Any]
        The table, mapping element names to preset values.

    Raises
    ------
    ValueError
        If ``spec`` is neither a bundled table nor an existing file, or if the
        file does not hold a JSON object.
    """
    tables = bundled_preset_out_bias_tables()
    if spec in tables:
        return tables[spec]
    if not Path(spec).is_file():
        raise ValueError(
            f"the preset_out_bias table {spec!r} is neither one of the bundled "
            f"tables {sorted(tables)} nor an existing JSON file"
        )
    with open(spec) as stream:
        table = json.load(stream)
    if not isinstance(table, dict):
        raise ValueError(
            f"the preset_out_bias table {spec!r} must be a JSON object keyed by "
            f"element name, got {type(table).__name__}"
        )
    return table


def resolve_preset_out_bias_tables(model_config: dict[str, Any]) -> dict[str, Any]:
    """Replace the string entries of ``preset_out_bias`` by their tables.

    Parameters
    ----------
    model_config
        A model configuration, or a multi-task configuration whose branches
        live under ``model_dict``; a preset next to ``model_dict`` is resolved
        as well.

    Returns
    -------
    dict[str, Any]
        The configuration with every string entry of ``preset_out_bias``
        replaced by its bundled table or the content of its JSON file, so that
        the configuration is self-contained.
    """
    preset = model_config.get("preset_out_bias")
    if preset and any(isinstance(spec, str) for spec in preset.values()):
        resolved = {
            key: load_preset_out_bias_table(spec) if isinstance(spec, str) else spec
            for key, spec in preset.items()
        }
        model_config = {**model_config, "preset_out_bias": resolved}
    if "model_dict" in model_config:
        branches = {
            name: resolve_preset_out_bias_tables(branch)
            for name, branch in model_config["model_dict"].items()
        }
        model_config = {**model_config, "model_dict": branches}
    return model_config


def normalize_preset_out_bias(
    preset_out_bias: dict[str, Any] | None,
    type_map: list[str],
) -> dict[str, list[list | None]] | None:
    """Normalize the ``preset_out_bias`` model option into per-type entries.

    Parameters
    ----------
    preset_out_bias
        The preset bias of the atomic outputs, keyed by output name. Every value
        is a sequence with one entry per type of ``type_map``, where None
        leaves the type unassigned; a dict keyed by element name, whose
        elements outside ``type_map`` are ignored; or a string naming a bundled
        table or the path of a JSON file holding such a dict. An entry is a
        number, or a nested list or array of the output shape. The normalized
        form is accepted as well, so the function is idempotent.
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
        If a value is neither a sequence with one entry per type, a dict, a
        bundled table name nor a file path, or if an entry is not entirely
        finite and numeric.
    """
    if preset_out_bias is None:
        return None
    normalized = {}
    for key, spec in preset_out_bias.items():
        if isinstance(spec, str):
            spec = load_preset_out_bias_table(spec)
        if isinstance(spec, dict):
            entries = [spec.get(name) for name in type_map]
        elif isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == len(type_map):
            entries = list(spec)
        else:
            raise ValueError(
                f"preset_out_bias['{key}'] must be a list with one entry per type "
                f"of the type_map ({len(type_map)}), a dict keyed by element name, "
                f"a bundled table name or the path of a JSON file, got {spec!r}"
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
                    "or left unassigned"
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
        The preset bias on the new type map; a new type is unassigned.
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


def preset_out_bias_rows(
    preset_out_bias: dict[str, list[list | None]],
    type_map: list[str],
    observed_types: list[str],
    stored_bias: np.ndarray,
    keys: list[str],
    sizes: list[int],
    keep_unassigned: bool,
    excluded_types: Iterable[int] = (),
) -> dict[str, np.ndarray]:
    """Expand the preset into the per-type bias rows of every assigned output.

    Parameters
    ----------
    preset_out_bias
        Normalized preset bias.
    type_map
        Element names of the model, indexed by type.
    observed_types
        Element names that occur in the data whose bias is being set; names
        outside ``type_map`` are ignored.
    stored_bias
        Stored output bias with shape (n_out, ntypes, max_size); the output
        ``keys[i]`` occupies ``stored_bias[i, :, :sizes[i]]``.
    keys
        Output names in the order of the first axis of ``stored_bias``.
    sizes
        Flattened size of every output.
    keep_unassigned
        Whether a type without preset keeps its stored bias; otherwise its
        bias is zero.
    excluded_types
        Types whose atomic contribution is excluded from the outputs, such as
        the virtual types of a spin model; they need no preset.

    Returns
    -------
    dict[str, np.ndarray]
        For every output with at least one assigned type, the rows with shape
        (ntypes, size).

    Raises
    ------
    ValueError
        If a type that occurs in the data has no preset for an assigned
        output.
    """
    ntypes = len(type_map)
    index = {name: ii for ii, name in enumerate(type_map)}
    excluded = set(excluded_types)
    required = [
        index[name]
        for name in observed_types
        if name in index and index[name] not in excluded
    ]
    rows = {}
    for idx, (key, size) in enumerate(zip(keys, sizes, strict=True)):
        preset = make_preset_out_bias(ntypes, preset_out_bias.get(key, [None] * ntypes))
        if preset is None:
            continue
        preset = preset.reshape(ntypes, size)
        unassigned = np.isnan(preset).any(axis=1)
        missing = [type_map[ii] for ii in required if unassigned[ii]]
        if missing:
            raise ValueError(
                f"preset_out_bias['{key}'] does not assign the elements {missing} "
                "that occur in the data; an assigned output needs a preset for "
                "every element in the data"
            )
        fill = stored_bias[idx, :, :size] if keep_unassigned else np.zeros_like(preset)
        rows[key] = np.where(unassigned[:, None], fill, preset)
    return rows


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
