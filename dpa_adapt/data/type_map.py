# SPDX-License-Identifier: LGPL-3.0-or-later
# data/type_map.py
#
# Automatic type_map resolution: read from checkpoint, union from data,
# validate subsets.  Users should never need to touch ``_extra_state``.

from __future__ import (
    annotations,
)


def read_checkpoint_type_map(
    pretrained: str,
    branch: str | None = None,
) -> list[str]:
    """Read the global type_map from a DPA checkpoint.

    For multi-task checkpoints the type_map lives in
    ``shared_dict.<descriptor>.type_map`` or falls back to the branch's
    own ``type_map``.  For single-task checkpoints it is at the model root.

    Parameters
    ----------
    pretrained : str
        Path to the ``.pt`` checkpoint.
    branch : str, optional
        Branch name for multi-task checkpoints.  If not given the first
        available branch is used.

    Returns
    -------
    list[str]
        Element symbols.
    """
    from dpa_adapt._backend import (
        load_torch_file,
        resolve_pretrained_path,
    )

    pretrained = resolve_pretrained_path(pretrained)
    sd = load_torch_file(pretrained)
    if "model" in sd:
        sd = sd["model"]

    params = sd["_extra_state"]["model_params"]

    # Multi-task: type_map is in shared_dict or per-branch
    model_dict = params.get("model_dict", {})
    if model_dict:
        shared = params.get("shared_dict", {})
        # shared_dict values are descriptor/fitting_net dicts; some may
        # contain a type_map list directly, some use a "type_map" key that
        # points to a name in shared_dict.
        for v in shared.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                return v
        # Fall back to the branch's own type_map
        if branch and branch in model_dict:
            tm = model_dict[branch].get("type_map")
        else:
            first = next(iter(model_dict.values()))
            tm = first.get("type_map")
        if isinstance(tm, str):
            tm = shared.get(tm)
        if isinstance(tm, list):
            return tm

    # Single-task: type_map at model root
    tm = params.get("type_map")
    if isinstance(tm, list):
        return tm

    raise ValueError(
        f"Could not locate type_map in checkpoint {pretrained}. "
        "Pass type_map=[...] explicitly."
    )


def read_data_type_map_union(systems: list) -> list[str]:
    """Read ``atom_names`` from every system and return the union.

    Each system may declare a subset of elements (different dopants per
    formula).  The union covers all elements present across the dataset.

    Parameters
    ----------
    systems : list[dpdata.System]
        Systems to scan for element names.

    Returns
    -------
    list[str]
        Sorted union of all element symbols appearing in any system.
    """
    elems: set[str] = set()
    for sys in systems:
        names = sys.data.get("atom_names", [])
        for name in names:
            if name:
                elems.add(str(name))
    if not elems:
        raise ValueError(
            "No atom_names found in any system. "
            "Ensure data has been loaded with dpdata correctly."
        )
    return sorted(elems)


def validate_type_map_subset(
    data_elements: list[str],
    checkpoint_elements: list[str],
    *,
    label: str = "data",
) -> None:
    """Raise ``ValueError`` if *data_elements* is not a subset of *checkpoint_elements*.

    Parameters
    ----------
    data_elements : list[str]
        Element symbols appearing in the data (typically from
        ``read_data_type_map_union``).
    checkpoint_elements : list[str]
        Element symbols covered by the checkpoint (from
        ``read_checkpoint_type_map``).
    label : str
        Human-readable label for the error message (e.g. ``"OER data"``).

    Raises
    ------
    ValueError
        If any data element is not in the checkpoint type_map.
    """
    ckpt_set = set(checkpoint_elements)
    unsupported = [e for e in data_elements if e not in ckpt_set]
    if unsupported:
        ckpt_repr = (
            f"{checkpoint_elements[:3]}...{checkpoint_elements[-1:]} "
            f"({len(checkpoint_elements)} elements)"
            if len(checkpoint_elements) > 8
            else str(checkpoint_elements)
        )
        raise ValueError(
            f"Element(s) in {label} are not covered by the checkpoint.\n"
            f"  {label} type_map: {data_elements}\n"
            f"  Unsupported elements: {unsupported}\n"
            f"  Checkpoint covers: {ckpt_repr}\n"
            "Use a checkpoint whose type_map includes these elements, "
            "or filter the data to remove unsupported elements."
        )
