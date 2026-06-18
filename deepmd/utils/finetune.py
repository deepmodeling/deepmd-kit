# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections.abc import (
    Mapping,
)
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

log = logging.getLogger(__name__)

_IGNORED_DESCRIPTOR_KEYS = frozenset({"trainable"})
_MISSING = object()
_MAX_DESCRIPTOR_CONFIG_DIFFS = 20
_MAX_CONFIG_VALUE_LENGTH = 200


def _normalize_descriptor_for_compare(
    descriptor: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Normalize a descriptor config so implicit defaults do not warn."""
    from deepmd.utils.argcheck import (
        normalize,
    )

    config = {
        "model": {
            "descriptor": deepcopy(dict(descriptor)),
            "fitting_net": {"neuron": [240, 240, 240]},
            "type_map": ["H", "O"],
        },
        "training": {"training_data": {"systems": ["fake"]}, "numb_steps": 100},
    }
    return normalize(config, multi_task=False)["model"]["descriptor"]


def _format_config_value(value: Any) -> str:
    text = repr(value)
    if len(text) > _MAX_CONFIG_VALUE_LENGTH:
        text = text[: _MAX_CONFIG_VALUE_LENGTH - 3] + "..."
    return text


def _iter_descriptor_config_differences(
    input_config: Any,
    pretrained_config: Any,
    prefix: str = "",
) -> list[tuple[str, Any, Any]]:
    differences: list[tuple[str, Any, Any]] = []
    if isinstance(input_config, Mapping) and isinstance(pretrained_config, Mapping):
        keys = sorted(set(input_config) | set(pretrained_config))
        for key in keys:
            if key in _IGNORED_DESCRIPTOR_KEYS:
                continue
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if key not in input_config:
                differences.append((key_path, _MISSING, pretrained_config[key]))
            elif key not in pretrained_config:
                differences.append((key_path, input_config[key], _MISSING))
            else:
                differences.extend(
                    _iter_descriptor_config_differences(
                        input_config[key], pretrained_config[key], key_path
                    )
                )
        return differences
    if input_config != pretrained_config:
        return [(prefix, input_config, pretrained_config)]
    return differences


def _descriptor_config_differences(
    input_descriptor: Mapping[str, Any],
    pretrained_descriptor: Mapping[str, Any],
) -> list[tuple[str, Any, Any]]:
    """Return meaningful descriptor differences, ignoring implicit defaults."""
    input_descriptor_cmp: Mapping[str, Any] = input_descriptor
    pretrained_descriptor_cmp: Mapping[str, Any] = pretrained_descriptor
    try:
        input_descriptor_cmp = _normalize_descriptor_for_compare(input_descriptor)
        pretrained_descriptor_cmp = _normalize_descriptor_for_compare(
            pretrained_descriptor
        )
    except Exception:
        # Some in-flight or legacy descriptor schemas may not be normalizable with
        # the minimal synthetic config above. In that case, still compare the raw
        # descriptors so users get a best-effort warning.
        pass
    return _iter_descriptor_config_differences(
        input_descriptor_cmp, pretrained_descriptor_cmp
    )


def _format_descriptor_differences(
    differences: list[tuple[str, Any, Any]],
    *,
    overwrite: bool,
) -> str:
    lines = []
    shown = differences[:_MAX_DESCRIPTOR_CONFIG_DIFFS]
    for key, input_value, pretrained_value in shown:
        input_text = (
            "(missing)"
            if input_value is _MISSING
            else _format_config_value(input_value)
        )
        pretrained_text = (
            "(missing)"
            if pretrained_value is _MISSING
            else _format_config_value(pretrained_value)
        )
        if overwrite:
            lines.append(f"  {key}: {input_text} -> {pretrained_text}")
        else:
            lines.append(f"  {key}: input={input_text}, pretrained={pretrained_text}")
    remaining = len(differences) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more difference(s)")
    return "\n".join(lines)


def warn_descriptor_config_differences(
    input_descriptor: Mapping[str, Any],
    pretrained_descriptor: Mapping[str, Any],
    model_branch: str = "Default",
) -> None:
    """Warn when ``--use-pretrain-script`` overwrites descriptor config."""
    differences = _descriptor_config_differences(
        input_descriptor, pretrained_descriptor
    )
    if not differences:
        return
    log.warning(
        "Descriptor configuration in input.json differs from pretrained model "
        f"(branch '{model_branch}'). The input descriptor configuration will be "
        "overwritten with the pretrained model's descriptor configuration "
        "except for the trainable flag:\n"
        + _format_descriptor_differences(differences, overwrite=True)
    )


def warn_configuration_mismatch_during_finetune(
    input_descriptor: Mapping[str, Any],
    pretrained_descriptor: Mapping[str, Any],
    model_branch: str = "Default",
) -> None:
    """Warn when fine-tuning loads only compatible descriptor parameters."""
    differences = _descriptor_config_differences(
        input_descriptor, pretrained_descriptor
    )
    if not differences:
        return
    log.warning(
        "Descriptor configuration mismatch detected between input.json and "
        f"pretrained model (branch '{model_branch}'). State dict initialization "
        "will only use compatible descriptor parameters from the pretrained model; "
        "other parameters keep their current initialization:\n"
        + _format_descriptor_differences(differences, overwrite=False)
    )


class FinetuneRuleItem:
    def __init__(
        self,
        p_type_map: list[str],
        type_map: list[str],
        model_branch: str = "Default",
        random_fitting: bool = False,
        resuming: bool = False,
    ) -> None:
        """
        The rules for fine-tuning the model from pretrained model.

        Parameters
        ----------
        p_type_map
            The type map from the pretrained model.
        type_map
            The newly defined type map.
        model_branch
            From which branch the model should be fine-tuned.
        random_fitting
            If true, the fitting net will be randomly initialized instead of inherit from the pretrained model.
        resuming
            If true, the model will just resume from model_branch without fine-tuning.
        """
        self.p_type_map = p_type_map
        self.type_map = type_map
        self.model_branch = model_branch
        self.random_fitting = random_fitting
        self.resuming = resuming
        self.update_type = self.p_type_map != self.type_map

    def get_index_mapping(self) -> list[int]:
        """Returns the mapping index of newly defined types to those in the pretrained model."""
        return get_index_between_two_maps(self.p_type_map, self.type_map)[0]

    def get_has_new_type(self) -> bool:
        """Returns whether there are unseen types in the new type_map."""
        return get_index_between_two_maps(self.p_type_map, self.type_map)[1]

    def get_model_branch(self) -> str:
        """Returns the chosen model branch."""
        return self.model_branch

    def get_random_fitting(self) -> bool:
        """Returns whether to use random fitting."""
        return self.random_fitting

    def get_resuming(self) -> bool:
        """Returns whether to only do resuming."""
        return self.resuming

    def get_update_type(self) -> bool:
        """Returns whether to update the type related params when loading from pretrained model with redundant types."""
        return self.update_type

    def get_pretrained_tmap(self) -> list[str]:
        """Returns the type map in the pretrained model."""
        return self.p_type_map

    def get_finetune_tmap(self) -> list[str]:
        """Returns the type map in the fine-tuned model."""
        return self.type_map


def get_index_between_two_maps(
    old_map: list[str],
    new_map: list[str],
) -> tuple[list[int], bool]:
    """Returns the mapping index of types in new_map to those in the old_map.

    Parameters
    ----------
    old_map : list[str]
        The old list of atom type names.
    new_map : list[str]
        The new list of atom type names.

    Returns
    -------
    index_map: list[int]
        List contains `len(new_map)` indices, where `index_map[i]` is the index of `new_map[i]` in `old_map`.
        If `new_map[i]` is not in the `old_map`, the index will be `i - len(new_map)`.
    has_new_type: bool
        Whether there are unseen types in the new type_map.
        If True, some type related params in the model, such as statistics, need to be extended
        to have a length of `len(old_map) + len(new_map)` in the type related dimension.
        Then positive indices from 0 to `len(old_map) - 1` will select old params of types in `old_map`,
        while negative indices from `-len(new_map)` to -1 will select new params of types in `new_map`.
    """
    missing_type = [i for i in new_map if i not in old_map]
    has_new_type = False
    if len(missing_type) > 0:
        has_new_type = True
        log.warning(
            f"These types are not in the pretrained model and related params will be randomly initialized: {missing_type}."
        )
    index_map = []
    for ii, t in enumerate(new_map):
        index_map.append(old_map.index(t) if t in old_map else ii - len(new_map))
    return index_map, has_new_type


def map_atom_exclude_types(
    atom_exclude_types: list[int],
    remap_index: list[int],
) -> list[int]:
    """Return the remapped atom_exclude_types according to remap_index.

    Parameters
    ----------
    atom_exclude_types : list[int]
        Exclude the atomic contribution of the given types.
    remap_index : list[int]
        The indices in the old type list that correspond to the types in the new type list.

    Returns
    -------
    remapped_atom_exclude_types: list[int]
        Remapped atom_exclude_types that only keeps the types in the new type list.

    """
    remapped_atom_exclude_types = [
        remap_index.index(i) for i in atom_exclude_types if i in remap_index
    ]
    return remapped_atom_exclude_types


def map_pair_exclude_types(
    pair_exclude_types: list[tuple[int, int]],
    remap_index: list[int],
) -> list[tuple[int, int]]:
    """Return the remapped atom_exclude_types according to remap_index.

    Parameters
    ----------
    pair_exclude_types : list[tuple[int, int]]
        Exclude the pair of atoms of the given types from computing the output
        of the atomic model.
    remap_index : list[int]
        The indices in the old type list that correspond to the types in the new type list.

    Returns
    -------
    remapped_pair_exclude_typess: list[tuple[int, int]]
        Remapped pair_exclude_types that only keeps the types in the new type list.

    """
    remapped_pair_exclude_typess = [
        (remap_index.index(pair[0]), remap_index.index(pair[1]))
        for pair in pair_exclude_types
        if pair[0] in remap_index and pair[1] in remap_index
    ]
    return remapped_pair_exclude_typess
