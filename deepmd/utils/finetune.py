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

from deepmd.utils.model_branch_dict import (
    get_model_dict,
)

log = logging.getLogger(__name__)

_IGNORED_DESCRIPTOR_KEYS = frozenset({"trainable"})
_MISSING = object()
_MAX_DESCRIPTOR_CONFIG_DIFFS = 20
_MAX_CONFIG_VALUE_LENGTH = 200


def _infer_synthetic_type_count(descriptor: Mapping[str, Any]) -> int:
    """Infer a safe type count for descriptor-only normalization.

    The real model ``type_map`` is not available at every finetune warning call
    site. Use descriptor fields that explicitly encode per-type lists to avoid
    normalizing a 3+-type descriptor against the historical two-type stub. This
    is still a best-effort normalization helper: intentional type-map changes
    may still show up in type-count-dependent fields such as ``sel``.
    """
    type_count = 2
    for key in ("sel", "sel_a", "sel_r"):
        value = descriptor.get(key)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            type_count = max(type_count, len(value))
    exclude_types = descriptor.get("exclude_types")
    if isinstance(exclude_types, list):
        for pair in exclude_types:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(item, int) for item in pair)
            ):
                type_count = max(type_count, pair[0] + 1, pair[1] + 1)
    return type_count


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
            "type_map": [
                f"Type{ii}" for ii in range(_infer_synthetic_type_count(descriptor))
            ],
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
        # the minimal synthetic config above. If either side fails, compare raw
        # descriptor against raw descriptor; mixing normalized and raw values would
        # report implicit defaults as spurious differences.
        input_descriptor_cmp = input_descriptor
        pretrained_descriptor_cmp = pretrained_descriptor
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
        f"pretrained model (branch '{model_branch}'). Only descriptor parameters "
        "that are compatible with the pretrained model can be reused; "
        "incompatible parameters may be reinitialized, skipped, or rejected by "
        "backend-specific loading:\n"
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


class FinetuneRuleBuilder:
    """Build backend-independent fine-tuning config rules.

    The builder only handles model-config and branch-selection semantics. Backend
    wrappers are still responsible for loading pretrained model params, and
    backend trainers are responsible for copying tensor/state values.
    """

    def __init__(
        self,
        pretrained_model_params: dict[str, Any],
        target_model_config: dict[str, Any],
        *,
        model_branch: str = "",
        change_model_params: bool = True,
        default_branch: str = "Default",
        multitask_branch_error: str | None = None,
        missing_model_params_error: str | None = None,
    ) -> None:
        self.pretrained_model_params = pretrained_model_params
        self.target_model_config = target_model_config
        self.model_branch = model_branch
        self.change_model_params = change_model_params
        self.default_branch = default_branch
        self.multitask_branch_error = multitask_branch_error or (
            "Multi-task fine-tuning does not support command-line branch "
            "selection. Define 'finetune_head' in each model branch."
        )
        self.missing_model_params_error = missing_model_params_error or (
            "Cannot use --use-pretrain-script: the pretrained model does not "
            "contain full model params."
        )

    @property
    def target_is_multitask(self) -> bool:
        """Whether the target model config is multi-task."""
        return "model_dict" in self.target_model_config

    @property
    def pretrained_is_multitask(self) -> bool:
        """Whether the pretrained model params are multi-task."""
        return "model_dict" in self.pretrained_model_params

    def build(self) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem]]:
        """Return updated target model config and fine-tuning rules."""
        model_config = deepcopy(self.target_model_config)
        finetune_links: dict[str, FinetuneRuleItem] = {}

        if not self.target_is_multitask:
            model_branch = self.model_branch
            if model_branch == "" and "finetune_head" in model_config:
                model_branch = model_config["finetune_head"]
            model_config, finetune_rule = self.build_single_rule(
                model_config,
                model_branch=self.default_branch,
                model_branch_from=model_branch,
            )
            finetune_links[self.default_branch] = finetune_rule
            return model_config, finetune_links

        if self.model_branch != "":
            raise ValueError(self.multitask_branch_error)

        pretrained_keys = self._pretrained_keys()
        pretrained_aliases = self._pretrained_aliases()
        for model_key in model_config["model_dict"]:
            target_branch = model_config["model_dict"][model_key]
            resuming = False
            if (
                "finetune_head" in target_branch
                and target_branch["finetune_head"] != "RANDOM"
            ):
                model_branch_from = target_branch["finetune_head"]
                if model_branch_from not in pretrained_aliases:
                    raise ValueError(
                        f"'{model_branch_from}' head chosen to finetune does not "
                        "exist in the pretrained model. Available heads are: "
                        f"{pretrained_keys}"
                    )
            elif "finetune_head" not in target_branch and model_key in pretrained_keys:
                model_branch_from = model_key
                resuming = True
            else:
                model_branch_from = "RANDOM"

            model_config["model_dict"][model_key], finetune_rule = (
                self.build_single_rule(
                    target_branch,
                    model_branch=model_key,
                    model_branch_from=model_branch_from,
                )
            )
            finetune_rule.resuming = resuming
            finetune_links[model_key] = finetune_rule
        return model_config, finetune_links

    def build_single_rule(
        self,
        single_param_target: dict[str, Any],
        *,
        model_branch: str = "Default",
        model_branch_from: str = "",
    ) -> tuple[dict[str, Any], FinetuneRuleItem]:
        """Build a fine-tuning rule for one target branch."""
        single_config = deepcopy(single_param_target)
        new_fitting = False
        model_branch_chosen = self.default_branch

        if not self.pretrained_is_multitask:
            if model_branch_from not in ("", self.default_branch, "RANDOM"):
                raise ValueError(
                    "Single-task pretrained models only provide the "
                    f"{self.default_branch!r} branch, got {model_branch_from!r}."
                )
            single_config_chosen = deepcopy(self.pretrained_model_params)
            if model_branch_from == "RANDOM":
                new_fitting = True
        else:
            model_dict_params = self.pretrained_model_params["model_dict"]
            if model_branch_from in ["", "RANDOM"]:
                model_branch_chosen = next(iter(model_dict_params))
                new_fitting = True
                log.warning(
                    "The fitting net will be re-initialized instead of using the "
                    "pretrained fitting net. The bias_adjust_mode will be "
                    "set-by-statistic."
                )
            else:
                model_branch_chosen = model_branch_from
            model_alias_dict, _ = get_model_dict(model_dict_params)
            if model_branch_chosen not in model_alias_dict:
                raise ValueError(
                    f"No model branch or alias named '{model_branch_chosen}'. "
                    f"Available branches are {list(model_dict_params)}."
                )
            model_branch_chosen = model_alias_dict[model_branch_chosen]
            single_config_chosen = deepcopy(model_dict_params[model_branch_chosen])

        old_type_map = single_config_chosen["type_map"]
        new_type_map = single_config["type_map"]
        finetune_rule = FinetuneRuleItem(
            p_type_map=old_type_map,
            type_map=new_type_map,
            model_branch=model_branch_chosen,
            random_fitting=new_fitting,
        )
        if self.change_model_params:
            self._apply_pretrained_model_params(
                single_config,
                single_config_chosen,
                model_branch=model_branch,
                model_branch_chosen=model_branch_chosen,
                random_fitting=new_fitting,
            )
        return single_config, finetune_rule

    def _apply_pretrained_model_params(
        self,
        single_config: dict[str, Any],
        pretrained_config: dict[str, Any],
        *,
        model_branch: str,
        model_branch_chosen: str,
        random_fitting: bool,
    ) -> None:
        if "descriptor" not in pretrained_config:
            raise ValueError(self.missing_model_params_error)
        if not random_fitting and "fitting_net" not in pretrained_config:
            raise ValueError(self.missing_model_params_error)
        if "descriptor" in single_config:
            warn_descriptor_config_differences(
                single_config["descriptor"],
                pretrained_config["descriptor"],
                model_branch_chosen,
            )
        trainable_param = {
            "descriptor": single_config.get("descriptor", {}).get("trainable", True),
            "fitting_net": single_config.get("fitting_net", {}).get("trainable", True),
        }
        single_config["descriptor"] = deepcopy(pretrained_config["descriptor"])
        if not random_fitting:
            single_config["fitting_net"] = deepcopy(pretrained_config["fitting_net"])
        log.info(
            "Change the '%s' model configurations according to pretrained branch '%s'.",
            model_branch,
            model_branch_chosen,
        )
        for net_type, trainable in trainable_param.items():
            if net_type in single_config:
                single_config[net_type]["trainable"] = trainable
            else:
                single_config[net_type] = {"trainable": trainable}

    def _pretrained_keys(self) -> list[str]:
        if self.pretrained_is_multitask:
            return list(self.pretrained_model_params["model_dict"])
        return [self.default_branch]

    def _pretrained_aliases(self) -> dict[str, str]:
        if self.pretrained_is_multitask:
            model_alias_dict, _ = get_model_dict(
                self.pretrained_model_params["model_dict"]
            )
            return dict(model_alias_dict)
        return {self.default_branch: self.default_branch}


def get_finetune_rule_single(
    single_param_target: dict[str, Any],
    model_param_pretrained: dict[str, Any],
    from_multitask: bool = False,
    model_branch: str = "Default",
    model_branch_from: str = "",
    change_model_params: bool = False,
) -> tuple[dict[str, Any], FinetuneRuleItem]:
    """Build one backend-independent fine-tuning rule."""
    builder = FinetuneRuleBuilder(
        model_param_pretrained,
        single_param_target,
        change_model_params=change_model_params,
    )
    if from_multitask != builder.pretrained_is_multitask:
        raise ValueError("from_multitask does not match pretrained model params.")
    return builder.build_single_rule(
        single_param_target,
        model_branch=model_branch,
        model_branch_from=model_branch_from,
    )


def get_finetune_rules_from_model_params(
    pretrained_model_params: dict[str, Any],
    model_config: dict[str, Any],
    *,
    model_branch: str = "",
    change_model_params: bool = True,
    multitask_branch_error: str | None = None,
    missing_model_params_error: str | None = None,
) -> tuple[dict[str, Any], dict[str, FinetuneRuleItem]]:
    """Build fine-tuning rules from already-loaded pretrained model params."""
    return FinetuneRuleBuilder(
        pretrained_model_params,
        model_config,
        model_branch=model_branch,
        change_model_params=change_model_params,
        multitask_branch_error=multitask_branch_error,
        missing_model_params_error=missing_model_params_error,
    ).build()


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
