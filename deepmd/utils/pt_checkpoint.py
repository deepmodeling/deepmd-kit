# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities shared by PyTorch checkpoint backends."""

import logging
from collections.abc import (
    Mapping,
    MutableMapping,
)
from typing import (
    Any,
)

log = logging.getLogger(__name__)


def detect_pt_checkpoint_backend(checkpoint: Any) -> str | None:
    """Detect the parameter dialect of a raw PyTorch checkpoint.

    Parameters
    ----------
    checkpoint : Any
        A checkpoint payload or its unwrapped model state dictionary.

    Returns
    -------
    str or None
        ``"pt-expt"`` or ``"pt"`` when the parameter names identify one
        backend unambiguously, otherwise ``None``.
    """
    state_dict = checkpoint
    if isinstance(state_dict, Mapping) and "model" in state_dict:
        state_dict = state_dict["model"]
    if not isinstance(state_dict, Mapping):
        return None

    keys = tuple(key for key in state_dict if isinstance(key, str))

    # Weight names are decisive. pt_expt DPA4 also contains ordinary
    # torch-native ``.bias`` parameters, so bias names cannot override a
    # clear ``.w`` versus ``.matrix`` distinction.
    has_pt_expt_weight = any(key.endswith(".w") for key in keys)
    has_pt_weight = any(key.endswith(".matrix") for key in keys)
    if has_pt_expt_weight or has_pt_weight:
        if has_pt_expt_weight == has_pt_weight:
            return None
        return "pt-expt" if has_pt_expt_weight else "pt"

    # A lone ``.b`` is specific to pt_expt's NativeLayer. A lone ``.bias``
    # is not specific to pt because pt_expt models can contain torch-native
    # modules with that suffix, so it remains deliberately unclassified.
    has_pt_expt_bias = any(key.endswith(".b") for key in keys)
    has_pt_bias = any(key.endswith(".bias") for key in keys)
    if has_pt_expt_bias and not has_pt_bias:
        return "pt-expt"
    return None


def normalize_pt_model_params(model_params: dict[str, Any]) -> dict[str, Any]:
    """Apply compatibility fixes to model parameters loaded from a ``.pt`` file.

    Property fitting networks used to omit ``distinguish_types`` from their
    serialized model parameters.  The historical checkpoint behavior for
    those networks is ``False`` even though the constructor default remains
    ``True`` for newly-created models.  This helper fills in the omitted value
    while loading a checkpoint and leaves all other model parameters untouched.

    The model parameter layout can contain nested ``model_dict`` branches,
    ``shared_dict`` entries, or child models in ``models``.  Walking every
    mapping/list makes the compatibility fix apply to each of these layouts,
    including a shared fitting-network definition, without changing the
    behavior of regular model construction.

    Parameters
    ----------
    model_params : dict[str, Any]
        The ``_extra_state["model_params"]`` dictionary from a PyTorch
        checkpoint.  The dictionary is normalized in place.

    Returns
    -------
    dict[str, Any]
        ``model_params`` itself, after applying compatibility fixes.
    """
    seen: set[int] = set()
    changed = False

    def visit(value: Any, shared_dict: Mapping[str, Any] | None = None) -> None:
        nonlocal changed
        if isinstance(value, MutableMapping):
            # A shared config can be reachable through more than one path.
            # Avoid revisiting it (and emitting duplicate warnings) when the
            # object is referenced multiple times in the loaded parameters.
            value_id = id(value)
            if value_id in seen:
                return
            seen.add(value_id)

            local_shared_dict = value.get("shared_dict")
            if not isinstance(local_shared_dict, Mapping):
                local_shared_dict = shared_dict

            fitting_net = value.get("fitting_net")
            if isinstance(fitting_net, str) and local_shared_dict is not None:
                # Shared references may carry a parameter-sharing level (for
                # example ``"property_head:1"``); the level does not change
                # the fitting-net configuration to normalize.
                fitting_net = local_shared_dict.get(fitting_net.split(":", 1)[0])
            if (
                isinstance(fitting_net, MutableMapping)
                and str(fitting_net.get("type", "")).lower() == "property"
                and "distinguish_types" not in fitting_net
            ):
                fitting_net["distinguish_types"] = False
                changed = True

            for child in value.values():
                visit(child, local_shared_dict)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, shared_dict)

    visit(model_params)
    if changed:
        log.warning(
            "The `distinguish_types` option is missing from a property fitting "
            "network in a PyTorch checkpoint; defaulting to False for backward "
            "compatibility."
        )
    return model_params


def apply_pt_checkpoint_defaults(
    model_params: dict[str, Any],
    checkpoint_model_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill omitted compatibility fields in a model config from a checkpoint.

    This is used before constructing a model for ``init_model`` and
    ``restart_model``.  A checkpoint that predates the property fitting-net
    metadata field represents ``distinguish_types=False``.  Only fields that
    are omitted from both the current config and the checkpoint are filled, so
    an explicit value in the current training input remains authoritative.

    Parameters
    ----------
    model_params : dict[str, Any]
        The model parameters that will be used to construct the new model.
    checkpoint_model_params : Mapping[str, Any]
        The raw ``_extra_state["model_params"]`` from the checkpoint.

    Returns
    -------
    dict[str, Any]
        ``model_params`` itself, after applying compatible defaults.
    """
    changed = False
    seen: set[tuple[int, int]] = set()

    def resolve_fitting_net(
        fitting_net: Any,
        shared_dict: Mapping[str, Any] | None,
    ) -> Any:
        if isinstance(fitting_net, str) and shared_dict is not None:
            return shared_dict.get(fitting_net.split(":", 1)[0])
        return fitting_net

    def visit(
        value: Any,
        checkpoint_value: Any,
        shared_dict: Mapping[str, Any] | None = None,
        checkpoint_shared_dict: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal changed
        if isinstance(value, MutableMapping) and isinstance(checkpoint_value, Mapping):
            pair = (id(value), id(checkpoint_value))
            if pair in seen:
                return
            seen.add(pair)

            local_shared_dict = value.get("shared_dict")
            if not isinstance(local_shared_dict, Mapping):
                local_shared_dict = shared_dict
            local_checkpoint_shared_dict = checkpoint_value.get("shared_dict")
            if not isinstance(local_checkpoint_shared_dict, Mapping):
                local_checkpoint_shared_dict = checkpoint_shared_dict

            fitting_net = resolve_fitting_net(
                value.get("fitting_net"), local_shared_dict
            )
            checkpoint_fitting_net = resolve_fitting_net(
                checkpoint_value.get("fitting_net"), local_checkpoint_shared_dict
            )
            if (
                isinstance(fitting_net, MutableMapping)
                and isinstance(checkpoint_fitting_net, Mapping)
                and str(fitting_net.get("type", "")).lower() == "property"
                and str(checkpoint_fitting_net.get("type", "")).lower() == "property"
                and "distinguish_types" not in fitting_net
                and "distinguish_types" not in checkpoint_fitting_net
            ):
                fitting_net["distinguish_types"] = False
                changed = True

            for key, child in value.items():
                if key in checkpoint_value:
                    visit(
                        child,
                        checkpoint_value[key],
                        local_shared_dict,
                        local_checkpoint_shared_dict,
                    )
        elif isinstance(value, list) and isinstance(checkpoint_value, list):
            for child, checkpoint_child in zip(value, checkpoint_value, strict=False):
                visit(
                    child,
                    checkpoint_child,
                    shared_dict,
                    checkpoint_shared_dict,
                )

    visit(model_params, checkpoint_model_params)
    if changed:
        log.warning(
            "The `distinguish_types` option is missing from a property fitting "
            "network in the model config and checkpoint; defaulting to False "
            "for backward compatibility."
        )
    return model_params
