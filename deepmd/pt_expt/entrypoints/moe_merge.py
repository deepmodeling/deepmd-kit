# SPDX-License-Identifier: LGPL-3.0-or-later
"""Merge the experts of an MoE pt_expt checkpoint for one scenario.

The merged checkpoint holds a model of the same structure and type map whose
shared weights carry the task of the model and the experts at the
coefficients of the scenario. Without kept elements it is the plain model;
with them the named elements keep weights of their own. The checkpoint holds
the weights of the merged model, which freezes like any other model and
starts a training or a fine-tuning. What the merge changes is stated by
the descriptor and the fitting themselves (``merge_experts``,
``expert_options``), so every MoE component merges the same way.
"""

from __future__ import (
    annotations,
)

import logging
from copy import (
    deepcopy,
)

import torch

log = logging.getLogger(__name__)


def moe_merge(
    input_file: str,
    output: str,
    elements: list[str] | None = None,
    charge_spin: list[float] | None = None,
    head: str | None = None,
) -> None:
    """
    Merge the experts of an MoE checkpoint for one scenario.

    Parameters
    ----------
    input_file
        Checkpoint of an MoE model.
    output
        Path of the merged checkpoint.
    elements
        Elements that keep weights of their own in the merged model, or
        ``["all"]`` for every element with them; None or empty merges into the
        plain model.
    charge_spin
        Frame charge and spin multiplicity; None takes the model default.
    head
        Branch of a multi-task checkpoint, which is the task of the scenario.

    Raises
    ------
    ValueError
        If the checkpoint is not a pt_expt training checkpoint, the branch of
        a multi-task checkpoint is not given or unknown, a branch is given for
        a single-task checkpoint, ``all`` is given next to other elements, or
        the model cannot be merged for the scenario.
    NotImplementedError
        If the model is not built from one top-level descriptor and fitting.
    """
    from deepmd.dpmodel.model.base_model import BaseModel as BaseModelDP
    from deepmd.dpmodel.model.expert_merge import (
        merge_experts,
        merged_model_params,
    )
    from deepmd.dpmodel.utils.expert_route import (
        ALL_ELEMENTS,
        ExpertScenario,
    )
    from deepmd.pt_expt.entrypoints.main import (
        load_checkpoint_model,
    )
    from deepmd.pt_expt.model import (
        BaseModel,
    )
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )

    # === Step 1. Load the MoE model ===
    state_dict = torch.load(input_file, map_location="cpu", weights_only=True)
    state_dict = state_dict.get("model", state_dict)
    extra_state = state_dict.get("_extra_state")
    if not isinstance(extra_state, dict) or "model_params" not in extra_state:
        raise ValueError(
            f"'{input_file}' is not a pt_expt training checkpoint: its model state "
            "has no '_extra_state.model_params'"
        )
    branches = extra_state["model_params"].get("model_dict")
    if branches is None and head is not None:
        raise ValueError(
            f"--task selects a branch of a multi-task checkpoint; '{input_file}' "
            "is a single-task checkpoint"
        )
    if branches is not None and head not in branches:
        raise ValueError(
            f"a multi-task checkpoint needs --task among {list(branches)}, got {head!r}"
        )
    model, params = load_checkpoint_model(state_dict, head)

    # === Step 2. Merge the experts on the array-API model ===
    elements = list(elements or [])
    if ALL_ELEMENTS in elements and len(elements) > 1:
        raise ValueError(
            f"`{ALL_ELEMENTS}` names every element with weights of its own and "
            f"stands alone; got {elements}"
        )
    scenario = ExpertScenario(
        charge_spin=charge_spin,
        elements=ALL_ELEMENTS if elements == [ALL_ELEMENTS] else tuple(elements),
    )
    merged = BaseModelDP.deserialize(model.serialize())
    merge_experts(merged, scenario)

    # === Step 3. Build the merged model from its parameters ===
    params = merged_model_params(params, merged)
    if head is not None:
        # The merged model continues the task of its branch.
        params["task"] = head
    result = get_model(deepcopy(params)).to(DEVICE)
    result.load_state_dict(
        BaseModel.deserialize(merged.serialize()).to(DEVICE).state_dict()
    )
    torch.save(
        {"model": ModelWrapper(result, model_params=params).state_dict()}, output
    )
    kept = "no element" if not elements else " ".join(elements)
    log.info("Saved the merged model to %s (weights of their own: %s)", output, kept)
