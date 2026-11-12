# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model-level operations on a mixture of experts (MoE).

An MoE descriptor or fitting (:mod:`deepmd.dpmodel.utils.expert_route`)
declares how it merges its experts for one scenario (``merge_experts``) and
which options of its input configure its MoE (``expert_options``). The
functions here apply these declarations to the learned atomic models of a
model, so that a model is merged, and the input that rebuilds the merged
model is derived, without naming a descriptor or a fitting class.
"""

from __future__ import (
    annotations,
)

from copy import (
    deepcopy,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.utils.expert_route import (
    element_names,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.expert_route import (
        ExpertScenario,
    )


def learned_atomic_models(atomic_model: Any) -> list[DPAtomicModel]:
    """The learned atomic models of an atomic model, including bridged children."""
    if isinstance(atomic_model, DPAtomicModel):
        return [atomic_model]
    if isinstance(atomic_model, LinearEnergyAtomicModel):
        return [
            model for model in atomic_model.models if isinstance(model, DPAtomicModel)
        ]
    return []


def expert_atomic_models(model: Any) -> list[DPAtomicModel]:
    """The learned atomic models of a model that have MoE."""
    return [
        atomic_model
        for atomic_model in learned_atomic_models(model.atomic_model)
        if atomic_model.has_experts()
    ]


def has_expert_router(model: Any) -> bool:
    """Whether a router mixes experts in a learned atomic model of a model."""
    return any(
        atomic_model.has_expert_router()
        for atomic_model in learned_atomic_models(model.atomic_model)
    )


def fitting_has_experts(model: Any) -> bool:
    """Whether a learned fitting network of a model has MoE."""
    return any(
        atomic_model.fitting_net.has_experts()
        for atomic_model in learned_atomic_models(model.atomic_model)
    )


def branch_local_parameters(model: Any) -> list[Any]:
    """
    Parameters of a model that a branch of a multi-task model owns.

    Parameters
    ----------
    model
        Model whose learned descriptors and fittings may be shared with the
        other branches.

    Returns
    -------
    list
        The parameters that the descriptors and the fittings keep per branch
        (``branch_local_parameters``), which are the task of the model.
    """
    return [
        parameter
        for atomic_model in learned_atomic_models(model.atomic_model)
        for component in (atomic_model.descriptor, atomic_model.fitting_net)
        for parameter in component.branch_local_parameters()
    ]


def element_banks(model: Any) -> list[tuple[list[Any], list[int]]]:
    """
    Element banks of a model, with the element slot of every atom type of the model.

    A learned atomic model whose atom types are those of the model, in any
    order, reads the rows of its banks only for the types of the atoms of the
    model (``element_banks`` of its descriptor and fitting). A learned atomic
    model with types of its own, such as the virtual types of a spin model,
    reads rows that the atom types of the model do not name; its banks are
    left out.

    Parameters
    ----------
    model
        Model whose learned descriptors and fittings may have element banks.

    Returns
    -------
    list
        For every descriptor and fitting with element banks, its banks and the
        element slot of every atom type of the model, in the type order of
        the model.
    """
    type_map = model.get_type_map()
    found = []
    for atomic_model in learned_atomic_models(model.atomic_model):
        own = atomic_model.get_type_map()
        if sorted(own) != sorted(type_map):
            continue
        order = [own.index(name) for name in type_map]
        for component in (atomic_model.descriptor, atomic_model.fitting_net):
            banks, slots = component.element_banks()
            if banks:
                found.append((banks, [slots[index] for index in order]))
    return found


def slot_elements(model: Any) -> list[list[str] | None]:
    """
    Elements that the element increments of a model belong to.

    Parameters
    ----------
    model
        Model whose descriptors and fittings state their MoE with the
        ``moe`` option.

    Returns
    -------
    list
        For the descriptor and the fitting of every learned atomic model, in
        that order, the elements of the element slots ``1, 2, ...``; None for
        a component without element increments.
    """
    type_map = model.get_type_map()
    elements = []
    for atomic_model in learned_atomic_models(model.atomic_model):
        for component in (atomic_model.descriptor, atomic_model.fitting_net):
            moe = component.expert_options().get("moe")
            elements.append(
                None
                if moe is None or moe["element"] is False
                else element_names(moe["element"], type_map)
            )
    return elements


def check_slot_elements(model: Any, source: Any) -> None:
    """
    Check that a model can take the element increments of another model.

    The increments are stored by element slot, so both models must give the
    slots of a component to the same elements.

    Parameters
    ----------
    model
        Model that takes the weights.
    source
        Model that the weights come from.

    Raises
    ------
    ValueError
        If a component has element increments in both models and their slots
        belong to different elements.
    """
    for ours, theirs in zip(slot_elements(model), slot_elements(source), strict=False):
        if ours is not None and theirs is not None and ours != theirs:
            raise ValueError(
                "the element weights of the pretrained model belong to "
                f"{theirs}, in this order, and those of this model to {ours}; "
                "state `moe.element` as the list of the pretrained model"
            )


def merge_experts(model: Any, scenario: ExpertScenario) -> None:
    """
    Merge the experts of an MoE model for one scenario, in place.

    Every learned atomic model with MoE takes the descriptor and
    the fitting of the scenario. A scenario that the model cannot be merged
    for leaves the model unchanged.

    Parameters
    ----------
    model
        dpmodel MoE model.
    scenario
        The charge/spin state of the frames and the elements that keep their
        own weights.

    Raises
    ------
    ValueError
        If the model has no MoE, the scenario gives a charge/spin
        state that no router of a descriptor reads, or it names elements
        while no component of the model has weights for elements.
    """
    atomic_models = expert_atomic_models(model)
    if not atomic_models:
        raise ValueError("the model has no MoE to merge")
    if scenario.charge_spin is not None and not any(
        atomic_model.descriptor.has_expert_router() for atomic_model in atomic_models
    ):
        raise ValueError(
            "no router of the model reads the charge and spin multiplicity of a "
            "frame; merge it without a charge/spin state"
        )
    merged = [
        (
            atomic_model.descriptor.merge_experts(scenario),
            atomic_model.fitting_net.merge_experts(scenario),
        )
        for atomic_model in atomic_models
    ]
    if scenario.elements and not any(
        component.has_experts() for pair in merged for component in pair
    ):
        raise ValueError(
            "the model has no weights of elements to keep; merge it without "
            "naming elements"
        )
    for atomic_model, (descriptor, fitting) in zip(atomic_models, merged, strict=True):
        atomic_model.descriptor = descriptor
        atomic_model.fitting_net = fitting


def merged_model_params(params: dict[str, Any], model: Any) -> dict[str, Any]:
    """
    Model parameters that build a merged model.

    The merged descriptor and fitting state the options that configure their
    MoE, and the descriptor the charge/spin state it was merged for.

    Parameters
    ----------
    params : dict
        Parameters of the MoE model.
    model
        The merged dpmodel model.

    Returns
    -------
    dict
        The parameters of the merged model.

    Raises
    ------
    NotImplementedError
        If the model is not built from one top-level descriptor and fitting.
    """
    atomic_models = learned_atomic_models(model.atomic_model)
    if len(atomic_models) != 1 or not {"descriptor", "fitting_net"} <= set(params):
        raise NotImplementedError(
            "the parameters of a merged model are derived for standard and "
            "bridged models with one top-level descriptor and fitting"
        )
    (atomic_model,) = atomic_models
    params = deepcopy(params)
    for key, component in (
        ("descriptor", atomic_model.descriptor),
        ("fitting_net", atomic_model.fitting_net),
    ):
        for option, value in component.expert_options().items():
            if value is None:
                params[key].pop(option, None)
            else:
                params[key][option] = deepcopy(value)
    if atomic_model.descriptor.has_chg_spin_ebd():
        params["descriptor"]["default_chg_spin"] = (
            atomic_model.descriptor.get_default_chg_spin()
        )
    # The merge changes the inference default, not the reference the preset
    # already resolved: carry the resolved preset of the model, which holds
    # the reference condition of every element.
    params["preset_out_bias"] = deepcopy(atomic_model.preset_out_bias)
    return params
