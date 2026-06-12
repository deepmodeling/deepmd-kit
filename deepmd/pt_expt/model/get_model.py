# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model factory for the pt_expt backend.

Mirrors ``deepmd.dpmodel.model.model`` but uses the pt_expt
``BaseDescriptor`` / ``BaseFitting`` registries so that the
constructed objects are ``torch.nn.Module`` subclasses.
"""

import copy
from typing import (
    Any,
)

from deepmd.pt_expt.descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.fitting import (
    BaseFitting,
)

# Import from submodules directly to avoid circular import via __init__.py
from deepmd.pt_expt.model.dipole_model import (
    DipoleModel,
)
from deepmd.pt_expt.model.dos_model import (
    DOSModel,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.model.polar_model import (
    PolarModel,
)
from deepmd.pt_expt.model.property_model import (
    PropertyModel,
)
from deepmd.pt_expt.model.spin_ener_model import (
    SpinEnergyModel,
)
from deepmd.utils.spin import (
    Spin,
)


def _get_standard_model_components(
    data: dict[str, Any],
    ntypes: int,
) -> tuple:
    """Build descriptor and fitting from config dict."""
    # descriptor
    data["descriptor"]["ntypes"] = ntypes
    data["descriptor"]["type_map"] = copy.deepcopy(data["type_map"])
    descriptor = BaseDescriptor(**data["descriptor"])

    # fitting
    fitting_net = data.get("fitting_net", {})
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["type_map"] = copy.deepcopy(data["type_map"])
    fitting_net["mixed_types"] = descriptor.mixed_types()
    if fitting_net["type"] in ["dipole", "polar"]:
        fitting_net["embedding_width"] = descriptor.get_dim_emb()
    fitting_net["dim_descrpt"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = BaseFitting(**fitting_net)
    return descriptor, fitting, fitting_net["type"]


def get_standard_model(data: dict) -> EnergyModel:
    """Get a standard model from a config dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    data = copy.deepcopy(data)
    ntypes = len(data["type_map"])
    descriptor, fitting, fitting_net_type = _get_standard_model_components(data, ntypes)
    atom_exclude_types = data.get("atom_exclude_types", [])
    pair_exclude_types = data.get("pair_exclude_types", [])

    if fitting_net_type == "dipole":
        modelcls = DipoleModel
    elif fitting_net_type == "polar":
        modelcls = PolarModel
    elif fitting_net_type == "dos":
        modelcls = DOSModel
    elif fitting_net_type in ["ener", "direct_force_ener"]:
        modelcls = EnergyModel
    elif fitting_net_type == "property":
        modelcls = PropertyModel
    else:
        raise RuntimeError(f"Unknown fitting type: {fitting_net_type}")

    model = modelcls(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )
    return model


def get_sezm_model(data: dict) -> EnergyModel:
    """Build a pt_expt energy model from a DPA4/SeZM model config.

    Mirrors :func:`deepmd.pt.model.model.get_sezm_model` so that dpa4/sezm
    training configs are interchangeable between the pt and pt_expt backends.
    The pt-only SeZM extensions (bridging, LoRA, compile, spin) are not
    supported here and raise ``NotImplementedError``.
    """
    data = copy.deepcopy(data)
    if "spin" in data:
        raise NotImplementedError(
            "Spin DPA4/SeZM models are not supported in the pt_expt backend."
        )
    if str(data.get("bridging_method", "none")).lower() != "none":
        raise NotImplementedError(
            "`bridging_method` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    if data.get("lora") is not None:
        raise NotImplementedError(
            "`lora` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    if data.get("use_compile"):
        raise NotImplementedError(
            "`use_compile` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    data.pop("type", None)
    data.setdefault("descriptor", {})
    data.setdefault("fitting_net", {})
    data["descriptor"].setdefault("type", "dpa4")
    data["fitting_net"].setdefault("type", "dpa4_ener")

    # keep descriptor.exclude_types and model pair_exclude_types consistent
    descriptor_exclude_types = [
        list(pair) for pair in (data["descriptor"].get("exclude_types") or [])
    ]
    if "pair_exclude_types" in data:
        pair_exclude_types = [list(pair) for pair in (data["pair_exclude_types"] or [])]
        if descriptor_exclude_types and descriptor_exclude_types != pair_exclude_types:
            raise ValueError(
                "SeZM `pair_exclude_types` and `descriptor.exclude_types` must match "
                "when both are provided."
            )
    else:
        pair_exclude_types = descriptor_exclude_types
    data["pair_exclude_types"] = pair_exclude_types
    data["descriptor"]["exclude_types"] = copy.deepcopy(pair_exclude_types)

    ntypes = len(data["type_map"])
    descriptor, fitting, _ = _get_standard_model_components(data, ntypes)
    return EnergyModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=pair_exclude_types,
    )


def get_linear_model(model_params: dict) -> BaseModel:
    """Get a linear energy model from a config dictionary.

    Parameters
    ----------
    model_params : dict
        The model parameters.
    """
    from deepmd.dpmodel.atomic_model.dp_atomic_model import (
        DPAtomicModel,
    )
    from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
        PairTabAtomicModel,
    )

    from .dp_linear_model import (
        LinearEnergyModel,
    )

    model_params = copy.deepcopy(model_params)
    weights = model_params.get("weights", "mean")
    list_of_models = []
    ntypes = len(model_params["type_map"])
    for sub_model_params in model_params["models"]:
        if "type_map" not in sub_model_params:
            sub_model_params["type_map"] = model_params["type_map"]
        if "descriptor" in sub_model_params:
            sub_model_params["descriptor"]["ntypes"] = ntypes
            descriptor, fitting, _ = _get_standard_model_components(
                sub_model_params, ntypes
            )
            list_of_models.append(
                DPAtomicModel(descriptor, fitting, type_map=model_params["type_map"])
            )
        else:
            assert (
                "type" in sub_model_params and sub_model_params["type"] == "pairtab"
            ), "Sub-models in LinearEnergyModel must be a DPModel or a PairTable Model"
            list_of_models.append(
                PairTabAtomicModel(
                    sub_model_params["tab_file"],
                    sub_model_params["rcut"],
                    sub_model_params["sel"],
                    type_map=model_params["type_map"],
                )
            )

    atom_exclude_types = model_params.get("atom_exclude_types", [])
    pair_exclude_types = model_params.get("pair_exclude_types", [])
    return LinearEnergyModel(
        models=list_of_models,
        type_map=model_params["type_map"],
        weights=weights,
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )


def get_spin_model(data: dict) -> SpinEnergyModel:
    """Build a pt_expt spin energy model from a config dictionary.

    Mirrors :func:`deepmd.dpmodel.model.model.get_spin_model`: expands the
    type map and descriptor sel for virtual spin atoms, then wraps the
    backbone EnergyModel as a :class:`SpinEnergyModel`.
    """
    data = copy.deepcopy(data)
    data["type_map"] += [item + "_spin" for item in data["type_map"]]
    spin = Spin(
        use_spin=data["spin"]["use_spin"],
        virtual_scale=data["spin"]["virtual_scale"],
    )
    pair_exclude_types = spin.get_pair_exclude_types(
        exclude_types=data.get("pair_exclude_types", None)
    )
    data["pair_exclude_types"] = pair_exclude_types
    data["descriptor"]["exclude_types"] = pair_exclude_types
    atom_exclude_types = spin.get_atom_exclude_types(
        exclude_types=data.get("atom_exclude_types", None)
    )
    data["atom_exclude_types"] = atom_exclude_types
    if "env_protection" not in data["descriptor"]:
        data["descriptor"]["env_protection"] = 1e-6
    if data["descriptor"]["type"] in ["se_e2_a"]:
        data["descriptor"]["sel"] += data["descriptor"]["sel"]
    backbone_model = get_standard_model(data)
    return SpinEnergyModel(backbone_model=backbone_model, spin=spin)


def get_model(data: dict) -> BaseModel:
    """Get a model from a config dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    model_type = data.get("type", "standard")
    if model_type == "standard":
        if "spin" in data:
            return get_spin_model(data)
        return get_standard_model(data)
    elif model_type == "linear_ener":
        return get_linear_model(data)
    elif model_type in ("dpa4", "DPA4", "sezm", "SeZM"):
        return get_sezm_model(data)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(data)
