# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting.ener_fitting import (
    EnergyFittingNet,
)
from deepmd.dpmodel.model.dp_model import (
    DPModel,
)
from deepmd.dpmodel.model.spin_model import (
    SpinModel,
)
from deepmd.utils.spin import (
    Spin,
)


def get_standard_model(data: dict) -> DPModel:
    """Get a standard DPModel from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    descriptor_type = data["descriptor"].pop("type")
    fitting_type = data["fitting_net"].pop("type")
    if descriptor_type == "se_e2_a":
        descriptor = DescrptSeA(
            **data["descriptor"],
        )
    else:
        raise ValueError(f"Unknown descriptor type {descriptor_type}")
    if fitting_type == "ener":
        fitting = EnergyFittingNet(
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            **data["fitting_net"],
        )
    else:
        raise ValueError(f"Unknown fitting type {fitting_type}")
    return DPModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=data.get("pair_exclude_types", []),
    )


def get_spin_model(data: dict) -> SpinModel:
    """Get a spin model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    descriptor_type = data["descriptor"].pop("type")
    fitting_type = data["fitting_net"].pop("type")
    data["type_map"] += [item + "_spin" for item in data["type_map"]]
    ntypes = len(data["type_map"])  # include virtual spin and placeholder types
    spin = Spin(
        use_spin=data["spin"]["use_spin"],
        virtual_scale=data["spin"]["virtual_scale"],
    )
    pair_exclude_types = spin.get_pair_exclude_types(
        exclude_types=data["descriptor"].get("exclude_types", None)
    )
    data["descriptor"]["exclude_types"] = pair_exclude_types
    if "env_protection" not in data["descriptor"]:
        data["descriptor"]["env_protection"] = 1e-6
    if descriptor_type in ["se_e2_a"]:
        data["descriptor"]["sel"] += data["descriptor"]["sel"]

    atom_exclude_types = spin.get_atom_exclude_types(
        exclude_types=data["fitting_net"].get("exclude_types", None)
    )
    data["fitting_net"]["exclude_types"] = atom_exclude_types
    data["descriptor"]["ntypes"] = ntypes
    if descriptor_type == "se_e2_a":
        data["descriptor"].pop("ntypes")
        descriptor = DescrptSeA(
            **data["descriptor"],
        )
    else:
        raise ValueError(f"Unknown descriptor type {descriptor_type}")
    if fitting_type == "ener":
        fitting = EnergyFittingNet(
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            mixed_types=descriptor.mixed_types(),
            **data["fitting_net"],
        )
    else:
        raise ValueError(f"Unknown fitting type {fitting_type}")
    backbone_model = DPModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
    )
    return SpinModel(backbone_model=backbone_model, spin=spin)


def get_model(data: dict):
    """Get a model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    model_type = data.get("type", "standard")
    if model_type == "standard":
        return get_standard_model(data)
    elif model_type == "spin":
        return get_spin_model(data)
    else:
        raise ValueError(f"unknown model type: {model_type}")
