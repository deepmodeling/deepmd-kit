# SPDX-License-Identifier: LGPL-3.0-or-later
"""The model that takes the coordinates, cell and atom types as input
and predicts some property. The models are automatically generated from
atomic models by the `deepmd.dpmodel.make_model` method.

The `make_model` method does the reduction, auto-differentiation and
communication of the atomic properties according to output variable
definition `deepmd.dpmodel.OutputVariableDef`.

"""

import copy

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    PairTabAtomicModel,
)
from deepmd.pt.model.descriptor.descriptor import (
    Descriptor,
)
from deepmd.pt.model.task import (
    Fitting,
)
from deepmd.utils.spin import (
    Spin,
)

from .dp_model import (
    DPModel,
)
from .dp_zbl_model import (
    DPZBLModel,
)
from .ener_model import (
    EnergyModel,
)
from .make_hessian_model import (
    make_hessian_model,
)
from .make_model import (
    make_model,
)
from .model import (
    BaseModel,
)
from .spin_model import (
    SpinEnergyModel,
    SpinModel,
)


def get_zbl_model(model_params):
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    descriptor = Descriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", None)
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["mixed_types"] = descriptor.mixed_types()
    fitting_net["embedding_width"] = descriptor.get_dim_out()
    fitting_net["dim_descrpt"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = Fitting(**fitting_net)
    dp_model = DPAtomicModel(descriptor, fitting, type_map=model_params["type_map"])
    # pairtab
    filepath = model_params["use_srtab"]
    pt_model = PairTabAtomicModel(
        filepath, model_params["descriptor"]["rcut"], model_params["descriptor"]["sel"]
    )

    rmin = model_params["sw_rmin"]
    rmax = model_params["sw_rmax"]
    return DPZBLModel(
        dp_model,
        pt_model,
        rmin,
        rmax,
    )


def get_ener_model(model_params):
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    descriptor = Descriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", None)
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["mixed_types"] = descriptor.mixed_types()
    fitting_net["embedding_width"] = descriptor.get_dim_out()
    fitting_net["dim_descrpt"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = Fitting(**fitting_net)

    return EnergyModel(descriptor, fitting, type_map=model_params["type_map"])


def get_spin_model(model_params):
    model_params = copy.deepcopy(model_params)
    model_params["type_map"] += [item + "_spin" for item in model_params["type_map"]]
    ntypes = len(model_params["type_map"])  # include virtual spin and placeholder types
    spin = Spin(
        use_spin=model_params["spin"]["use_spin"],
        virtual_scale=model_params["spin"]["virtual_scale"],
    )

    pair_exclude_types = spin.get_pair_exclude_types(
        exclude_types=model_params["descriptor"].get("exclude_types", None)
    )
    model_params["descriptor"]["exclude_types"] = pair_exclude_types
    if "env_protection" not in model_params["descriptor"]:
        model_params["descriptor"]["env_protection"] = 1e-6
    if model_params["descriptor"]["type"] in ["se_e2_a"]:
        model_params["descriptor"]["sel"] += model_params["descriptor"]["sel"]

    atom_exclude_types = spin.get_atom_exclude_types(
        exclude_types=model_params["fitting_net"].get("exclude_types", None)
    )
    model_params["fitting_net"]["exclude_types"] = atom_exclude_types
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    descriptor = Descriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", None)
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["mixed_types"] = descriptor.mixed_types()
    fitting_net["embedding_width"] = descriptor.get_dim_out()
    fitting_net["dim_descrpt"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = Fitting(**fitting_net)
    backbone_model = DPModel(descriptor, fitting, type_map=model_params["type_map"])
    return SpinEnergyModel(backbone_model=backbone_model, spin=spin)


def get_model(model_params):
    model_type = model_params.get("type", "standard")
    if model_type == "standard":
        return get_ener_model(model_params)
    elif model_type == "spin":
        return get_spin_model(model_params)
    elif model_type == "zbl":
        return get_zbl_model(model_params)
    else:
        raise ValueError(f"unknown model type: {model_type}")


__all__ = [
    "BaseModel",
    "get_model",
    "DPModel",
    "EnergyModel",
    "SpinModel",
    "SpinEnergyModel",
    "DPZBLModel",
    "make_model",
    "make_hessian_model",
]
