# SPDX-License-Identifier: LGPL-3.0-or-later
"""The model that takes the coordinates, cell and atom types as input
and predicts some property. The models are automatically generated from
atomic models by the `deepmd.dpmodel.make_model` method.

The `make_model` method does the reduction, auto-differentiation and
communication of the atomic properties according to output variable
definition `deepmd.dpmodel.OutputVariableDef`.

"""

import copy
import json

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    PairTabAtomicModel,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task import (
    BaseFitting,
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


def get_spin_model(model_params):
    model_params = copy.deepcopy(model_params)
    # include virtual spin and placeholder types
    model_params["type_map"] += [item + "_spin" for item in model_params["type_map"]]
    spin = Spin(
        use_spin=model_params["spin"]["use_spin"],
        virtual_scale=model_params["spin"]["virtual_scale"],
    )
    pair_exclude_types = spin.get_pair_exclude_types(
        exclude_types=model_params.get("pair_exclude_types", None)
    )
    model_params["pair_exclude_types"] = pair_exclude_types
    # for descriptor data stat
    model_params["descriptor"]["exclude_types"] = pair_exclude_types
    atom_exclude_types = spin.get_atom_exclude_types(
        exclude_types=model_params.get("atom_exclude_types", None)
    )
    model_params["atom_exclude_types"] = atom_exclude_types
    if (
        "env_protection" not in model_params["descriptor"]
        or model_params["descriptor"]["env_protection"] == 0.0
    ):
        model_params["descriptor"]["env_protection"] = 1e-6
    if model_params["descriptor"]["type"] in ["se_e2_a"]:
        model_params["descriptor"]["sel"] += model_params["descriptor"]["sel"]
    backbone_model = get_standard_model(model_params)
    return SpinEnergyModel(backbone_model=backbone_model, spin=spin)


def get_zbl_model(model_params):
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    descriptor = BaseDescriptor(**model_params["descriptor"])
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
    fitting = BaseFitting(**fitting_net)
    dp_model = DPAtomicModel(descriptor, fitting, type_map=model_params["type_map"])
    # pairtab
    filepath = model_params["use_srtab"]
    pt_model = PairTabAtomicModel(
        filepath,
        model_params["descriptor"]["rcut"],
        model_params["descriptor"]["sel"],
        type_map=model_params["type_map"],
    )

    rmin = model_params["sw_rmin"]
    rmax = model_params["sw_rmax"]
    atom_exclude_types = model_params.get("atom_exclude_types", [])
    pair_exclude_types = model_params.get("pair_exclude_types", [])
    return DPZBLModel(
        dp_model,
        pt_model,
        rmin,
        rmax,
        type_map=model_params["type_map"],
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )


def get_standard_model(model_params):
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    descriptor = BaseDescriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", None)
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["mixed_types"] = descriptor.mixed_types()
    fitting_net["embedding_width"] = descriptor.get_dim_emb()
    fitting_net["dim_descrpt"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = BaseFitting(**fitting_net)
    atom_exclude_types = model_params.get("atom_exclude_types", [])
    pair_exclude_types = model_params.get("pair_exclude_types", [])

    model = DPModel(
        descriptor,
        fitting,
        type_map=model_params["type_map"],
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )
    model.model_def_script = json.dumps(model_params)
    return model


def get_model(model_params):
    if "spin" in model_params:
        return get_spin_model(model_params)
    elif "use_srtab" in model_params:
        return get_zbl_model(model_params)
    else:
        return get_standard_model(model_params)


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
