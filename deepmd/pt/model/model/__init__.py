# SPDX-License-Identifier: LGPL-3.0-or-later
"""The model that takes the coordinates, cell and atom types as input
and predicts some property. The models are automatically generated from
atomic models by the `deepmd.dpmodel.make_model` method.

The `make_model` method does the reduction, auto-differentiation and
communication of the atomic properties according to output variable
definition `deepmd.dpmodel.OutputVariableDef`.

All models should be inherited from :class:`deepmd.pt.model.model.model.BaseModel`.
Models generated by `make_model` have already done it.
"""

import copy
import json

import numpy as np

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

from .dipole_model import (
    DipoleModel,
)
from .dos_model import (
    DOSModel,
)
from .dp_model import (
    DPModelCommon,
)
from .dp_zbl_model import (
    DPZBLModel,
)
from .ener_model import (
    EnergyModel,
)
from .frozen import (
    FrozenModel,
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
from .polar_model import (
    PolarModel,
)
from .spin_model import (
    SpinEnergyModel,
    SpinModel,
)


def get_spin_model(model_params):
    model_params = copy.deepcopy(model_params)
    if not model_params["spin"]["use_spin"] or isinstance(
        model_params["spin"]["use_spin"][0], int
    ):
        use_spin = np.full(len(model_params["type_map"]), False)
        use_spin[model_params["spin"]["use_spin"]] = True
        model_params["spin"]["use_spin"] = use_spin.tolist()
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
        # only expand sel for se_e2_a
        model_params["descriptor"]["sel"] += model_params["descriptor"]["sel"]
    backbone_model = get_standard_model(model_params)
    return SpinEnergyModel(backbone_model=backbone_model, spin=spin)


def get_zbl_model(model_params):
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    model_params["descriptor"]["type_map"] = copy.deepcopy(model_params["type_map"])
    descriptor = BaseDescriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", None)
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["type_map"] = copy.deepcopy(model_params["type_map"])
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
    model_params_old = model_params
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    model_params["descriptor"]["type_map"] = copy.deepcopy(model_params["type_map"])
    descriptor = BaseDescriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", {})
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["type_map"] = copy.deepcopy(model_params["type_map"])
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
    atom_exclude_types = model_params.get("atom_exclude_types", [])
    pair_exclude_types = model_params.get("pair_exclude_types", [])

    if fitting_net["type"] == "dipole":
        modelcls = DipoleModel
    elif fitting_net["type"] == "polar":
        modelcls = PolarModel
    elif fitting_net["type"] == "dos":
        modelcls = DOSModel
    elif fitting_net["type"] in ["ener", "direct_force_ener"]:
        modelcls = EnergyModel
    else:
        raise RuntimeError(f"Unknown fitting type: {fitting_net['type']}")

    model = modelcls(
        descriptor=descriptor,
        fitting=fitting,
        type_map=model_params["type_map"],
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )
    model.model_def_script = json.dumps(model_params_old)
    return model


def get_model(model_params):
    model_type = model_params.get("type", "standard")
    if model_type == "standard":
        if "spin" in model_params:
            return get_spin_model(model_params)
        elif "use_srtab" in model_params:
            return get_zbl_model(model_params)
        else:
            return get_standard_model(model_params)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(model_params)


__all__ = [
    "BaseModel",
    "get_model",
    "DPModelCommon",
    "EnergyModel",
    "DipoleModel",
    "PolarModel",
    "DOSModel",
    "FrozenModel",
    "SpinModel",
    "SpinEnergyModel",
    "DPZBLModel",
    "make_model",
    "make_hessian_model",
]
