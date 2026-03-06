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


def get_model(data: dict) -> BaseModel:
    """Get a model from a config dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    model_type = data.get("type", "standard")
    if model_type == "standard":
        return get_standard_model(data)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(data)
