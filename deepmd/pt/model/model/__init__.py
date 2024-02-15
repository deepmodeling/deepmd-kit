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
    fitting_net["distinguish_types"] = descriptor.distinguish_types()
    fitting_net["embedding_width"] = descriptor.get_dim_out()
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


def get_model(model_params):
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    descriptor = Descriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", None)
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["distinguish_types"] = descriptor.distinguish_types()
    fitting_net["embedding_width"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = Fitting(**fitting_net)

    return EnergyModel(descriptor, fitting, type_map=model_params["type_map"])


__all__ = [
    "BaseModel",
    "get_model",
    "get_zbl_model",
    "DPModel",
    "EnergyModel",
    "DPZBLModel",
    "make_model",
    "make_hessian_model",
]
