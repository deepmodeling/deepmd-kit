# SPDX-License-Identifier: LGPL-3.0-or-later
import copy

from deepmd.pt.model.descriptor.descriptor import (
    Descriptor,
)
from deepmd.pt.model.task import (
    Fitting,
)

from .ener import (
    EnergyModel,
)
from .model import (
    BaseModel,
)


def get_model(model_params, sampled=None):
    model_params = copy.deepcopy(model_params)
    ntypes = len(model_params["type_map"])
    # descriptor
    model_params["descriptor"]["ntypes"] = ntypes
    descriptor = Descriptor(**model_params["descriptor"])
    # fitting
    fitting_net = model_params.get("fitting_net", None)
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntype()
    fitting_net["distinguish_types"] = descriptor.distinguish_types()
    fitting_net["embedding_width"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = Fitting(**fitting_net)

    return EnergyModel(
        descriptor,
        fitting,
        type_map=model_params["type_map"],
        type_embedding=model_params.get("type_embedding", None),
        resuming=model_params.get("resuming", False),
        stat_file_dir=model_params.get("stat_file_dir", None),
        stat_file_path=model_params.get("stat_file_path", None),
        sampled=sampled,
    )


__all__ = [
    "BaseModel",
    "EnergyModel",
    "get_model",
]
