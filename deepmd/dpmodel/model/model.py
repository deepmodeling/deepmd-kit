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


def get_model(data: dict) -> DPModel:
    """Get a DPModel from a dictionary.

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
