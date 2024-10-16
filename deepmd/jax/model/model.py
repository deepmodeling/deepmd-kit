# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)

from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)


def get_standard_model(data: dict):
    """Get a Model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    data = deepcopy(data)
    descriptor_type = data["descriptor"].pop("type")
    data["descriptor"]["type_map"] = data["type_map"]
    fitting_type = data["fitting_net"].pop("type")
    data["fitting_net"]["type_map"] = data["type_map"]
    descriptor = BaseDescriptor.get_class_by_type(descriptor_type)(
        **data["descriptor"],
    )
    fitting = BaseFitting.get_class_by_type(fitting_type)(
        ntypes=descriptor.get_ntypes(),
        dim_descrpt=descriptor.get_dim_out(),
        mixed_types=descriptor.mixed_types(),
        **data["fitting_net"],
    )
    return BaseModel.get_class_by_type(fitting_type)(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=data.get("pair_exclude_types", []),
    )


def get_model(data: dict):
    """Get a model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    model_type = data.get("type", "standard")
    if model_type == "standard":
        if "spin" in data:
            raise NotImplementedError("Spin model is not implemented yet.")
        else:
            return get_standard_model(data)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(data)
