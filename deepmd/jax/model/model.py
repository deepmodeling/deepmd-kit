# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)

from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.jax.fitting.fitting import (
    EnergyFittingNet,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_zbl_model import (
    DPZBLModel,
)


def get_standard_model(data: dict):
    """Get a Model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    data = deepcopy(data)
    if "type_embedding" in data:
        raise ValueError(
            "In the JAX backend, type_embedding is not at the model level, but within the descriptor. See type embedding documentation for details."
        )
    descriptor_type = data["descriptor"].pop("type")
    data["descriptor"]["type_map"] = data["type_map"]
    data["descriptor"]["ntypes"] = len(data["type_map"])
    fitting_type = data["fitting_net"].pop("type")
    data["fitting_net"]["type_map"] = data["type_map"]
    descriptor = BaseDescriptor.get_class_by_type(descriptor_type)(
        **data["descriptor"],
    )
    if fitting_type in {"dipole", "polar"}:
        data["fitting_net"]["embedding_width"] = descriptor.get_dim_emb()
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


def get_zbl_model(data: dict) -> DPZBLModel:
    data["descriptor"]["ntypes"] = len(data["type_map"])
    descriptor_type = data["descriptor"].pop("type")
    descriptor = BaseDescriptor.get_class_by_type(descriptor_type)(**data["descriptor"])
    fitting_type = data["fitting_net"].pop("type")
    if fitting_type == "ener":
        fitting = EnergyFittingNet(
            ntypes=descriptor.get_ntypes(),
            dim_descrpt=descriptor.get_dim_out(),
            mixed_types=descriptor.mixed_types(),
            **data["fitting_net"],
        )
    else:
        raise ValueError(f"Unknown fitting type {fitting_type}")

    dp_model = DPAtomicModel(descriptor, fitting, type_map=data["type_map"])
    # pairtab
    filepath = data["use_srtab"]
    pt_model = PairTabAtomicModel(
        filepath,
        data["descriptor"]["rcut"],
        data["descriptor"]["sel"],
        type_map=data["type_map"],
    )
    rmin = data["sw_rmin"]
    rmax = data["sw_rmax"]
    atom_exclude_types = data.get("atom_exclude_types", [])
    pair_exclude_types = data.get("pair_exclude_types", [])
    return DPZBLModel(
        dp_model,
        pt_model,
        rmin,
        rmax,
        type_map=data["type_map"],
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
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
        elif "use_srtab" in data:
            return get_zbl_model(data)
        else:
            return get_standard_model(data)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(data)
