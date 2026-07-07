# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)

from deepmd.tf2.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.tf2.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.tf2.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.tf2.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.tf2.fitting.fitting import (
    EnergyFittingNet,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)
from deepmd.tf2.model.dp_zbl_model import (
    DPZBLModel,
)


def get_standard_model(data: dict) -> BaseModel:
    """Get a Model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    data = deepcopy(data)
    if "type_embedding" in data:
        raise ValueError(
            "In the tf2 backend, type_embedding is not at the model level, but within the descriptor. See type embedding documentation for details."
        )
    descriptor_type = data["descriptor"].pop("type")
    data["descriptor"]["type_map"] = data["type_map"]
    data["descriptor"]["ntypes"] = len(data["type_map"])
    data["fitting_net"] = data.get("fitting_net", {})
    fitting_type = data["fitting_net"].pop("type", "ener")
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
    data = deepcopy(data)
    data["descriptor"]["ntypes"] = len(data["type_map"])
    data["descriptor"]["type_map"] = data["type_map"]
    descriptor_type = data["descriptor"].pop("type")
    descriptor = BaseDescriptor.get_class_by_type(descriptor_type)(**data["descriptor"])
    fitting_type = data["fitting_net"].pop("type")
    data["fitting_net"]["type_map"] = data["type_map"]
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


def get_sezm_model(data: dict) -> BaseModel:
    """Build a DPA4/SeZM energy model from the pt-style model config."""
    data = deepcopy(data)
    if "spin" in data:
        raise NotImplementedError("Spin DPA4/SeZM models are not supported in TF2.")
    if str(data.get("bridging_method", "none")).lower() != "none":
        raise NotImplementedError("DPA4/SeZM bridging is not supported in TF2.")
    if data.get("lora") is not None:
        raise NotImplementedError("DPA4/SeZM LoRA is not supported in TF2.")
    if data.get("use_compile"):
        raise NotImplementedError("model.use_compile is not supported in TF2.")
    if data.get("preset_out_bias"):
        raise NotImplementedError("DPA4/SeZM preset_out_bias is not supported in TF2.")

    data.pop("type", None)
    data.setdefault("descriptor", {})
    data.setdefault("fitting_net", {})
    data["descriptor"].setdefault("type", "dpa4")
    data["fitting_net"].setdefault("type", "dpa4_ener")
    if data["descriptor"]["type"] not in ("dpa4", "DPA4", "sezm", "SeZM"):
        raise ValueError(
            "Model type 'dpa4' requires a DPA4/SeZM descriptor, but got "
            f"descriptor type '{data['descriptor']['type']}'."
        )
    if data["fitting_net"]["type"] not in ("dpa4_ener", "sezm_ener"):
        raise ValueError(
            "Model type 'dpa4' requires the DPA4/SeZM energy fitting net, but got "
            f"fitting_net type '{data['fitting_net']['type']}'."
        )

    descriptor_exclude_types = [
        list(pair) for pair in (data["descriptor"].get("exclude_types") or [])
    ]
    if "pair_exclude_types" in data:
        pair_exclude_types = [list(pair) for pair in (data["pair_exclude_types"] or [])]
        if descriptor_exclude_types and descriptor_exclude_types != pair_exclude_types:
            raise ValueError(
                "DPA4/SeZM pair_exclude_types and descriptor.exclude_types must "
                "match when both are provided."
            )
    else:
        pair_exclude_types = descriptor_exclude_types
    data["pair_exclude_types"] = pair_exclude_types
    data["descriptor"]["exclude_types"] = deepcopy(pair_exclude_types)
    return get_standard_model(data)


def get_model(data: dict) -> BaseModel:
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
    elif model_type in ("SeZM", "sezm", "DPA4", "dpa4"):
        return get_sezm_model(data)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(data)
