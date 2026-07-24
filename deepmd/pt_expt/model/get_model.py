# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model factory for the pt_expt backend.

Mirrors ``deepmd.dpmodel.model.model`` but uses the pt_expt
``BaseDescriptor`` / ``BaseFitting`` registries so that the
constructed objects are ``torch.nn.Module`` subclasses.
"""

import copy
import logging

from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.dpmodel.model.model_factory import (
    BackendModelFactory,
)
from deepmd.dpmodel.model.model_factory import (
    get_spin_model as get_spin_model_from_factory,
)
from deepmd.pt_expt.descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.fitting import (
    BaseFitting,
)
from deepmd.pt_expt.model.dp_zbl_model import (
    DPZBLModel,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.model.spin_ener_model import (
    SpinEnergyModel,
)

log = logging.getLogger(__name__)

# Warn at most once per process for backend-ignored switches (keyed by name).
_WARNED_ONCE: set[str] = set()


_model_factory = BackendModelFactory(
    descriptor_base=BaseDescriptor,
    fitting_base=BaseFitting,
    model_base=BaseModel,
    backend_name="pt_expt",
    atomic_model=DPAtomicModel,
    pairtab_model=PairTabAtomicModel,
    zbl_model=DPZBLModel,
)
get_standard_model = _model_factory.get_standard_model
get_zbl_model = _model_factory.get_zbl_model


def get_sezm_model(data: dict) -> EnergyModel:
    """Build a pt_expt energy model from a DPA4/SeZM model config.

    Mirrors :func:`deepmd.pt.model.model.get_sezm_model` so that dpa4/sezm
    training configs are interchangeable between the pt and pt_expt backends.
    In addition to the ``SeZM``/``sezm``/``dpa4`` aliases accepted by pt,
    pt_expt also accepts ``DPA4``.
    The pt-only SeZM extensions (bridging, LoRA, compile, spin,
    preset_out_bias) are not supported here and raise
    ``NotImplementedError``.

    Notes
    -----
    ``enable_tf32`` is accepted but ignored: the pt backend uses it to toggle
    TF32 matmul precision, while the pt_expt backend always runs at full
    ("highest") matmul precision, which is numerically conservative.
    """
    data = copy.deepcopy(data)
    if bool(data.get("enable_tf32", True)) and "enable_tf32" not in _WARNED_ONCE:
        log.warning(
            "`enable_tf32` has no effect on the pt_expt backend, which "
            "always runs at full ('highest') matmul precision; ignoring it."
        )
        _WARNED_ONCE.add("enable_tf32")
    if "spin" in data:
        raise NotImplementedError(
            "Spin DPA4/SeZM models are not supported in the pt_expt backend."
        )
    if str(data.get("bridging_method", "none")).lower() != "none":
        raise NotImplementedError(
            "`bridging_method` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    if data.get("lora") is not None:
        raise NotImplementedError(
            "`lora` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    if data.get("use_compile"):
        raise NotImplementedError(
            "`use_compile` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    if data.get("preset_out_bias"):
        raise NotImplementedError(
            "`preset_out_bias` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    data.pop("type", None)
    data.setdefault("descriptor", {})
    data.setdefault("fitting_net", {})
    data["descriptor"].setdefault("type", "dpa4")
    data["fitting_net"].setdefault("type", "dpa4_ener")
    # the DPA4/SeZM model type is a fixed descriptor/fitting contract; reject
    # explicit mismatching component types instead of silently building them
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

    # keep descriptor.exclude_types and model pair_exclude_types consistent
    descriptor_exclude_types = [
        list(pair) for pair in (data["descriptor"].get("exclude_types") or [])
    ]
    if "pair_exclude_types" in data:
        pair_exclude_types = [list(pair) for pair in (data["pair_exclude_types"] or [])]
        if descriptor_exclude_types and descriptor_exclude_types != pair_exclude_types:
            raise ValueError(
                "SeZM `pair_exclude_types` and `descriptor.exclude_types` must match "
                "when both are provided."
            )
    else:
        pair_exclude_types = descriptor_exclude_types
    data["pair_exclude_types"] = pair_exclude_types
    data["descriptor"]["exclude_types"] = copy.deepcopy(pair_exclude_types)

    descriptor, fitting, _ = _model_factory.get_model_components(data)
    return EnergyModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=pair_exclude_types,
    )


def get_linear_model(model_params: dict) -> BaseModel:
    """Get a linear energy model from a config dictionary.

    Parameters
    ----------
    model_params : dict
        The model parameters.
    """
    from .dp_linear_model import (
        LinearEnergyModel,
    )

    model_params = copy.deepcopy(model_params)
    weights = model_params.get("weights", "mean")
    list_of_models = []
    ntypes = len(model_params["type_map"])
    for sub_model_params in model_params["models"]:
        if "type_map" not in sub_model_params:
            sub_model_params["type_map"] = model_params["type_map"]
        if "descriptor" in sub_model_params:
            sub_model_params["descriptor"]["ntypes"] = ntypes
            descriptor, fitting, _ = _model_factory.get_model_components(
                sub_model_params
            )
            list_of_models.append(
                DPAtomicModel(descriptor, fitting, type_map=model_params["type_map"])
            )
        else:
            assert (
                "type" in sub_model_params and sub_model_params["type"] == "pairtab"
            ), "Sub-models in LinearEnergyModel must be a DPModel or a PairTable Model"
            list_of_models.append(
                PairTabAtomicModel(
                    sub_model_params["tab_file"],
                    sub_model_params["rcut"],
                    sub_model_params["sel"],
                    type_map=model_params["type_map"],
                )
            )

    atom_exclude_types = model_params.get("atom_exclude_types", [])
    pair_exclude_types = model_params.get("pair_exclude_types", [])
    return LinearEnergyModel(
        models=list_of_models,
        type_map=model_params["type_map"],
        weights=weights,
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )


def get_spin_model(data: dict) -> SpinEnergyModel:
    """Build a pt_expt spin energy model from a config dictionary.

    Mirrors :func:`deepmd.dpmodel.model.model.get_spin_model`: expands the
    type map and descriptor sel for virtual spin atoms, then wraps the
    backbone EnergyModel as a :class:`SpinEnergyModel`.
    """
    return get_spin_model_from_factory(
        data,
        standard_model_factory=get_standard_model,
        spin_model=SpinEnergyModel,
    )


def get_model(data: dict) -> BaseModel:
    """Get a model from a config dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    return _model_factory.get_model(
        data,
        spin_model_factory=get_spin_model,
        model_factories={
            "linear_ener": get_linear_model,
            "dpa4": get_sezm_model,
            "DPA4": get_sezm_model,
            "sezm": get_sezm_model,
            "SeZM": get_sezm_model,
        },
    )
