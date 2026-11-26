# SPDX-License-Identifier: LGPL-3.0-or-later
import copy

from deepmd.dpmodel.fitting.dpa4_ener import (
    _resolve_auto_neuron,
)
from deepmd.dpmodel.model.model_factory import (
    BackendModelFactory,
)
from deepmd.jax.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.jax.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
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
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_linear_model import (
    LinearEnergyModel,
)
from deepmd.jax.model.dp_zbl_model import (
    DPZBLModel,
)
from deepmd.jax.model.native_spin_model import (
    NativeSpinEnergyModel,
)
from deepmd.utils.bridging import (
    expand_bridging_method,
)
from deepmd.utils.spin import (
    Spin,
    normalize_spin_use_spin,
)

_model_factory = BackendModelFactory(
    descriptor_base=BaseDescriptor,
    fitting_base=BaseFitting,
    model_base=BaseModel,
    backend_name="JAX",
    atomic_model=DPAtomicModel,
    pairtab_model=PairTabAtomicModel,
    zbl_model=DPZBLModel,
    linear_atomic_model=LinearEnergyAtomicModel,
)
get_zbl_model = _model_factory.get_zbl_model


def get_standard_model(data: dict) -> BaseModel:
    """Construct a learned model whose analytical terms are already expanded."""
    if str(data.get("bridging_method", "none")).lower() not in ("none", ""):
        raise ValueError(
            "Analytical bridging requires get_model() or an explicit linear_ener "
            "composition; it is not a standard-model option."
        )
    return _model_factory.get_standard_model(data)


def get_sezm_model(data: dict) -> BaseModel:
    """Construct the DPA4 descriptor and gated energy fitting contract."""
    data = copy.deepcopy(data)
    if "spin" in data:
        return get_native_spin_model(data)
    data.setdefault("descriptor", {}).setdefault("type", "dpa4")
    data.setdefault("fitting_net", {}).setdefault("type", "dpa4_ener")
    if str(data["descriptor"]["type"]).lower() not in ("dpa4", "sezm"):
        raise ValueError("A DPA4 model requires a DPA4/SeZM descriptor.")
    if data["fitting_net"]["type"] not in ("dpa4_ener", "sezm_ener", "property"):
        raise ValueError(
            "A DPA4 model requires an energy or invariant-property fitting net."
        )
    descriptor_pairs = [
        list(pair) for pair in (data["descriptor"].get("exclude_types") or [])
    ]
    pairs = [
        list(pair) for pair in (data.get("pair_exclude_types", descriptor_pairs) or [])
    ]
    if descriptor_pairs and pairs != descriptor_pairs:
        raise ValueError(
            "Model pair_exclude_types and descriptor exclude_types must match."
        )
    data["pair_exclude_types"] = pairs
    data["descriptor"]["exclude_types"] = pairs
    if data["fitting_net"]["type"] == "property":
        return _get_dpa4_property_model(data)
    return get_standard_model(data)


def _get_dpa4_property_model(data: dict) -> BaseModel:
    """Apply the DPA4 width convention to the generic invariant-property model."""
    descriptor_options = dict(data["descriptor"])
    descriptor_type = descriptor_options.pop("type")
    descriptor_options.update(ntypes=len(data["type_map"]), type_map=data["type_map"])
    descriptor = BaseDescriptor.get_class_by_type(descriptor_type)(**descriptor_options)
    fitting_options = dict(data["fitting_net"])
    fitting_options.pop("type")
    fitting_options["neuron"] = _resolve_auto_neuron(
        fitting_options.get("neuron"),
        dim_descrpt=descriptor.get_dim_out(),
        numb_fparam=fitting_options.get("numb_fparam", 0),
        numb_aparam=fitting_options.get("numb_aparam", 0),
        dim_case_embd=fitting_options.get("dim_case_embd", 0),
        case_film_embd=False,
        use_aparam_as_mask=False,
    )
    fitting_options.update(
        ntypes=descriptor.get_ntypes(),
        type_map=data["type_map"],
        mixed_types=descriptor.mixed_types(),
        dim_descrpt=descriptor.get_dim_out(),
    )
    fitting = BaseFitting.get_class_by_type("property")(**fitting_options)
    return BaseModel.get_class_by_type("property")(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=data.get("pair_exclude_types", []),
        preset_out_bias=data.get("preset_out_bias"),
    )


def _spin_config(data: dict) -> Spin:
    """Resolve the native-spin configuration against the model type map."""
    spin_cfg = data.pop("spin")
    if spin_cfg.get("scheme", "deepspin") != "native":
        raise NotImplementedError("JAX DPA4 supports spin scheme 'native'.")
    return Spin(
        use_spin=normalize_spin_use_spin(spin_cfg["use_spin"], data["type_map"]),
        allow_missing_label=spin_cfg.get("allow_missing_label", False),
    )


def get_native_spin_model(data: dict) -> NativeSpinEnergyModel:
    """Build a native-spin model without virtual atoms or type expansion."""
    data = copy.deepcopy(data)
    if data.get("fitting_net", {}).get("type") == "property":
        raise ValueError("Native-spin DPA4 models require an energy fitting net.")
    spin = _spin_config(data)
    data.setdefault("descriptor", {})["use_spin"] = spin.use_spin.tolist()
    builder = (
        get_sezm_model
        if str(data.get("type")).lower() in ("dpa4", "sezm")
        else get_standard_model
    )
    backbone = builder(data)
    if not backbone.atomic_model.supports_native_spin():
        raise ValueError("The atomic model does not support native spin.")
    return NativeSpinEnergyModel(atomic_model_=backbone.atomic_model, spin=spin)


def _dpa4_child(sub: dict) -> DPAtomicModel | None:
    if sub.get("fitting_net", {}).get("type") == "property":
        raise ValueError("Linear energy compositions require energy fitting nets.")
    model_type = str(sub.get("type", "standard")).lower()
    descriptor_type = str(sub.get("descriptor", {}).get("type", "")).lower()
    if model_type in ("dpa4", "sezm") or descriptor_type in ("dpa4", "sezm"):
        return get_sezm_model(sub).atomic_model
    return None


def get_linear_model(data: dict) -> BaseModel:
    """Compose learned energies and analytical short-range pair potentials."""
    data = copy.deepcopy(data)
    # Adapter injection belongs to get_model after the composition is built.
    data.pop("lora", None)
    spin = _spin_config(data) if "spin" in data else None
    if spin is not None:
        for sub in data["models"]:
            if "descriptor" in sub:
                sub["descriptor"]["use_spin"] = spin.use_spin.tolist()
    atomic_model = _model_factory.get_linear_atomic_model(
        data, descriptor_child_builder=_dpa4_child
    )
    if spin is not None:
        if not atomic_model.supports_native_spin():
            raise ValueError("The atomic model does not support native spin.")
        return NativeSpinEnergyModel(atomic_model_=atomic_model, spin=spin)
    return LinearEnergyModel(atomic_model_=atomic_model)


def get_model(data: dict) -> BaseModel:
    """Construct a JAX model, resolving analytical bridging compositions."""
    if data.get("fitting_net", {}).get("type") == "property" and str(
        data.get("bridging_method", "none")
    ).lower() not in ("none", ""):
        raise ValueError("DPA4 property fitting does not support analytical bridging.")
    data = expand_bridging_method(data)
    lora = data.get("lora")
    model = _model_factory.get_model(
        data,
        native_spin_model_factory=get_native_spin_model,
        model_factories={
            "linear_ener": get_linear_model,
            "dpa4": get_sezm_model,
            "DPA4": get_sezm_model,
            "sezm": get_sezm_model,
            "SeZM": get_sezm_model,
        },
    )
    if lora is not None:
        from deepmd.jax.descriptor.lora import (
            apply_lora_to_sezm,
        )

        apply_lora_to_sezm(model, rank=int(lora["rank"]), alpha=lora.get("alpha"))
    return model
