# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model factory for the pt_expt backend.

Mirrors ``deepmd.dpmodel.model.model`` but uses the pt_expt
``BaseDescriptor`` / ``BaseFitting`` registries so that the
constructed objects are ``torch.nn.Module`` subclasses.
"""

import copy
import logging
import os

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
from deepmd.pt_expt.model.dpa4_model import (
    DPA4EnergyModel,
)
from deepmd.pt_expt.model.ener_model import (
    EnergyModel,
)
from deepmd.pt_expt.model.model import (
    BaseModel,
)
from deepmd.pt_expt.model.native_spin_model import (
    NativeSpinEnergyModel,
)
from deepmd.pt_expt.model.spin_ener_model import (
    SpinEnergyModel,
)
from deepmd.utils.spin import (
    Spin,
    normalize_spin_use_spin,
)

log = logging.getLogger(__name__)

#: ``DP_TF32_INFER`` -> eval-time matmul precision, copied from the pt backend's
#: ``deepmd.pt.model.model.sezm_model._TF32_INFER_PRECISION_CHOICES`` so the two
#: backends read the same environment variable the same way.
_TF32_INFER_PRECISION_CHOICES = {
    "0": "highest",
    "1": "high",
    "2": "medium",
}


def _apply_tf32_policy(model: BaseModel, data: dict) -> BaseModel:
    """Attach the DPA4/SeZM TF32 matmul-precision policy to a built model.

    Mirrors the pt backend: TRAINING forwards follow ``model.enable_tf32``
    (default ``True``) and EVAL forwards follow ``DP_TF32_INFER``.  The model
    layer applies the policy -- see ``call_common`` in
    :func:`deepmd.pt_expt.model.make_model.make_model` and
    ``_CompiledModel.forward`` for the compiled path.

    Parameters
    ----------
    model : BaseModel
        The freshly built model to configure.
    data : dict
        The model config section, read for ``enable_tf32``.

    Returns
    -------
    BaseModel
        The same model, with the precision policy attached.

    Raises
    ------
    ValueError
        If ``DP_TF32_INFER`` is set to anything other than ``0``, ``1``, or
        ``2``.
    """
    model.enable_tf32 = bool(data.get("enable_tf32", True))
    tf32_infer_env = os.environ.get("DP_TF32_INFER", "0").strip().lower()
    if tf32_infer_env not in _TF32_INFER_PRECISION_CHOICES:
        raise ValueError(f"DP_TF32_INFER must be one of 0/1/2, got {tf32_infer_env!r}")
    model.tf32_infer_precision = _TF32_INFER_PRECISION_CHOICES[tf32_infer_env]
    return model


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
    ``enable_tf32`` follows the pt backend: TRAINING forwards run at TF32
    ("high") matmul precision when it is true (the default), while EVAL
    forwards follow ``DP_TF32_INFER``.  See :func:`_apply_tf32_policy`.
    """
    data = copy.deepcopy(data)
    if "spin" in data:
        if str(data["spin"].get("scheme", "deepspin")) != "native":
            raise NotImplementedError(
                "Spin DPA4/SeZM models with the virtual-atom (deepspin) "
                "scheme are not supported in the pt_expt backend; use spin "
                "scheme 'native' instead."
            )
        return get_native_spin_model(data)
    # Analytical bridging (e.g. ZBL): the radii feed the DESCRIPTOR's
    # InnerClamp/BridgingSwitch (mirrors pt's builder); the method builds the
    # atomic model's InnerPotential at construction below.
    bridging_method = str(data.get("bridging_method", "none"))
    bridging_enabled = bridging_method.lower() not in ("none", "")
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
    if bridging_enabled:
        data["descriptor"]["inner_clamp_r_inner"] = data.get("bridging_r_inner", 0.5)
        data["descriptor"]["inner_clamp_r_outer"] = data.get("bridging_r_outer", 0.8)
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
    model = DPA4EnergyModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=pair_exclude_types,
    )
    if bridging_enabled:
        # Composition, not a flag (first-principles design): the analytical
        # bridging term is its own atomic model, summed with the learned one
        # by the existing linear composition machinery.
        from deepmd.dpmodel.atomic_model.inner_potential import (
            InnerPotentialAtomicModel,
        )
        from deepmd.dpmodel.atomic_model.linear_atomic_model import (
            LinearEnergyAtomicModel,
        )
        from deepmd.pt_expt.model.dp_linear_model import (
            LinearEnergyModel,
        )

        zbl_atomic = InnerPotentialAtomicModel(
            type_map=data["type_map"],
            mode=bridging_method,
            rcut=descriptor.get_rcut(),
            sel=descriptor.get_sel(),
        )
        composed = LinearEnergyAtomicModel(
            models=[model.atomic_model, zbl_atomic],
            type_map=data["type_map"],
            weights="sum",
            # Both exclusions belong to the composition: its children share one
            # graph, so "excluded" must cover the analytical term too.
            atom_exclude_types=data.get("atom_exclude_types", []),
            pair_exclude_types=pair_exclude_types,
        )
        return _apply_tf32_policy(LinearEnergyModel(atomic_model_=composed), data)
    return _apply_tf32_policy(model, data)


def get_native_spin_model(data: dict) -> NativeSpinEnergyModel:
    """Build a pt_expt native (virtual-atom-free) spin model.

    Mirrors :func:`deepmd.dpmodel.model.model.get_native_spin_model`: no
    virtual atoms or doubled type map are introduced, and ``use_spin`` is
    injected into the descriptor config (consumed by the descriptor's
    equivariant spin embedding). The non-spin backbone is built by the
    standard builder for the config's model type -- :func:`get_sezm_model`
    for the DPA4/SeZM family (keeping its bridging/lora/compile/
    preset_out_bias rejections and ``exclude_types`` consistency check),
    else :func:`get_standard_model` -- then re-classed through the
    registered :class:`NativeSpinEnergyModel`. Eligibility is the atomic
    model's own ``supports_native_spin()`` capability, not a descriptor-type
    list -- so a bridging composition answers for itself.

    Parameters
    ----------
    data : dict
        The data to construct the model. Must carry a top-level ``"spin"``
        key with ``scheme == "native"``.
    """
    data = copy.deepcopy(data)
    spin_cfg = data.pop("spin")
    data.setdefault("descriptor", {})
    # Expand index/symbol forms of ``use_spin`` against ``type_map`` into the
    # per-type boolean list (pure; validates symbols).
    use_spin = normalize_spin_use_spin(spin_cfg["use_spin"], data["type_map"])
    spin = Spin(
        use_spin=use_spin,
        virtual_scale=spin_cfg.get("virtual_scale", 1.0),
        allow_missing_label=spin_cfg.get("allow_missing_label", False),
    )
    data["descriptor"]["use_spin"] = use_spin
    model_type = str(data.get("type", "standard")).lower()
    backbone_builder = (
        get_sezm_model if model_type in ("dpa4", "sezm") else get_standard_model
    )
    try:
        backbone_model = backbone_builder(data)
    except TypeError as err:
        if "use_spin" not in str(err):
            # Unrelated construction error (e.g. a bogus fitting kwarg):
            # propagate with its real context instead of masking it as a
            # capability failure.
            raise
        # A descriptor without native spin support rejects the injected
        # ``use_spin`` keyword at construction; translate to the
        # capability-gate error.
        raise NotImplementedError(
            "spin scheme 'native' requires a descriptor with native spin "
            "support (supports_native_spin()); descriptor type "
            f"{data['descriptor'].get('type')!r} does not accept `use_spin`"
        ) from err
    # The ATOMIC MODEL answers the capability -- it knows its own structure,
    # so this holds for a plain descriptor+fitting model and for a bridging
    # composition alike, with no assumption here about either.
    if not backbone_model.atomic_model.supports_native_spin():
        raise NotImplementedError(
            "spin scheme 'native' requires an atomic model declaring "
            "supports_native_spin()"
        )
    return _apply_tf32_policy(
        NativeSpinEnergyModel(atomic_model_=backbone_model.atomic_model, spin=spin),
        data,
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
        native_spin_model_factory=get_native_spin_model,
        model_factories={
            "linear_ener": get_linear_model,
            "dpa4": get_sezm_model,
            "DPA4": get_sezm_model,
            "sezm": get_sezm_model,
            "SeZM": get_sezm_model,
        },
    )
