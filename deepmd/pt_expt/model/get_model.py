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
from deepmd.pt_expt.model.dpa4_model import (
    DPA4EnergyModel,
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
from deepmd.utils.bridging import (
    expand_bridging_method,
)
from deepmd.utils.spin import (
    Spin,
    normalize_spin_use_spin,
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
get_zbl_model = _model_factory.get_zbl_model


def get_sezm_model(data: dict) -> BaseModel:
    """Build a pt_expt energy model from a DPA4/SeZM model config.

    Mirrors :func:`deepmd.pt.model.model.get_sezm_model` so that dpa4/sezm
    training configs are interchangeable between the pt and pt_expt backends.
    In addition to the ``SeZM``/``sezm``/``dpa4`` aliases accepted by pt,
    pt_expt also accepts ``DPA4``.
    Supported SeZM extension: native-scheme spin, routed to
    :func:`get_native_spin_model`. Analytical bridging is a ``linear_ener``
    composition and routes through :func:`get_linear_model` instead (the
    ``bridging_method`` sugar expands to that form in :func:`get_model`);
    this builder rejects the flag.

    Still unsupported here, each raising ``NotImplementedError``: the
    virtual-atom (``deepspin``) spin scheme, ``lora``, ``use_compile``, and
    ``preset_out_bias``.

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
        if str(data["spin"].get("scheme", "deepspin")) != "native":
            raise NotImplementedError(
                "Spin DPA4/SeZM models with the virtual-atom (deepspin) "
                "scheme are not supported in the pt_expt backend; use spin "
                "scheme 'native' instead."
            )
        return get_native_spin_model(data)
    bridging_method = str(data.get("bridging_method", "none"))
    if bridging_method.lower() not in ("none", ""):
        raise ValueError(
            "`bridging_method` is not supported by the DPA4/SeZM builder: "
            "analytical bridging builds a linear composition. Route the "
            "config through `get_model` (which expands the flag), or spell "
            'the composition explicitly with `type: "linear_ener"` and an '
            "`inner_potential` sub-model."
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
    return DPA4EnergyModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=pair_exclude_types,
    )


def get_standard_model(data: dict) -> BaseModel:
    """Build a pt_expt standard model: one descriptor plus one fitting net.

    ``bridging_method`` is rejected here rather than honored. Analytical
    bridging is a COMPOSITION -- it yields a ``LinearEnergyModel`` over
    ``[learned, InnerPotential]`` -- so a builder that accepted it would
    return a model of a different kind than the one requested. The ONE
    owner of the flag is the shared
    :func:`deepmd.utils.bridging.expand_bridging_method` normalizer,
    applied in :func:`get_model`; the composition itself is built by
    :func:`get_linear_model`.

    Rejecting is deliberate over silently ignoring: dropping a bridging
    term without a word yields a physically different model than the config
    asks for.

    Parameters
    ----------
    data : dict
        The data to construct the model.

    Returns
    -------
    BaseModel
        The constructed standard model.

    Raises
    ------
    ValueError
        If ``bridging_method`` is set: bridging is not expressible on a
        non-composite model type.
    """
    bridging_method = str(data.get("bridging_method", "none"))
    if bridging_method.lower() not in ("none", ""):
        raise ValueError(
            "`bridging_method` is not supported for a standard model in the "
            "pt_expt backend: analytical bridging builds a linear "
            "composition, not a standard model. Route the config through "
            "`get_model` (which expands the flag), or spell the composition "
            'explicitly with `type: "linear_ener"` and an `inner_potential` '
            "sub-model."
        )
    return _model_factory.get_standard_model(data)


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
    return NativeSpinEnergyModel(atomic_model_=backbone_model.atomic_model, spin=spin)


def _dpa4_family_child_builder(sub: dict) -> "BaseModel | None":
    """Route DPA4/SeZM linear children through the family builder.

    A ``linear_ener`` child of the DPA4/SeZM model type must get exactly
    the semantics of a standalone ``type: "dpa4"`` model -- the
    descriptor/fitting type defaults, the exclusion consistency check, and
    the loud rejections of unsupported options (``lora``, ``use_compile``,
    ``preset_out_bias``) -- instead of the generic component build that
    would silently ignore them. Returns ``None`` for non-DPA4-family
    children so the shared builder uses its generic path.

    Parameters
    ----------
    sub : dict
        The sub-model config (``type_map`` and any derived clamp radii
        already injected by the shared linear builder).
    """
    family_types = ("dpa4", "sezm")
    model_type = str(sub.get("type", "standard")).lower()
    descriptor = sub.get("descriptor")
    descriptor_type = (
        str(descriptor.get("type", "")).lower() if isinstance(descriptor, dict) else ""
    )
    if model_type not in family_types and descriptor_type not in family_types:
        return None
    return get_sezm_model(sub).atomic_model


def get_linear_model(model_params: dict) -> BaseModel:
    """Get a linear energy model from a ``linear_ener`` config dictionary.

    Children with a ``descriptor`` build as learned atomic models;
    ``pairtab`` children build as pair-tabulation atomic models; an
    ``inner_potential`` child builds the analytical bridging term, with
    the learned sibling descriptor's ``inner_clamp_r_inner``/``_outer``
    derived from the child's ``r_inner``/``r_outer`` (issue #5948). A
    top-level ``spin`` section (scheme ``native``) wraps the composition
    as a :class:`NativeSpinEnergyModel`.

    Parameters
    ----------
    model_params : dict
        The model parameters.
    """
    from .dp_linear_model import (
        LinearEnergyModel,
    )

    model_params = copy.deepcopy(model_params)
    spin = None
    if "spin" in model_params:
        spin_cfg = model_params.pop("spin")
        if str(spin_cfg.get("scheme", "deepspin")) != "native":
            raise NotImplementedError(
                "Spin linear_ener models support only spin scheme 'native' "
                "in the pt_expt backend."
            )
        use_spin = normalize_spin_use_spin(
            spin_cfg["use_spin"], model_params["type_map"]
        )
        spin = Spin(
            use_spin=use_spin,
            virtual_scale=spin_cfg.get("virtual_scale", 1.0),
            allow_missing_label=spin_cfg.get("allow_missing_label", False),
        )
        for sub in model_params["models"]:
            if "descriptor" in sub:
                sub["descriptor"]["use_spin"] = use_spin
    composed = _model_factory.get_linear_atomic_model(
        model_params,
        descriptor_child_builder=_dpa4_family_child_builder,
    )
    if spin is not None:
        if not composed.supports_native_spin():
            raise NotImplementedError(
                "spin scheme 'native' requires an atomic model declaring "
                "supports_native_spin()"
            )
        return NativeSpinEnergyModel(atomic_model_=composed, spin=spin)
    return LinearEnergyModel(atomic_model_=composed)


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
    data = expand_bridging_method(data)
    if data.get("lora") is not None:
        # The expansion keeps trainer-owned `lora` at the composition top
        # level (the pt trainer reads it there); pt_expt has no LoRA
        # support, so reject it here instead of silently training a plain
        # full model.
        raise NotImplementedError(
            "`lora` is not supported for DPA4/SeZM in the pt_expt backend."
        )
    return _model_factory.get_model(
        data,
        standard_model_factory=get_standard_model,
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
