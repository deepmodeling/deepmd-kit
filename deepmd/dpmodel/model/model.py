# SPDX-License-Identifier: LGPL-3.0-or-later
import copy

from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dp_zbl_model import (
    DPZBLModel,
)
from deepmd.dpmodel.model.model_factory import (
    BackendModelFactory,
)
from deepmd.dpmodel.model.model_factory import (
    get_spin_model as get_spin_model_from_factory,
)
from deepmd.dpmodel.model.native_spin_model import (
    NativeSpinEnergyModel,
)
from deepmd.dpmodel.model.spin_model import (
    SpinModel,
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
    backend_name="DP",
    atomic_model=DPAtomicModel,
    pairtab_model=PairTabAtomicModel,
    zbl_model=DPZBLModel,
)
get_zbl_model = _model_factory.get_zbl_model

_DPA4_SEZM_DESCRIPTOR_TYPES = ("dpa4", "DPA4", "sezm", "SeZM")


def get_standard_model(data: dict) -> BaseModel:
    """Get a standard model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    bridging_method = str(data.get("bridging_method", "none"))
    if bridging_method.lower() not in ("none", ""):
        raise ValueError(
            "`bridging_method` is not supported for a standard model: "
            "analytical bridging builds a linear composition, not a "
            "standard model. Route the config through `get_model` (which "
            "expands the flag), or spell the composition explicitly with "
            '`type: "linear_ener"` and an `inner_potential` sub-model.'
        )
    return _model_factory.get_standard_model(data)


def get_linear_model(data: dict) -> BaseModel:
    """Build a linear energy model from a ``linear_ener`` config.

    Children with a ``descriptor`` build as standard learned atomic
    models; ``pairtab`` children build as pair-tabulation atomic models;
    an ``inner_potential`` child builds the analytical bridging term. The
    composition is the ONE owner of the bridging coupling: it derives the
    learned sibling descriptor's ``inner_clamp_r_inner``/``_outer`` from
    the ``inner_potential`` child's ``r_inner``/``r_outer``, so the radii
    are written once in the config (issue #5948, task 2).

    A top-level ``spin`` section (scheme ``native``) wraps the composed
    atomic model as a :class:`NativeSpinEnergyModel`, with ``use_spin``
    injected into every learned child's descriptor.

    Parameters
    ----------
    data : dict
        The model configuration.
    """
    from deepmd.dpmodel.model.dp_linear_model import (
        LinearEnergyModel,
    )

    data = copy.deepcopy(data)
    spin = None
    if "spin" in data:
        spin_cfg = data.pop("spin")
        if str(spin_cfg.get("scheme", "deepspin")) != "native":
            raise NotImplementedError(
                "Spin linear_ener models support only spin scheme 'native'."
            )
        use_spin = normalize_spin_use_spin(spin_cfg["use_spin"], data["type_map"])
        spin = Spin(
            use_spin=use_spin,
            virtual_scale=spin_cfg.get("virtual_scale", 1.0),
            allow_missing_label=spin_cfg.get("allow_missing_label", False),
        )
        for sub in data["models"]:
            if "descriptor" in sub:
                sub["descriptor"]["use_spin"] = use_spin
    composed = _model_factory.get_linear_atomic_model(data)
    if spin is not None:
        if not composed.supports_native_spin():
            raise NotImplementedError(
                "spin scheme 'native' requires an atomic model declaring "
                "supports_native_spin()"
            )
        return NativeSpinEnergyModel(atomic_model_=composed, spin=spin)
    return LinearEnergyModel(atomic_model_=composed)


def get_spin_model(data: dict) -> SpinModel:
    """Get a spin model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    if data["descriptor"]["type"] in _DPA4_SEZM_DESCRIPTOR_TYPES:
        raise NotImplementedError(
            "the virtual-atom (deepspin) scheme is not supported for "
            "DPA4/SeZM; use spin scheme 'native'"
        )
    return get_spin_model_from_factory(
        data,
        standard_model_factory=get_standard_model,
        spin_model=SpinModel,
    )


def get_native_spin_model(data: dict) -> NativeSpinEnergyModel:
    """Get a native (virtual-atom-free) spin model from a dictionary.

    Unlike :func:`get_spin_model`, no virtual atoms or doubled type map are
    introduced: ``spin`` is injected into the descriptor config as
    ``use_spin`` and consumed by the descriptor's equivariant spin
    embedding. Any atomic model declaring ``supports_native_spin()`` is
    eligible; the gate is that capability method, not a descriptor-type
    list.

    The non-spin backbone is built by :func:`get_standard_model`. A spin
    model with analytical bridging is a ``linear_ener`` composition and
    routes through :func:`get_linear_model` instead (the ``bridging_method``
    sugar expands to that form in :func:`get_model`).

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    data = copy.deepcopy(data)
    spin_cfg = data.pop("spin")
    # Expand index/symbol forms of ``use_spin`` against ``type_map`` into the
    # per-type boolean list (pure; validates symbols).
    use_spin = normalize_spin_use_spin(spin_cfg["use_spin"], data["type_map"])
    spin = Spin(
        use_spin=use_spin,
        allow_missing_label=spin_cfg.get("allow_missing_label", False),
    )
    data.setdefault("descriptor", {})
    data["descriptor"]["use_spin"] = use_spin
    try:
        backbone_model = get_standard_model(data)
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


def get_model(data: dict) -> BaseModel:
    """Get a model from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    data = expand_bridging_method(data)
    return _model_factory.get_model(
        data,
        standard_model_factory=get_standard_model,
        spin_model_factory=get_spin_model,
        native_spin_model_factory=get_native_spin_model,
        model_factories={
            "linear_ener": get_linear_model,
        },
    )
