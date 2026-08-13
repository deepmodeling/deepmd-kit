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
    data = copy.deepcopy(data)
    # Analytical bridging (e.g. ZBL): the radii feed the DESCRIPTOR's
    # InnerClamp/BridgingSwitch (mirrors pt's builder); the method builds the
    # atomic model's InnerPotential below.
    bridging_method = str(data.get("bridging_method", "none"))
    bridging_enabled = bridging_method.lower() not in ("none", "")
    if bridging_enabled:
        data["descriptor"]["inner_clamp_r_inner"] = data.get("bridging_r_inner", 0.5)
        data["descriptor"]["inner_clamp_r_outer"] = data.get("bridging_r_outer", 0.8)
    model = _model_factory.get_standard_model(data)
    if not bridging_enabled:
        return model

    descriptor = model.atomic_model.descriptor
    atom_exclude_types = data.get("atom_exclude_types", [])
    pair_exclude_types = data.get("pair_exclude_types", [])
    # Composition, not a flag (first-principles design): the analytical
    # bridging term is its own atomic model, summed with the learned one by the
    # existing linear composition machinery.
    from deepmd.dpmodel.atomic_model.inner_potential import (
        InnerPotentialAtomicModel,
    )
    from deepmd.dpmodel.atomic_model.linear_atomic_model import (
        LinearEnergyAtomicModel,
    )
    from deepmd.dpmodel.model.dp_linear_model import (
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
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )
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

    The non-spin backbone is built by :func:`get_standard_model`, which OWNS
    everything about assembling the atomic model -- descriptor/fitting,
    exclusions and the analytical-bridging composition -- so ``spin`` and
    ``bridging_method`` combine for free: the wrapper re-classes whatever
    atomic model came back, be it a single learned model or a
    ``LinearEnergyAtomicModel`` over ``[learned, InnerPotential]`` (the
    analytical child accepts and ignores ``spin``; the learned child consumes
    it).

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
    return _model_factory.get_model(
        data,
        standard_model_factory=get_standard_model,
        spin_model_factory=get_spin_model,
        native_spin_model_factory=get_native_spin_model,
    )
