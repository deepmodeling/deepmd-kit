# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Any,
)

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
from deepmd.dpmodel.fitting.ener_fitting import (
    EnergyFittingNet,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dipole_model import (
    DipoleModel,
)
from deepmd.dpmodel.model.dos_model import (
    DOSModel,
)
from deepmd.dpmodel.model.dp_zbl_model import (
    DPZBLModel,
)
from deepmd.dpmodel.model.dpa4_model import (
    DPA4EnergyModel,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.model.native_spin_model import (
    NativeSpinEnergyModel,
)
from deepmd.dpmodel.model.polar_model import (
    PolarModel,
)
from deepmd.dpmodel.model.property_model import (
    PropertyModel,
)
from deepmd.dpmodel.model.spin_model import (
    SpinModel,
)
from deepmd.utils.spin import (
    Spin,
    normalize_spin_use_spin,
)

_DPA4_SEZM_DESCRIPTOR_TYPES = ("dpa4", "DPA4", "sezm", "SeZM")


def _get_standard_model_components(
    data: dict[str, Any], ntypes: int
) -> tuple[BaseDescriptor, BaseFitting, str]:
    # descriptor
    data["descriptor"]["ntypes"] = ntypes
    data["descriptor"]["type_map"] = copy.deepcopy(data["type_map"])
    descriptor = BaseDescriptor(**data["descriptor"])
    # fitting
    fitting_net = data.get("fitting_net", {})
    fitting_net["type"] = fitting_net.get("type", "ener")
    fitting_net["ntypes"] = descriptor.get_ntypes()
    fitting_net["type_map"] = copy.deepcopy(data["type_map"])
    fitting_net["mixed_types"] = descriptor.mixed_types()
    if fitting_net["type"] in ["dipole", "polar"]:
        fitting_net["embedding_width"] = descriptor.get_dim_emb()
    fitting_net["dim_descrpt"] = descriptor.get_dim_out()
    grad_force = "direct" not in fitting_net["type"]
    if not grad_force:
        fitting_net["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_net["type"]:
            fitting_net["return_energy"] = True
    fitting = BaseFitting(**fitting_net)
    return descriptor, fitting, fitting_net["type"]


def get_standard_model(data: dict) -> EnergyModel:
    """Get a EnergyModel from a dictionary.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    if "type_embedding" in data:
        raise ValueError(
            "In the DP backend, type_embedding is not at the model level, but within the descriptor. See type embedding documentation for details."
        )
    data = copy.deepcopy(data)
    ntypes = len(data["type_map"])
    # Analytical bridging (e.g. ZBL): the radii feed the DESCRIPTOR's
    # InnerClamp/BridgingSwitch (mirrors pt's builder); the method builds the
    # atomic model's InterPotential below.
    bridging_method = str(data.get("bridging_method", "none"))
    bridging_enabled = bridging_method.lower() not in ("none", "")
    if bridging_enabled:
        data["descriptor"]["inner_clamp_r_inner"] = data.get("bridging_r_inner", 0.5)
        data["descriptor"]["inner_clamp_r_outer"] = data.get("bridging_r_outer", 0.8)
    descriptor, fitting, fitting_net_type = _get_standard_model_components(data, ntypes)
    atom_exclude_types = data.get("atom_exclude_types", [])
    pair_exclude_types = data.get("pair_exclude_types", [])

    if fitting_net_type == "dipole":
        modelcls = DipoleModel
    elif fitting_net_type == "polar":
        modelcls = PolarModel
    elif fitting_net_type == "dos":
        modelcls = DOSModel
    elif fitting_net_type in ["ener", "direct_force_ener"]:
        modelcls = EnergyModel
    elif fitting_net_type in ["dpa4_ener", "sezm_ener"]:
        modelcls = DPA4EnergyModel
    elif fitting_net_type == "property":
        modelcls = PropertyModel
    else:
        raise RuntimeError(f"Unknown fitting type: {fitting_net_type}")

    model = modelcls(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=atom_exclude_types,
        pair_exclude_types=pair_exclude_types,
    )
    if bridging_enabled:
        # Composition, not a flag (first-principles design): the analytical
        # bridging term is its own atomic model, summed with the learned one
        # by the existing linear composition machinery.
        from deepmd.dpmodel.atomic_model.inter_potential import (
            InterPotentialAtomicModel,
        )
        from deepmd.dpmodel.atomic_model.linear_atomic_model import (
            LinearEnergyAtomicModel,
        )
        from deepmd.dpmodel.model.dp_linear_model import (
            LinearEnergyModel,
        )

        zbl_atomic = InterPotentialAtomicModel(
            type_map=data["type_map"],
            mode=bridging_method,
            rcut=descriptor.get_rcut(),
            sel=descriptor.get_sel(),
        )
        composed = LinearEnergyAtomicModel(
            models=[model.atomic_model, zbl_atomic],
            type_map=data["type_map"],
            weights="sum",
        )
        return LinearEnergyModel(atomic_model_=composed)
    return model


def get_zbl_model(data: dict) -> DPZBLModel:
    data = copy.deepcopy(data)
    data["descriptor"]["ntypes"] = len(data["type_map"])
    data["descriptor"]["type_map"] = data["type_map"]
    descriptor = BaseDescriptor(**data["descriptor"])
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
        descriptor.get_rcut(),
        descriptor.get_sel(),
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
    data = copy.deepcopy(data)
    # include virtual spin and placeholder types
    data["type_map"] += [item + "_spin" for item in data["type_map"]]
    spin = Spin(
        use_spin=data["spin"]["use_spin"],
        virtual_scale=data["spin"]["virtual_scale"],
    )
    pair_exclude_types = spin.get_pair_exclude_types(
        exclude_types=data.get("pair_exclude_types", None)
    )
    data["pair_exclude_types"] = pair_exclude_types
    # for descriptor data stat
    data["descriptor"]["exclude_types"] = pair_exclude_types
    atom_exclude_types = spin.get_atom_exclude_types(
        exclude_types=data.get("atom_exclude_types", None)
    )
    data["atom_exclude_types"] = atom_exclude_types
    if "env_protection" not in data["descriptor"]:
        data["descriptor"]["env_protection"] = 1e-6
    if data["descriptor"]["type"] in ["se_e2_a"]:
        # only expand sel for se_e2_a
        data["descriptor"]["sel"] += data["descriptor"]["sel"]
    backbone_model = get_standard_model(data)
    return SpinModel(backbone_model=backbone_model, spin=spin)


def get_native_spin_model(data: dict) -> NativeSpinEnergyModel:
    """Get a native (virtual-atom-free) spin model from a dictionary.

    Unlike :func:`get_spin_model`, no virtual atoms or doubled type map are
    introduced: ``spin`` is injected into the descriptor config as
    ``use_spin`` and consumed by the descriptor's equivariant spin
    embedding. Any descriptor declaring ``supports_native_spin()`` is
    eligible; the gate is the capability method, not a descriptor-type
    list.

    Parameters
    ----------
    data : dict
        The data to construct the model.
    """
    data = copy.deepcopy(data)
    spin_cfg = data.pop("spin")
    if str(data.get("bridging_method", "none")).lower() not in ("none", ""):
        raise NotImplementedError(
            "analytical bridging combined with the native spin scheme is a "
            "follow-up (the bridged model is a linear composition; the "
            "native-spin factory composes over a single standard model)"
        )
    # Expand index/symbol forms of ``use_spin`` against ``type_map`` into the
    # per-type boolean list (pure; validates symbols).
    use_spin = normalize_spin_use_spin(spin_cfg["use_spin"], data["type_map"])
    spin = Spin(
        use_spin=use_spin,
        virtual_scale=spin_cfg.get("virtual_scale", 1.0),
        allow_missing_label=spin_cfg.get("allow_missing_label", False),
    )
    data["descriptor"]["use_spin"] = use_spin
    ntypes = len(data["type_map"])
    try:
        descriptor, fitting, _ = _get_standard_model_components(data, ntypes)
    except TypeError as err:
        # A descriptor without native spin support rejects the injected
        # ``use_spin`` keyword at construction; translate to the
        # capability-gate error.
        raise NotImplementedError(
            "spin scheme 'native' requires a descriptor with native spin "
            "support (supports_native_spin()); descriptor type "
            f"{data['descriptor'].get('type')!r} does not accept `use_spin`"
        ) from err
    if not descriptor.supports_native_spin():
        raise NotImplementedError(
            "spin scheme 'native' requires a descriptor declaring "
            "supports_native_spin()"
        )
    return NativeSpinEnergyModel(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=data.get("pair_exclude_types", []),
        spin=spin,
    )


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
            if data["spin"].get("scheme", "deepspin") == "native":
                return get_native_spin_model(data)
            return get_spin_model(data)
        elif "use_srtab" in data:
            return get_zbl_model(data)
        else:
            return get_standard_model(data)
    else:
        return BaseModel.get_class_by_type(model_type).get_model(data)
