# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared model-factory dispatch for dpmodel-driven backends."""

import copy
from collections.abc import (
    Callable,
    Mapping,
)
from typing import (
    Any,
)

from deepmd.utils.spin import (
    Spin,
)

ModelBuilder = Callable[[dict], Any]


def get_model_components(
    data: dict,
    *,
    descriptor_base: type,
    fitting_base: type,
    backend_name: str,
) -> tuple[Any, Any, str]:
    """Construct a backend descriptor and fitting net from model config.

    The backend registries expose the same descriptor/fitting constructor
    contract. Keeping the parameter injection here prevents subtle differences
    in ``type_map``, ``ntypes``, embedding width, and direct-force handling.
    """
    data = copy.deepcopy(data)
    if "type_embedding" in data:
        raise ValueError(
            f"In the {backend_name} backend, type_embedding is not at the model "
            "level, but within the descriptor. See type embedding documentation "
            "for details."
        )
    type_map = copy.deepcopy(data["type_map"])
    descriptor_data = data["descriptor"]
    descriptor_type = descriptor_data.pop("type")
    descriptor_data["ntypes"] = len(type_map)
    descriptor_data["type_map"] = copy.deepcopy(type_map)
    descriptor = descriptor_base.get_class_by_type(descriptor_type)(**descriptor_data)

    fitting_data = data.get("fitting_net", {})
    fitting_type = fitting_data.pop("type", "ener")
    fitting_data["ntypes"] = descriptor.get_ntypes()
    fitting_data["type_map"] = copy.deepcopy(type_map)
    fitting_data["mixed_types"] = descriptor.mixed_types()
    if fitting_type in {"dipole", "polar"}:
        fitting_data["embedding_width"] = descriptor.get_dim_emb()
    fitting_data["dim_descrpt"] = descriptor.get_dim_out()
    if "direct" in fitting_type:
        fitting_data["out_dim"] = descriptor.get_dim_emb()
        if "ener" in fitting_type:
            fitting_data["return_energy"] = True
    fitting = fitting_base.get_class_by_type(fitting_type)(**fitting_data)
    return descriptor, fitting, fitting_type


def get_standard_model(
    data: dict,
    *,
    descriptor_base: type,
    fitting_base: type,
    model_base: type,
    backend_name: str,
) -> Any:
    """Construct a standard model through backend registries."""
    descriptor, fitting, fitting_type = get_model_components(
        data,
        descriptor_base=descriptor_base,
        fitting_base=fitting_base,
        backend_name=backend_name,
    )
    model_type = "ener" if fitting_type == "direct_force_ener" else fitting_type
    model_cls = model_base.get_class_by_type(model_type)
    return model_cls(
        descriptor=descriptor,
        fitting=fitting,
        type_map=data["type_map"],
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=data.get("pair_exclude_types", []),
    )


def get_zbl_model(
    data: dict,
    *,
    descriptor_base: type,
    fitting_base: type,
    atomic_model: type,
    pairtab_model: type,
    zbl_model: type,
    backend_name: str,
) -> Any:
    """Construct a ZBL model from backend-native atomic model classes."""
    data = copy.deepcopy(data)
    descriptor, fitting, fitting_type = get_model_components(
        data,
        descriptor_base=descriptor_base,
        fitting_base=fitting_base,
        backend_name=backend_name,
    )
    if fitting_type != "ener":
        raise ValueError(f"Unknown fitting type {fitting_type}")
    dp_model = atomic_model(descriptor, fitting, type_map=data["type_map"])
    pairtab = pairtab_model(
        data["use_srtab"],
        descriptor.get_rcut(),
        descriptor.get_sel(),
        type_map=data["type_map"],
    )
    return zbl_model(
        dp_model,
        pairtab,
        data["sw_rmin"],
        data["sw_rmax"],
        type_map=data["type_map"],
        smin_alpha=data.get("smin_alpha", 0.1),
        atom_exclude_types=data.get("atom_exclude_types", []),
        pair_exclude_types=data.get("pair_exclude_types", []),
    )


def get_spin_model(
    data: dict,
    *,
    standard_model_factory: ModelBuilder,
    spin_model: type,
) -> Any:
    """Construct a legacy spin model using a backend standard-model factory."""
    data = copy.deepcopy(data)
    data["type_map"] += [item + "_spin" for item in data["type_map"]]
    spin = Spin(
        use_spin=data["spin"]["use_spin"],
        virtual_scale=data["spin"]["virtual_scale"],
    )
    pair_exclude_types = spin.get_pair_exclude_types(
        exclude_types=data.get("pair_exclude_types")
    )
    data["pair_exclude_types"] = pair_exclude_types
    data["descriptor"]["exclude_types"] = pair_exclude_types
    data["atom_exclude_types"] = spin.get_atom_exclude_types(
        exclude_types=data.get("atom_exclude_types")
    )
    data["descriptor"].setdefault("env_protection", 1e-6)
    if data["descriptor"]["type"] == "se_e2_a":
        data["descriptor"]["sel"] += data["descriptor"]["sel"]
    backbone_model = standard_model_factory(data)
    return spin_model(backbone_model=backbone_model, spin=spin)


def get_model(
    data: dict,
    *,
    base_model: type,
    standard_model_factory: ModelBuilder,
    spin_model_factory: ModelBuilder | None = None,
    zbl_model_factory: ModelBuilder | None = None,
    model_factories: Mapping[str, ModelBuilder] | None = None,
) -> Any:
    """Construct a backend model using the shared model-type routing rules.

    Backend modules supply the concrete constructors while this function owns
    the routing precedence. In particular, legacy ``standard`` configurations
    select spin before ZBL, matching the established dpmodel and PyTorch input
    contract. Explicit model types may be handled by backend-specific factories
    before falling back to the backend model plugin registry.

    Parameters
    ----------
    data : dict
        Model configuration.
    base_model : type
        Backend model base class providing ``get_class_by_type``.
    standard_model_factory : callable
        Constructor for an ordinary standard model.
    spin_model_factory : callable, optional
        Constructor for a legacy standard model containing ``spin``.
    zbl_model_factory : callable, optional
        Constructor for a legacy standard model containing ``use_srtab``.
    model_factories : mapping, optional
        Backend-specific constructors keyed by explicit model type.

    Returns
    -------
    Any
        The backend-native model instance.
    """
    model_type = data.get("type", "standard")
    if model_type == "standard":
        if "spin" in data:
            if spin_model_factory is None:
                raise NotImplementedError("Spin model is not implemented yet.")
            return spin_model_factory(data)
        if "use_srtab" in data:
            if zbl_model_factory is None:
                raise NotImplementedError("ZBL model is not implemented yet.")
            return zbl_model_factory(data)
        return standard_model_factory(data)

    if model_factories is not None and model_type in model_factories:
        return model_factories[model_type](data)
    return base_model.get_class_by_type(model_type).get_model(data)


class BackendModelFactory:
    """Bind backend registries once and expose the shared factory operations."""

    def __init__(
        self,
        *,
        descriptor_base: type,
        fitting_base: type,
        model_base: type,
        backend_name: str,
        atomic_model: type | None = None,
        pairtab_model: type | None = None,
        zbl_model: type | None = None,
    ) -> None:
        """Store backend-native classes used by all model construction paths."""
        self.descriptor_base = descriptor_base
        self.fitting_base = fitting_base
        self.model_base = model_base
        self.backend_name = backend_name
        self.atomic_model = atomic_model
        self.pairtab_model = pairtab_model
        self.zbl_model = zbl_model

    def get_model_components(self, data: dict) -> tuple[Any, Any, str]:
        """Construct descriptor and fitting objects for this backend."""
        return get_model_components(
            data,
            descriptor_base=self.descriptor_base,
            fitting_base=self.fitting_base,
            backend_name=self.backend_name,
        )

    def get_standard_model(self, data: dict) -> Any:
        """Construct a standard model for this backend."""
        return get_standard_model(
            data,
            descriptor_base=self.descriptor_base,
            fitting_base=self.fitting_base,
            model_base=self.model_base,
            backend_name=self.backend_name,
        )

    def get_zbl_model(self, data: dict) -> Any:
        """Construct a ZBL model for this backend."""
        if (
            self.atomic_model is None
            or self.pairtab_model is None
            or self.zbl_model is None
        ):
            raise NotImplementedError("ZBL model is not implemented yet.")
        return get_zbl_model(
            data,
            descriptor_base=self.descriptor_base,
            fitting_base=self.fitting_base,
            atomic_model=self.atomic_model,
            pairtab_model=self.pairtab_model,
            zbl_model=self.zbl_model,
            backend_name=self.backend_name,
        )

    def get_model(
        self,
        data: dict,
        *,
        spin_model_factory: ModelBuilder | None = None,
        model_factories: Mapping[str, ModelBuilder] | None = None,
    ) -> Any:
        """Construct a model using this backend and the shared routing rules."""
        return get_model(
            data,
            base_model=self.model_base,
            standard_model_factory=self.get_standard_model,
            spin_model_factory=spin_model_factory,
            zbl_model_factory=self.get_zbl_model,
            model_factories=model_factories,
        )
