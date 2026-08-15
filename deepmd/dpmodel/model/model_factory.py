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

from deepmd.utils.bridging import (
    route_canonical_learned_options,
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


def get_linear_atomic_model(
    data: dict,
    *,
    descriptor_base: type,
    fitting_base: type,
    backend_name: str,
    atomic_model: type,
    pairtab_model: type,
    inner_potential_model: type | None = None,
    linear_atomic_model: type | None = None,
    descriptor_child_builder: "Callable[[dict], Any | None] | None" = None,
) -> Any:
    """Build the ``LinearEnergyAtomicModel`` composition from a config.

    Children with a ``descriptor`` build as learned atomic models through
    the backend registries; ``pairtab`` children build as pair-tabulation
    atomic models; an ``inner_potential`` child builds the analytical
    bridging term. The composition is the ONE owner of the bridging
    coupling: it derives the learned sibling descriptor's
    ``inner_clamp_r_inner``/``_outer`` from the ``inner_potential``
    child's ``r_inner``/``r_outer``, so the radii are written once in the
    config (issue #5948).

    Parameters
    ----------
    data : dict
        The ``linear_ener`` model configuration.
    descriptor_base : type
        Backend descriptor registry base class.
    fitting_base : type
        Backend fitting registry base class.
    backend_name : str
        Backend name used in error messages.
    atomic_model : type
        Backend learned atomic-model class.
    pairtab_model : type
        Backend pair-tabulation atomic-model class.
    inner_potential_model : type, optional
        Backend analytical-bridging atomic-model class. Defaults to the
        dpmodel class; a backend that wraps dpmodel classes must pass its
        own wrapper so the composition is assembled from backend-native
        children rather than converted afterwards.
    linear_atomic_model : type, optional
        Backend linear composition atomic-model class. Defaults to the
        dpmodel class, with the same obligation as
        ``inner_potential_model``: a wrapping backend that leaves this
        unset gets a composition that its model wrapper must convert, and
        conversion keeps only what the portable record carries.
    descriptor_child_builder : callable, optional
        Backend hook for descriptor-bearing children: called with the
        child config (``type_map`` and derived clamp radii already
        injected) and returns the child atomic model, or ``None`` to fall
        back to the generic registry build. Backends use it to route
        family-specific model types (e.g. DPA4/SeZM) through their
        validated builders.

    Raises
    ------
    ValueError
        If more than one ``inner_potential`` child is given, if an
        ``inner_potential`` child has no unique learned sibling, if a
        bridged composition does not use ``weights: "sum"``, if a child
        carries a ``bridging_method`` flag, or if a child is of an
        unsupported kind.
    """
    from deepmd.dpmodel.atomic_model.inner_potential import (
        InnerPotentialAtomicModel as InnerPotentialAtomicModelDP,
    )
    from deepmd.dpmodel.atomic_model.linear_atomic_model import (
        LinearEnergyAtomicModel as LinearEnergyAtomicModelDP,
    )

    InnerPotentialAtomicModel = inner_potential_model or InnerPotentialAtomicModelDP
    LinearEnergyAtomicModel = linear_atomic_model or LinearEnergyAtomicModelDP

    data = copy.deepcopy(data)
    type_map = data["type_map"]
    children = data["models"]
    inner_indices = [
        i for i, sub in enumerate(children) if sub.get("type") == "inner_potential"
    ]
    learned_indices = [
        i
        for i, sub in enumerate(children)
        if "descriptor" in sub and i not in inner_indices
    ]
    for i in inner_indices:
        if "descriptor" in children[i]:
            raise ValueError(
                "An `inner_potential` sub-model must not carry a "
                "`descriptor`: the analytical term has no learned "
                "component."
            )
    # Consume-or-reject: this builder has no consumer for these keys, so
    # accepting them silently would train/evaluate a different model than
    # the config asks for. (The pt backend consumes top-level `lora` in its
    # trainer and `shared_dict` in its own linear builder; this shared
    # builder serves backends without either consumer.)
    if data.get("lora") is not None:
        raise NotImplementedError(
            f"`lora` on a linear_ener composition is not supported in the "
            f"{backend_name} backend."
        )
    if data.get("shared_dict"):
        raise NotImplementedError(
            f"`shared_dict` is not supported for linear_ener in the "
            f"{backend_name} backend."
        )
    for sub in children:
        if str(sub.get("bridging_method", "none")).lower() not in ("none", ""):
            raise ValueError(
                "`bridging_method` is not supported on a linear_ener "
                "sub-model: add an `inner_potential` sub-model to the "
                "composition instead."
            )
        if sub.get("lora") is not None:
            raise NotImplementedError(
                "`lora` on a linear_ener sub-model is not supported in the "
                f"{backend_name} backend."
            )
    if inner_indices:
        if len(inner_indices) > 1:
            raise ValueError(
                "A linear_ener composition supports at most one "
                "`inner_potential` sub-model."
            )
        if len(learned_indices) != 1 or len(children) != 2:
            # A third child (e.g. pairtab) has no common execution route
            # with the graph-only bridged pair; reject at construction
            # like the pt builder does.
            raise ValueError(
                "An `inner_potential` sub-model bridges exactly one learned "
                "sibling: expected a linear_ener composition over "
                "[learned, inner_potential]."
            )
        if str(data.get("weights", "mean")) != "sum":
            raise ValueError(
                'A bridged linear_ener composition requires `weights: "sum"`.'
            )
        learned_descriptor_type = str(
            children[learned_indices[0]]["descriptor"].get("type", "dpa4")
        )
        if learned_descriptor_type not in ("dpa4", "DPA4", "sezm", "SeZM"):
            # same family restriction as the pt builder: the clamp window
            # below only exists on DPA4/SeZM descriptors, so any other
            # family would die on an obscure unknown-kwarg TypeError
            raise NotImplementedError(
                f"The {backend_name} backend implements `inner_potential` "
                "bridging only for the DPA4/SeZM descriptor family, but got "
                f"{learned_descriptor_type!r}."
            )
        # The composition derives the sibling descriptor's clamp window from
        # the inner_potential child: one source of truth for the radii.
        inner_cfg = children[inner_indices[0]]
        learned_descriptor = children[learned_indices[0]]["descriptor"]
        learned_descriptor["inner_clamp_r_inner"] = float(inner_cfg.get("r_inner", 0.5))
        learned_descriptor["inner_clamp_r_outer"] = float(inner_cfg.get("r_outer", 0.8))
        route_canonical_learned_options(data, children[learned_indices[0]])

    built: dict[int, Any] = {}
    for i, sub in enumerate(children):
        if i in inner_indices:
            continue
        if "type_map" not in sub:
            sub["type_map"] = copy.deepcopy(type_map)
        elif inner_indices and i == learned_indices[0] and sub["type_map"] != type_map:
            # The analytical child always uses the composition's type_map,
            # and the graph route rejects a non-identity remap at forward
            # time; fail at construction like the pt builder does.
            raise ValueError(
                "A bridged linear_ener composition requires the learned "
                "child's type_map to match the composition type_map."
            )
        if "descriptor" in sub:
            child = None
            if descriptor_child_builder is not None:
                child = descriptor_child_builder(sub)
            if child is None:
                descriptor, fitting, _ = get_model_components(
                    sub,
                    descriptor_base=descriptor_base,
                    fitting_base=fitting_base,
                    backend_name=backend_name,
                )
                child = atomic_model(descriptor, fitting, type_map=sub["type_map"])
            built[i] = child
        else:
            if sub.get("type") != "pairtab":
                raise ValueError(
                    "Sub-models in LinearEnergyModel must be a standard model, "
                    "a pairtab model, or an inner_potential model, but got "
                    f"type {sub.get('type')!r}."
                )
            built[i] = pairtab_model(
                sub["tab_file"],
                sub["rcut"],
                sub["sel"],
                type_map=copy.deepcopy(type_map),
            )
    for i in inner_indices:
        learned_descriptor_obj = built[learned_indices[0]].descriptor
        built[i] = InnerPotentialAtomicModel(
            type_map=copy.deepcopy(type_map),
            mode=children[i].get("mode", "zbl"),
            rcut=learned_descriptor_obj.get_rcut(),
            sel=learned_descriptor_obj.get_sel(),
        )
    return LinearEnergyAtomicModel(
        models=[built[i] for i in range(len(children))],
        type_map=type_map,
        weights=data.get("weights", "mean"),
        # Both exclusions belong to the composition: its children share one
        # graph, so "excluded" must cover the analytical term too.
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
    native_spin_model_factory: ModelBuilder | None = None,
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
    native_spin_model_factory : callable, optional
        Constructor for a standard model using the native spin scheme.
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
            if str(data["spin"].get("scheme", "deepspin")) == "native":
                if native_spin_model_factory is None:
                    raise NotImplementedError(
                        "Native spin model is not implemented yet."
                    )
                return native_spin_model_factory(data)
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
        inner_potential_model: type | None = None,
        linear_atomic_model: type | None = None,
    ) -> None:
        """Store backend-native classes used by all model construction paths."""
        self.descriptor_base = descriptor_base
        self.fitting_base = fitting_base
        self.model_base = model_base
        self.backend_name = backend_name
        self.atomic_model = atomic_model
        self.pairtab_model = pairtab_model
        self.zbl_model = zbl_model
        self.inner_potential_model = inner_potential_model
        self.linear_atomic_model = linear_atomic_model

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

    def get_linear_atomic_model(
        self,
        data: dict,
        *,
        descriptor_child_builder: "Callable[[dict], Any | None] | None" = None,
    ) -> Any:
        """Construct the linear atomic-model composition for this backend."""
        if self.atomic_model is None or self.pairtab_model is None:
            raise NotImplementedError("Linear model is not implemented yet.")
        return get_linear_atomic_model(
            data,
            descriptor_base=self.descriptor_base,
            fitting_base=self.fitting_base,
            backend_name=self.backend_name,
            atomic_model=self.atomic_model,
            pairtab_model=self.pairtab_model,
            inner_potential_model=self.inner_potential_model,
            linear_atomic_model=self.linear_atomic_model,
            descriptor_child_builder=descriptor_child_builder,
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
        standard_model_factory: ModelBuilder | None = None,
        spin_model_factory: ModelBuilder | None = None,
        native_spin_model_factory: ModelBuilder | None = None,
        model_factories: Mapping[str, ModelBuilder] | None = None,
    ) -> Any:
        """Construct a model using this backend and the shared routing rules.

        Backends may override the standard builder when they need a thin
        backend-specific wrapper, such as analytical bridging composition.
        """
        return get_model(
            data,
            base_model=self.model_base,
            standard_model_factory=(
                self.get_standard_model
                if standard_model_factory is None
                else standard_model_factory
            ),
            spin_model_factory=spin_model_factory,
            native_spin_model_factory=native_spin_model_factory,
            zbl_model_factory=self.get_zbl_model,
            model_factories=model_factories,
        )
