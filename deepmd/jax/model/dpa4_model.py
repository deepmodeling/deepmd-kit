# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 energy models and conversion of PyTorch SeZM model records."""

from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model.inner_potential import (
    InnerPotentialAtomicModel,
)
from deepmd.jax.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.dp_linear_model import (
    LinearEnergyModel,
)
from deepmd.jax.model.ener_model import (
    EnergyModel,
)
from deepmd.jax.model.native_spin_model import (
    NativeSpinEnergyModel,
)
from deepmd.jax.model.property_model import (
    PropertyModel,
)
from deepmd.utils.bridging import (
    DEFAULT_TRUNCATION,
)
from deepmd.utils.dpa4_serialization import (
    normalize_sezm_atomic,
)
from deepmd.utils.spin import (
    Spin,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseModel.register("dpa4_ener")
@BaseModel.register("sezm_ener")
@BaseModel.register("dpa4")
@BaseModel.register("DPA4")
@BaseModel.register("sezm")
@BaseModel.register("SeZM")
@BaseModel.register("sezm_atomic")
@BaseModel.register("sezm_native_spin")
class DPA4EnergyModel(EnergyModel):
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EnergyModel:
        """Restore native JAX records or the PyTorch SeZM wrapper schema."""
        model_type = str(data.get("type", "standard")).lower()
        if model_type in ("sezm", "dpa4", "sezm_native_spin"):
            check_version_compatibility(int(data.get("@version", 1)), 2, 1)
            model = cls.deserialize(data["atomic_model"])
            mode = str(data.get("bridging_method", "none")).lower()
            if mode not in ("none", ""):
                atomic = model.atomic_model
                analytical = InnerPotentialAtomicModel(
                    type_map=model.get_type_map(),
                    mode=mode,
                    rcut=model.get_rcut(),
                    sel=model.get_sel(),
                    truncation=data.get("bridging_truncation", DEFAULT_TRUNCATION),
                )
                composed = LinearEnergyAtomicModel(
                    models=[atomic, analytical],
                    type_map=model.get_type_map(),
                    weights="sum",
                    atom_exclude_types=atomic.atom_exclude_types,
                    pair_exclude_types=atomic.pair_exclude_types,
                )
                model = LinearEnergyModel(atomic_model_=composed)
            if model_type == "sezm_native_spin":
                return NativeSpinEnergyModel(
                    atomic_model_=model.atomic_model,
                    spin=Spin.deserialize(data["spin"]),
                )
            return model
        if model_type == "sezm_atomic":
            data = normalize_sezm_atomic(data)
            if data["fitting"]["type"] == "property":
                return PropertyModel.deserialize(data)
        return super().deserialize(data)


@BaseModel.register("SeZMProperty")
@BaseModel.register("sezm_property")
@BaseModel.register("DPA4Property")
@BaseModel.register("dpa4_property")
class DPA4PropertyModel(PropertyModel):
    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> PropertyModel:
        """Restore a PyTorch SeZM invariant-property model record."""
        if str(data.get("type", "")).lower() in (
            "sezmproperty",
            "sezm_property",
            "dpa4property",
            "dpa4_property",
        ):
            check_version_compatibility(int(data.get("@version", 1)), 2, 1)
            if str(data.get("bridging_method", "none")).lower() not in ("none", ""):
                raise ValueError(
                    "DPA4 property fitting does not support analytical bridging."
                )
            data = data["atomic_model"]
        if data.get("type") == "sezm_atomic":
            data = normalize_sezm_atomic(data)
        return super().deserialize(data)
