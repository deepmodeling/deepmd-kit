# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4/SeZM-family energy model (pt_expt backend).

A THIN subclass of the descriptor-agnostic :class:`EnergyModel` that owns
exactly the dpa4-family concerns:

- the family's registry wire types (``dpa4_ener``/``sezm_ener``
  fitting-type dispatch keys and the pt model-type strings
  ``dpa4``/``sezm``/``sezm_atomic``);
- the pt-checkpoint conversion (pt's ``SeZMModel`` wrapper and
  ``sezm_atomic`` dict layouts), moved here from ``BaseModel`` so the
  generic base assumes nothing about a specific descriptor family.

May later absorb the dpa4-family builder (``get_sezm_model``) as a
``get_model`` classmethod, unifying dpa4 construction with the standard
registry dispatch.
"""

from typing import (
    Any,
)

from deepmd.utils.dpa4_serialization import (
    normalize_sezm_atomic,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .ener_model import (
    EnergyModel,
)
from .model import (
    BaseModel,
)


@BaseModel.register("dpa4_ener")
@BaseModel.register("sezm_ener")
@BaseModel.register("dpa4")
@BaseModel.register("DPA4")
@BaseModel.register("sezm")
@BaseModel.register("SeZM")
@BaseModel.register("sezm_atomic")
class DPA4EnergyModel(EnergyModel):
    r"""Energy model for the DPA4/SeZM descriptor family.

    Behaviorally identical to :class:`EnergyModel`; additionally owns the
    pt-checkpoint interop: pt's ``SeZMModel`` serialises with a model-level
    wrapper (``{type: "SeZM", atomic_model: <sezm_atomic dict>,
    bridging_method, bridging_truncation, lora}``) whose atomic dict carries
    pt-only extras -- :meth:`deserialize` recognises those layouts,
    normalises them to the standard flat dict, and delegates to the
    generic path.
    """

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DPA4EnergyModel":
        model_type = str(data.get("type", "standard")).lower()
        if model_type in ("sezm", "dpa4"):
            return cls.deserialize(cls._unwrap_pt_sezm_model(data))
        if model_type == "sezm_atomic":
            return cls.deserialize(normalize_sezm_atomic(data))
        return super().deserialize(data)

    @staticmethod
    def _unwrap_pt_sezm_model(data: dict[str, Any]) -> dict[str, Any]:
        """Unwrap pt's ``SeZMModel`` serialization to the inner atomic dict."""
        data = data.copy()
        # Validate the pt SeZM model wrapper version before discarding it so a
        # future incompatible wrapper schema is not silently mis-deserialized
        # (the wrapper only carries the guarded bridging/lora extras below, so
        # the accepted range is narrow).
        check_version_compatibility(int(data.get("@version", 1)), 2, 1)
        bridging_method = str(data.get("bridging_method", "none")).lower()
        if data.get("lora") is not None:
            raise NotImplementedError(
                "Deserializing a pt SeZM/DPA4 checkpoint with `lora` is "
                "not supported in pt_expt."
            )
        atomic_model = data.get("atomic_model")
        if atomic_model is None:
            raise ValueError(
                "SeZM/DPA4 model data is missing the 'atomic_model' entry."
            )
        if bridging_method not in ("none", ""):
            # pt serializes bridging as a flag on its model wrapper; our
            # architecture is a linear COMPOSITION with a different dict
            # shape. No conversion is claimed -- rebuild from the training
            # config instead.
            raise NotImplementedError(
                "Deserializing a pt SeZM/DPA4 checkpoint with "
                f"`bridging_method`={data.get('bridging_method')!r} is not "
                "supported; rebuild the bridged model from its config."
            )
        return atomic_model
