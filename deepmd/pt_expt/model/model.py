# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class BaseModel(make_base_model()):
    """Base class for pt_expt models.

    Provides the plugin registry so that model classes can be
    registered with ``@BaseModel.register("ener")`` etc.

    See Also
    --------
    deepmd.dpmodel.model.base_model.BaseBaseModel
        Backend-independent BaseModel class.
    """

    # The pt backend's ``SeZMModel`` (model_type "SeZM", aliases dpa4/sezm)
    # serialises with a *model-level wrapper*: ``{type: "SeZM",
    # atomic_model: <sezm_atomic dict>, bridging_method, bridging_r_*, lora}``,
    # and its atomic model uses ``type: "sezm_atomic"`` carrying pt-only
    # extras (``dens_fitting``/``active_mode`` plus a ``dens_force_rmsd``
    # @variable).  pt_expt builds the equivalent DPA4 model via the generic
    # ``make_model`` path, whose ``serialize()`` emits the standard atomic
    # dict directly (``type: "standard"``).  To load a pt-trained checkpoint
    # into pt_expt (the serialization-compat / checkpoint-interop
    # requirement), recognise the wrapper, reject the pt-only features pt_expt
    # does not implement (when they are non-default), strip the rest, and
    # delegate to the standard path.  The nested descriptor/fitting dicts are
    # already backend-agnostic dpmodel serializations and pass through intact.
    _SEZM_MODEL_TYPES = frozenset({"sezm", "dpa4"})
    _SEZM_ATOMIC_TYPES = frozenset({"sezm_atomic"})

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "BaseModel":
        model_type = str(data.get("type", "standard"))
        if model_type.lower() in cls._SEZM_MODEL_TYPES:
            return cls.deserialize(cls._unwrap_pt_sezm_model(data))
        if model_type.lower() in cls._SEZM_ATOMIC_TYPES:
            return cls.deserialize(cls._normalize_pt_sezm_atomic(data))
        return super().deserialize(data)

    @staticmethod
    def _unwrap_pt_sezm_model(data: dict[str, Any]) -> dict[str, Any]:
        """Unwrap pt's ``SeZMModel`` serialization to the inner atomic dict."""
        data = data.copy()
        # The pt SeZM model wrapper serialises with ``@version`` 1.  Validate
        # before discarding it so a future incompatible wrapper schema is not
        # silently mis-deserialized (the wrapper only carries the guarded
        # bridging/lora extras below, so the accepted range is narrow).
        check_version_compatibility(int(data.get("@version", 1)), 1, 1)
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
            # Map pt's model-level wrapper key onto OUR atomic-dict key: the
            # atomic layer owns the InterPotential here. The wrapper's
            # ``bridging_r_inner``/``bridging_r_outer`` are dropped -- pt's
            # descriptor serialization already carries the InnerClamp radii,
            # so the wrapper copies are redundant for reconstruction.
            atomic_model = dict(atomic_model)
            atomic_model["bridging_method"] = str(data["bridging_method"]).upper()
        return atomic_model

    @staticmethod
    def _normalize_pt_sezm_atomic(data: dict[str, Any]) -> dict[str, Any]:
        """Convert a pt ``sezm_atomic`` dict to a standard atomic dict.

        Strips the pt-only ``dens`` head state (``dens_fitting`` /
        ``active_mode`` / the ``dens_force_rmsd`` @variable) and rewrites the
        ``type``/``@version`` so the generic dpmodel atomic-model deserialize
        accepts it.  A non-energy active mode or a populated dens head is
        rejected because pt_expt only implements the energy path.
        """
        data = data.copy()
        # pt emits ``@version`` 3 for ``sezm_atomic``; the standard dpmodel
        # atomic-model deserialize requires exactly 2.  The only schema delta
        # between the two is the stripped ``dens`` state below, so coercion is
        # safe for the known-compatible range {2, 3}.  Validate the incoming
        # version BEFORE coercing so a future incompatible pt schema (e.g.
        # ``@version`` 4) is rejected loudly instead of mis-deserialized.
        check_version_compatibility(int(data.get("@version", 2)), 3, 2)
        if data.pop("dens_fitting", None) is not None:
            raise NotImplementedError(
                "Deserializing a pt SeZM/DPA4 checkpoint with a `dens` "
                "fitting head is not supported in pt_expt."
            )
        active_mode = data.pop("active_mode", None)
        if active_mode not in (None, "ener"):
            raise NotImplementedError(
                f"Deserializing a pt SeZM/DPA4 checkpoint in active_mode "
                f"{active_mode!r} is not supported in pt_expt (energy only)."
            )
        variables = data.get("@variables")
        if isinstance(variables, dict):
            data["@variables"] = {
                k: v for k, v in variables.items() if k in ("out_bias", "out_std")
            }
        # The standard dpmodel atomic-model deserialize checks @version == 2.
        data["@version"] = 2
        data["type"] = "standard"
        return data
