# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

from deepmd.backend.backend import (
    Backend,
)

log = logging.getLogger(__name__)


def convert_backend(
    *,  # Enforce keyword-only arguments
    INPUT: str,
    OUTPUT: str,
    atomic_virial: bool = False,
    **kwargs: Any,
) -> None:
    """Convert a model file from one backend to another.

    Parameters
    ----------
    INPUT : str
        The input model file.
    OUTPUT : str
        The output model file.
    atomic_virial : bool
        If True, export .pt2/.pte models with per-atom virial correction.
        This adds ~2.5x inference cost.  Default False.  Silently ignored
        (with a warning) for backends that don't support the flag.

    Notes
    -----
    Backend conversion preserves an explicit ``lower_input_kind`` reported by
    the source serializer. Sources without this metadata retain the target's
    automatic lower selection for backward compatibility. A target backend
    that cannot represent an explicit non-dense lower is rejected rather than
    silently changing the model function.
    """
    inp_backend: Backend = Backend.detect_backend_by_model(INPUT)()
    out_backend: Backend = Backend.detect_backend_by_model(OUTPUT)()
    inp_hook = inp_backend.serialize_hook
    out_hook = out_backend.deserialize_hook
    data = inp_hook(INPUT)
    import inspect

    sig = inspect.signature(out_hook)
    hook_kwargs: dict[str, Any] = {}
    lower_input_kind = data.get("lower_input_kind")
    if "lower_kind" in sig.parameters:
        hook_kwargs["lower_kind"] = (
            lower_input_kind if lower_input_kind is not None else "auto"
        )
    elif lower_input_kind not in (None, "nlist"):
        raise ValueError(
            f"Cannot preserve lower_input_kind {lower_input_kind!r} when "
            f"converting to output backend {out_backend.name!r}: its "
            "deserializer does not accept a lower_kind. Retrain or freeze the "
            "model with that backend instead of converting this artifact."
        )
    if "do_atomic_virial" in sig.parameters:
        hook_kwargs["do_atomic_virial"] = atomic_virial
    elif atomic_virial:
        log.warning(
            "--atomic-virial is only meaningful for pt_expt .pt2/.pte "
            "outputs; ignoring it for output backend %s",
            out_backend.name,
        )
    out_hook(OUTPUT, data, **hook_kwargs)
