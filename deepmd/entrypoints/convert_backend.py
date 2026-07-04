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
    """
    inp_backend: Backend = Backend.detect_backend_by_model(INPUT)()
    out_backend: Backend = Backend.detect_backend_by_model(OUTPUT)()
    inp_hook = inp_backend.serialize_hook
    out_hook = out_backend.deserialize_hook
    data = inp_hook(INPUT)
    # Pass the optional flags a backend's deserialize hook accepts, probed by
    # signature so scripts stay backend-agnostic. ``lower_kind="auto"`` lets the
    # pt_expt hook pick the deploy-optimal lowering (the graph lower with its
    # fused operator for a graph-eligible .pt2, the dense nlist lower otherwise);
    # ``do_atomic_virial`` forwards ``--atomic-virial``.
    import inspect

    sig = inspect.signature(out_hook)
    hook_kwargs: dict[str, Any] = {}
    if "lower_kind" in sig.parameters:
        hook_kwargs["lower_kind"] = "auto"
    if "do_atomic_virial" in sig.parameters:
        hook_kwargs["do_atomic_virial"] = atomic_virial
    elif atomic_virial:
        log.warning(
            "--atomic-virial is only meaningful for pt_expt .pt2/.pte "
            "outputs; ignoring it for output backend %s",
            out_backend.name,
        )
    out_hook(OUTPUT, data, **hook_kwargs)
