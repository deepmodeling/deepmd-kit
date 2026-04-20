# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.backend.backend import (
    Backend,
)


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
        This adds ~2.5x inference cost.  Default False.
    """
    inp_backend: Backend = Backend.detect_backend_by_model(INPUT)()
    out_backend: Backend = Backend.detect_backend_by_model(OUTPUT)()
    inp_hook = inp_backend.serialize_hook
    out_hook = out_backend.deserialize_hook
    data = inp_hook(INPUT)
    # Forward atomic_virial to pt_expt deserialize_to_file if applicable
    import inspect

    sig = inspect.signature(out_hook)
    if "do_atomic_virial" in sig.parameters:
        out_hook(OUTPUT, data, do_atomic_virial=atomic_virial)
    else:
        if atomic_virial:
            raise ValueError(
                "--atomic-virial is only supported for pt_expt .pt2/.pte outputs"
            )
        out_hook(OUTPUT, data)
