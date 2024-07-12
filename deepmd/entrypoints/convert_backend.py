# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.backend.backend import (
    Backend,
)


def convert_backend(
    *,  # Enforce keyword-only arguments
    INPUT: str,
    OUTPUT: str,
    **kwargs,
) -> None:
    """Convert a model file from one backend to another.

    Parameters
    ----------
    INPUT : str
        The input model file.
    INPUT : str
        The output model file.
    """
    inp_backend: Backend = Backend.detect_backend_by_model(INPUT)()
    out_backend: Backend = Backend.detect_backend_by_model(OUTPUT)()
    inp_hook = inp_backend.serialize_hook
    out_hook = out_backend.deserialize_hook
    data = inp_hook(INPUT)
    out_hook(OUTPUT, data)
