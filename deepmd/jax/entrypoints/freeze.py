# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze utilities for the JAX backend."""

from pathlib import (
    Path,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)


def freeze(
    *,
    checkpoint_folder: str,
    output: str,
    **kwargs: object,
) -> None:
    """Freeze a JAX checkpoint into a serialized model file.

    Parameters
    ----------
    checkpoint_folder : str
        Location of either the checkpoint directory or a folder containing the
        stable ``checkpoint`` pointer.
    output : str
        Output model filename or prefix. The JAX model suffix is added when the
        filename has no supported backend suffix.
    **kwargs
        Other CLI arguments accepted for backend entry-point compatibility.
    """
    del kwargs

    checkpoint_path = Path(checkpoint_folder)
    if (checkpoint_path / "checkpoint").is_file():
        checkpoint_pointer = (checkpoint_path / "checkpoint").read_text().strip()
        checkpoint_folder = str(checkpoint_path / checkpoint_pointer)

    output = format_model_suffix(
        output,
        preferred_backend="jax",
        strict_prefer=True,
    )
    data = serialize_from_file(checkpoint_folder)
    deserialize_to_file(output, data)
