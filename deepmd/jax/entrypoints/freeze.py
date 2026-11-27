# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze utilities for the JAX backend."""

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
    hessian: bool = False,
    head: str | None = None,
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
    hessian : bool, default=False
        Whether to include the Hessian in the frozen model outputs.
    head : str, optional
        Task head to export from a multi-task checkpoint. Required for
        multi-task inputs and omitted for a single-task checkpoint.
    **kwargs
        Other CLI arguments accepted for backend entry-point compatibility.
    """
    del kwargs

    output = format_model_suffix(
        output,
        preferred_backend="jax",
        strict_prefer=True,
    )
    data = serialize_from_file(checkpoint_folder, training=False)
    model_dict = data["model"].get("model_dict")
    if model_dict is not None:
        if head is None or head not in model_dict:
            raise ValueError(
                "A multi-task checkpoint requires a valid --task; "
                f"available tasks: {list(model_dict)}. Got {head!r}."
            )
        data = data.copy()
        script = data["model_def_script"]
        data["model"] = model_dict[head]
        data["model_def_script"] = dict(script["model_dict"][head])
        if "current_step" in script:
            data["model_def_script"]["current_step"] = script["current_step"]
        # Optimizer moments refer to the full shared graph, not a frozen head.
        data.pop("training", None)
    elif head is not None:
        raise ValueError("A single-task checkpoint does not accept --task.")
    deserialize_to_file(output, data, hessian=hessian)
