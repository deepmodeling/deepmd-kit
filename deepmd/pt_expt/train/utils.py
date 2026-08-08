# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training utility functions."""

from __future__ import (
    annotations,
)

import os
from contextlib import (
    contextmanager,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
    )

    import torch


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    """
    Count the parameters of a module.

    Parameters
    ----------
    module : torch.nn.Module
        The module to inspect.

    Returns
    -------
    tuple[int, int]
        The number of trainable parameters and the total number of parameters.
    """
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def infer_env_defaults(validating_params: dict[str, Any]) -> dict[str, str]:
    """
    Translate the eval-time policy options into environment defaults.

    Models sample these variables once, while they are being constructed, so
    the configuration has to reach them through the environment rather than
    through a constructor argument. A variable exported by the user takes
    precedence; see :func:`scoped_env_defaults`.

    Parameters
    ----------
    validating_params : dict[str, Any]
        The normalized ``validating`` section.

    Returns
    -------
    dict[str, str]
        The environment variables requested by the configuration.
    """
    flags = {
        "compiled_infer": "DP_COMPILE_INFER",
        "tf32_infer": "DP_TF32_INFER",
        "amp_infer": "DP_AMP_INFER",
    }
    return {
        name: "1" for flag, name in flags.items() if validating_params.get(flag, False)
    }


@contextmanager
def scoped_env_defaults(defaults: dict[str, str]) -> Generator[None, None, None]:
    """Temporarily set missing environment variables and restore them afterward."""
    previous = {key: os.environ.get(key) for key in defaults}
    try:
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_best_checkpoint_dir(
    validating_params: dict[str, Any], save_ckpt: str
) -> Path:
    """
    Resolve the directory for full-validation best checkpoints.

    Parameters
    ----------
    validating_params : dict
        The ``validating`` section of the training configuration.
    save_ckpt : str
        The regular checkpoint prefix from ``training.save_ckpt``.

    Returns
    -------
    Path
        ``validating.save_best_dir`` when set, otherwise the directory derived
        from ``save_ckpt``.
    """
    save_best_dir = validating_params.get("save_best_dir")
    if save_best_dir:
        return Path(save_best_dir)
    return Path(save_ckpt).parent
