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

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
    )

#: Accepted ``DP_TF32_INFER`` values and the matmul precision each selects.
_TF32_INFER_PRECISIONS = {"0": "highest", "1": "high", "2": "medium"}
_BOOL_ENV_VALUES = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
}


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


def infer_matmul_precision() -> str:
    """
    Resolve the eval-time matmul precision from ``DP_TF32_INFER``.

    Returns
    -------
    str
        A ``torch.set_float32_matmul_precision`` level: ``"highest"`` for
        ``0`` (the default), ``"high"`` for ``1``, ``"medium"`` for ``2``.

    Raises
    ------
    ValueError
        If ``DP_TF32_INFER`` is set to anything other than 0, 1 or 2.
    """
    value = os.environ.get("DP_TF32_INFER", "0").strip().lower()
    if value not in _TF32_INFER_PRECISIONS:
        raise ValueError(f"DP_TF32_INFER must be one of 0/1/2, got {value!r}")
    return _TF32_INFER_PRECISIONS[value]


def infer_compile_enabled() -> bool:
    """
    Resolve whether evaluation forwards use the compiled model.

    Returns
    -------
    bool
        Whether ``DP_COMPILE_INFER`` enables compiled evaluation.

    Raises
    ------
    ValueError
        If ``DP_COMPILE_INFER`` is not a supported boolean value.
    """
    value = os.environ.get("DP_COMPILE_INFER", "0").strip().lower()
    if value not in _BOOL_ENV_VALUES:
        choices = "/".join(_BOOL_ENV_VALUES)
        raise ValueError(f"DP_COMPILE_INFER must be one of {choices}, got {value!r}")
    return _BOOL_ENV_VALUES[value]


class MatmulPrecisionPolicy:
    """
    The fp32 matmul precision that each forward mode runs at.

    Training and evaluation are controlled separately, mirroring the split the
    pt backend applies inside ``SeZMModel``: training follows
    ``training.enable_tf32``, while evaluation follows ``DP_TF32_INFER``, so a
    run may train on TF32 yet validate at full precision.

    The eval level is sampled at construction, which must therefore happen
    while the ``validating`` section's environment defaults are in scope; see
    :func:`scoped_env_defaults`.

    Parameters
    ----------
    enable_tf32 : bool
        Whether training forwards may use TF32 tensor cores.

    Attributes
    ----------
    train_precision : str
        Matmul precision of training forwards.
    eval_precision : str
        Matmul precision of evaluation forwards.
    """

    def __init__(self, enable_tf32: bool) -> None:
        self.train_precision = "high" if enable_tf32 else "highest"
        self.eval_precision = infer_matmul_precision()

    @contextmanager
    def applied(self, *, training: bool) -> Generator[None, None, None]:
        """
        Run a block at the matmul precision of the requested mode.

        The previous process-wide level is restored on exit, so a caller that
        evaluates in the middle of training does not disturb the surrounding
        training precision. The setting only reaches CUDA matmuls, so on a
        CPU-only build the block runs unchanged.

        Parameters
        ----------
        training : bool
            Whether the block is a training forward.

        Yields
        ------
        None
            Control returns to the caller with the precision in effect.
        """
        if not torch.cuda.is_available():
            yield
            return
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(
            self.train_precision if training else self.eval_precision
        )
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous)


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
