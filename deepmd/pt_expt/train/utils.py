# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training utility functions."""

from __future__ import (
    annotations,
)

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

from deepmd.dpmodel.train.inference import (
    infer_compile_enabled as infer_compile_enabled,
)
from deepmd.dpmodel.train.inference import infer_env_defaults as infer_env_defaults
from deepmd.dpmodel.train.inference import (
    infer_matmul_precision,
)
from deepmd.dpmodel.train.inference import scoped_env_defaults as scoped_env_defaults

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
    )


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
