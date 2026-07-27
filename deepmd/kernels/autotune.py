# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze-time Triton launch-configuration autotuning.

Custom-kernel families register a tuner through :func:`register_autotuner`; a
model-freezing entry point calls :func:`run_autotune` once, before tracing, so
the exported artifact bakes launch configurations tuned for the exact
deployment shapes and GPU. A tuner sweeps only the shape keys its kernels need
that are not already covered by the built-in tables, keeping repeat freezes
cheap.

Autotuning is a no-op below ``DP_TRITON_INFER`` level 2 (level-1 kernels run a
single shape-independent configuration) and off CUDA -- Triton launch tables
are GPU-specific and AOTInductor artifacts are not portable across GPU models,
so the tuning target is always the local deployment GPU.

Adding a new custom-kernel family is one registration call at import time; the
freeze paths need no change.
"""

from __future__ import (
    annotations,
)

import logging
from collections.abc import (
    Callable,
)

import torch

from deepmd.kernels.utils import (
    triton_infer_level,
)

log = logging.getLogger(__name__)

# Family name -> tuner(model, level, device). Keying by name makes repeated
# registration (e.g. module re-import) idempotent.
Autotuner = Callable[[torch.nn.Module, int, torch.device], None]
_AUTOTUNERS: dict[str, Autotuner] = {}


def register_autotuner(name: str, tuner: Autotuner) -> None:
    """Register a kernel family's freeze-time launch-configuration tuner.

    Parameters
    ----------
    name : str
        Unique family identifier; a repeated name replaces the prior tuner.
    tuner : Callable[[torch.nn.Module, int, torch.device], None]
        Sweeps the shape keys ``model`` needs for this family that are not
        already covered and registers the winners into the family's launch
        table. Receives the resolved ``DP_TRITON_INFER`` level and the target
        CUDA device.
    """
    _AUTOTUNERS[name] = tuner


def run_autotune(model: torch.nn.Module, device: torch.device) -> None:
    """Tune every registered kernel family for ``model`` on ``device``.

    Called by a freeze path after the model is built and before it is traced,
    so ``resolve_*_config`` lookups made while tracing return freshly tuned
    launches. A no-op below ``DP_TRITON_INFER`` level 2 or off CUDA.

    Parameters
    ----------
    model : torch.nn.Module
        The model about to be frozen; tuners walk it for their shape keys.
    device : torch.device
        The target CUDA device the artifact will run on.
    """
    level = triton_infer_level()
    if level < 2 or device.type != "cuda" or not torch.cuda.is_available():
        return
    for name, tuner in _AUTOTUNERS.items():
        # A tuning failure must degrade to the built-in / default launch tables
        # rather than abort the freeze; the frozen artifact stays correct, only
        # its launch configuration is untuned for uncovered shapes.
        try:
            tuner(model, level, device)
        except Exception:
            log.warning(
                "Triton autotuner %r failed; keeping default launch configurations.",
                name,
                exc_info=True,
            )
