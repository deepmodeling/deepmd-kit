# SPDX-License-Identifier: LGPL-3.0-or-later
"""Freeze-time autotuning of custom-kernel launch choices.

Custom-kernel families register a tuner through :func:`register_autotuner`; a
model-freezing entry point calls :func:`run_autotune` once, before tracing, so
the exported artifact bakes launch choices tuned for the exact deployment
shapes and GPU: the Triton launch configurations of the DPA1 kernels, and the
fitting routes of the fused energy-force operator of compressed DPA4C. A tuner
measures only the keys its kernels need that are not already covered, keeping
repeat freezes cheap.

Each tuner runs from the ``DP_TRITON_INFER`` level it registers with (level-1
Triton kernels run a single shape-independent configuration, so the Triton
tuners start at level 2), for the lower kinds whose programs call its kernels,
and only on CUDA: the tables are GPU-specific and AOTInductor artifacts are not
portable across GPU models, so the tuning target is always the local
deployment GPU.

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

from deepmd.pt_expt.kernels.utils import (
    triton_infer_level,
)

log = logging.getLogger(__name__)

# Family name -> (tuner(model, level, device), lowest level, lower kinds or
# None for all). Keying by name makes repeated registration (e.g. module
# re-import) idempotent.
Autotuner = Callable[[torch.nn.Module, int, torch.device], None]
_AUTOTUNERS: dict[str, tuple[Autotuner, int, frozenset[str] | None]] = {}


def register_autotuner(
    name: str,
    tuner: Autotuner,
    min_level: int = 2,
    lower_kinds: tuple[str, ...] | None = None,
) -> None:
    """Register a kernel family's freeze-time tuner.

    Parameters
    ----------
    name : str
        Unique family identifier; a repeated name replaces the prior tuner.
    tuner : Callable[[torch.nn.Module, int, torch.device], None]
        Measures the keys ``model`` needs for this family that are not already
        covered and registers the winners into the family's table. Receives
        the resolved ``DP_TRITON_INFER`` level and the target CUDA device.
    min_level : int, optional
        Lowest ``DP_TRITON_INFER`` level at which the tuner runs, by default 2,
        that of the Triton launch tables.
    lower_kinds : tuple[str, ...] or None, optional
        Lower kinds whose programs call the kernels of the family; a freeze
        to another lower kind skips the tuner. By default it runs for every
        lower kind.
    """
    _AUTOTUNERS[name] = (
        tuner,
        min_level,
        None if lower_kinds is None else frozenset(lower_kinds),
    )


def run_autotune(model: torch.nn.Module, device: torch.device, lower_kind: str) -> None:
    """Tune every registered kernel family for ``model`` on ``device``.

    Called by a freeze path after the model is built and before it is traced,
    so the lookups made while tracing return freshly tuned choices. A no-op
    off CUDA; each tuner runs from the level it registered with, for the lower
    kinds it registered with.

    Parameters
    ----------
    model : torch.nn.Module
        The model about to be frozen; tuners walk it for their shape keys.
    device : torch.device
        The target CUDA device the artifact will run on.
    lower_kind : str
        The lower kind of the program the freeze exports.
    """
    level = triton_infer_level()
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    for name, (tuner, min_level, lower_kinds) in _AUTOTUNERS.items():
        if level < min_level or (
            lower_kinds is not None and lower_kind not in lower_kinds
        ):
            continue
        # A tuning failure must degrade to the built-in / default tables rather
        # than abort the freeze; the frozen artifact stays correct, only its
        # launch choices are untuned for uncovered keys.
        try:
            tuner(model, level, device)
        except Exception:
            log.warning(
                "Autotuner %r failed; keeping the default launch choices.",
                name,
                exc_info=True,
            )
