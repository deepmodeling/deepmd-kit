# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared NeighborGraph builder selection for pt_expt.

Single owner of the ``None`` / ``"auto"`` resolution ladder used by the
model-level default-flip (:meth:`deepmd.pt_expt.model.make_model` /
``_resolve_graph_method``), DeepEval graph-form ``.pt2`` inference, and
compiled training's eager builder outside the traced lower.

Policy
------
* CUDA device: ``nv`` if importable, else ``vesin`` if importable, else
  ``dense``.
* CPU device: ``vesin`` if importable, else ``dense``.

``ase`` is never chosen automatically (explicit opt-in only). All builders
emit the same carry-all neighbor set; the choice is performance-only.
Builders run eagerly outside traced / compiled regions (export and training
compile use synthetic dense graph inputs), so flipping the default does not
change ``.pt2`` artifacts.

Perf note: at small benchmark sizes the dense all-pairs builder is typically
not the bottleneck; the O(N) win appears for large systems (N of a few
thousand atoms and up). Time builders manually on a large system to document
the crossover before relying on auto for production throughput.
"""

from __future__ import (
    annotations,
)

import torch

from deepmd.pt.utils.nv_nlist import (
    is_nv_available,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)


def resolve_auto_graph_builder(
    device: torch.device | str,
) -> str:
    """Resolve ``neighbor_graph_method`` ``None`` / ``"auto"`` to a concrete builder.

    Parameters
    ----------
    device
        Device the coordinates live on (or will be moved to). Controls whether
        the CUDA-only ``nv`` builder is eligible.

    Returns
    -------
    str
        One of ``"nv"``, ``"vesin"``, or ``"dense"``.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        if is_nv_available():
            return "nv"
        if is_vesin_torch_available():
            return "vesin"
        return "dense"
    if is_vesin_torch_available():
        return "vesin"
    return "dense"
