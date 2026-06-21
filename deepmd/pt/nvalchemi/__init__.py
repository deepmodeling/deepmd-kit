# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optional bridge between DeePMD-kit SeZM / DPA-4 models and NVIDIA's
``nvalchemi-toolkit`` molecular-dynamics framework.

``nvalchemi-toolkit`` is an optional dependency.  Importing this subpackage
without it installed raises a clear, actionable error instead of an opaque
``ModuleNotFoundError`` deep in the import chain.

Example
-------
::

    from deepmd.pt.nvalchemi import DPA4Wrapper
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.neighbors import compute_neighbors

    model = DPA4Wrapper.from_checkpoint("model.pt", device="cuda")
    batch = Batch.from_data_list([data], device="cuda")
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    out = model(batch)  # {"energy": (B, 1), "forces": (N, 3), ...}
"""

from __future__ import (
    annotations,
)

try:
    import nvalchemi  # noqa: F401
except ImportError as e:  # pragma: no cover - exercised only without the dep
    raise ImportError(
        "deepmd.pt.nvalchemi requires the optional `nvalchemi-toolkit` package. "
        "Install it with `pip install deepmd-kit[nvalchemi]` "
        "(or `pip install nvalchemi-toolkit`)."
    ) from e

from .dpa4wrapper import (
    DPA4Wrapper,
    SeZMWrapper,
)

__all__ = ["DPA4Wrapper", "SeZMWrapper"]
