# SPDX-License-Identifier: LGPL-3.0-or-later
"""Root of the deepmd package, exposes all public classes and submodules."""

try:
    from importlib import (
        metadata,
    )
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata

import deepmd.tf.utils.network as network

from . import (
    cluster,
    descriptor,
    fit,
    loss,
    nvnmd,
    utils,
)
from .env import (
    set_mkl,
)
from .infer import (
    DeepEval,
    DeepPotential,
)
from .infer.data_modifier import (
    DipoleChargeModifier,
)

set_mkl()

try:
    from deepmd._version import version as __version__
except ImportError:
    from .__about__ import (
        __version__,
    )

# load third-party plugins
try:
    eps = metadata.entry_points(group="deepmd")
except TypeError:
    eps = metadata.entry_points().get("deepmd", [])
for ep in eps:
    ep.load()

__all__ = [
    "__version__",
    "descriptor",
    "fit",
    "loss",
    "utils",
    "cluster",
    "network",
    "DeepEval",
    "DeepPotential",
    "DipoleChargeModifier",
    "nvnmd",
]
