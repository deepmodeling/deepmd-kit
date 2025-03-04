# SPDX-License-Identifier: LGPL-3.0-or-later
"""Root of the deepmd package, exposes all public classes and submodules."""

import deepmd.tf.utils.network as network
from deepmd.utils.entry_point import (
    load_entry_point,
)

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
from .modifier import (
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
load_entry_point("deepmd")

__all__ = [
    "DeepEval",
    "DeepPotential",
    "DipoleChargeModifier",
    "__version__",
    "cluster",
    "descriptor",
    "fit",
    "loss",
    "network",
    "nvnmd",
    "utils",
]
