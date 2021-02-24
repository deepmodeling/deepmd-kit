"""Root of the deepmd package, exposes all public classes and submodules."""

import deepmd.utils.network as network

from . import cluster, descriptor, fit, loss, utils
from .env import set_mkl
from .infer import DeepPotential
from .infer.data_modifier import DipoleChargeModifier

set_mkl()

try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__

__all__ = [
    "descriptor",
    "fit",
    "loss",
    "utils",
    "cluster",
    "network",
    "DeepEval",
    "DeepPotential",
    "DipoleChargeModifier",
]
