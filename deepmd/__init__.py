"""Root of the deepmd package, exposes all public classes and submodules."""

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
import deepmd.utils.network as network

from . import cluster, descriptor, fit, loss, utils
from .env import set_mkl
from .infer import DeepEval, DeepPotential
from .infer.data_modifier import DipoleChargeModifier

set_mkl()

try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__

# load third-party plugins
for ep in metadata.entry_points().get('deepmd', []):
    ep.load()

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
