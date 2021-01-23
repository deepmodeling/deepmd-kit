from deepmd.train.DeepDipole import DeepDipole
from deepmd.train.DeepEval import DeepEval
from deepmd.train.DeepPolar import DeepGlobalPolar
from deepmd.train.DeepPolar import DeepPolar
from deepmd.train.DeepPot import DeepPot
from deepmd.train.DeepWFC import DeepWFC
from deepmd.train.env import set_mkl

set_mkl()

try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__
