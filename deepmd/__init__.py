from .env import set_mkl
from .deep_eval   import DeepEval
from .deep_pot    import DeepPot
from .deep_dipole import DeepDipole
from .deep_polar  import DeepPolar
from .deep_polar  import DeepGlobalPolar
from .deep_wfc    import DeepWFC

set_mkl()

try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__

