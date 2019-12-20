from .env import set_mkl
from .DeepEval   import DeepEval
from .DeepPot    import DeepPot
from .DeepDipole import DeepDipole
from .DeepPolar  import DeepPolar
from .DeepPolar  import DeepGlobalPolar
from .DeepWFC    import DeepWFC

set_mkl()

try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__

