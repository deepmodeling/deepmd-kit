from . import descriptor
from . import fit
from . import loss
from . import utils
import deepmd.utils.network as network
from .infer.deep_eval   import DeepEval
from .infer.deep_pot    import DeepPot
from .infer.deep_dipole import DeepDipole
from .infer.deep_polar  import DeepPolar
from .infer.deep_polar  import DeepGlobalPolar
from .infer.deep_wfc    import DeepWFC
from .infer.data_modifier    import DipoleChargeModifier
from .env import set_mkl

set_mkl()

try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__

