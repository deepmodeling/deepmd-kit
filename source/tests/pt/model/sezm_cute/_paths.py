# SPDX-License-Identifier: LGPL-3.0-or-later
"""Source paths for CuTe kernels that cannot be imported without CUDA."""

from pathlib import (
    Path,
)

from deepmd.pt_expt.kernels.cute import (
    sezm,
)

CUTE_ROOT = Path(sezm.__path__[0])
