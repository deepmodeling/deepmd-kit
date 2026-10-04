# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hardware-accelerated SeZM/DPA4 operators.

This package hosts ``make_fx``-composable Triton implementations of SeZM hot
paths.  Kernel entry points are internal implementation details of the SeZM
descriptor; the package-level API only exposes availability.
"""

from .force_assembly import (
    FORCE_ASSEMBLY_TRITON_AVAILABLE,
)
from .radial_mix import (
    RADIAL_MIX_TRITON_AVAILABLE,
)
from .so2_block_gemm import (
    SO2_BLOCK_GEMM_TRITON_AVAILABLE,
)
from .so2_rotation import (
    TRITON_ROTATION_AVAILABLE,
)
from .so2_stack_fp16x3 import (
    STACK_FP16X3_TRITON_AVAILABLE,
)
from .so2_value_path import (
    SO2_VALUE_PATH_TRITON_AVAILABLE,
)
from .wigner_monomials import (
    WIGNER_MONOMIALS_TRITON_AVAILABLE,
)

# Every kernel module guards its ``@triton.jit`` definitions behind a ``triton``
# import, so the module-level checks are equivalent. Expose a single
# package-level availability flag.
TRITON_AVAILABLE = (
    TRITON_ROTATION_AVAILABLE
    and RADIAL_MIX_TRITON_AVAILABLE
    and SO2_BLOCK_GEMM_TRITON_AVAILABLE
    and SO2_VALUE_PATH_TRITON_AVAILABLE
    and STACK_FP16X3_TRITON_AVAILABLE
    and WIGNER_MONOMIALS_TRITON_AVAILABLE
    and FORCE_ASSEMBLY_TRITON_AVAILABLE
)

__all__ = [
    "TRITON_AVAILABLE",
]
