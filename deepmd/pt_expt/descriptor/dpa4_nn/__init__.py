# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt overrides for DPA4/SeZM sub-modules.

These wrappers inject PyTorch-runtime behavior that the array-API dpmodel
implementation cannot express:

- :mod:`block` -- eval-time activation checkpointing of the interaction units.
- :mod:`so2` -- opt-in fused Triton kernels for the SO(2) rotation and the
  dynamic radial degree mixer.
- :mod:`radial` -- a torch-native radial embedding MLP whose linear / norm
  weights are trainable parameters (the dpmodel list mixes modules with a bare
  activation function, which the generic conversion cannot turn into a
  ``ModuleList``).
- :mod:`wignerd` -- opt-in fused Triton monomial fast path for the Wigner-D
  ``l = 2`` contraction and the shared ``l >= 3`` monomial kernels.

Importing this package registers the dpmodel -> pt_expt converters (via
``torch_module``), so the auto-wrapped descriptor tree picks up these subclasses
instead of the generic dpmodel wrappers.
"""

from . import (  # noqa: F401
    block,
    radial,
    so2,
    wignerd,
)
