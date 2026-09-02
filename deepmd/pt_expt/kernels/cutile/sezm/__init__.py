# SPDX-License-Identifier: LGPL-3.0-or-later
"""cuTile kernels covering the SeZM / DPA4 inference path.

The modules follow the stages of one interaction block. Each owns its generated
kernels, its operator registration and one public entry point:

:mod:`.wigner_monomials`
    quaternion monomial bases for the Wigner-D blocks of degree two and above;
:mod:`.so2_rotate_mix`
    the source gather, the block-diagonal rotation into the edge-aligned frame,
    and the edge-conditioned radial degree mixing;
:mod:`.so2_mixing_stack`
    the complete gated SO(2) mixing stack;
:mod:`.flash_atten`
    the inverse rotation, the attention weight, the destination reduction, and
    the CSR row offsets every segmented kernel here needs;
:mod:`.force_assembly`
    the force and per-atom virial segment reduction;
:mod:`.so2_value_path`
    the factory binding the rotate-and-mix and stack operators into the
    convolution's value path.

Supporting modules: :mod:`.indexing` for the reduced coefficient layout and its
padded tile extents, :mod:`.tile_configs` and :mod:`.tile_config_data` for launch
configuration, and :mod:`.sweep_tile_configs` for regenerating it.
"""

from __future__ import (
    annotations,
)

__all__: list[str] = []
