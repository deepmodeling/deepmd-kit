# SPDX-License-Identifier: LGPL-3.0-or-later
# Import to register converters. ``dpa4_nn`` registers the dpmodel -> pt_expt
# converters for the DPA4 interaction block (activation checkpointing) and the
# SO(2) modules / radial MLP (opt-in Triton kernels, trainable-weight promotion),
# so the auto-wrapped descriptor tree picks up those subclasses.
from . import (  # noqa: F401
    dpa4_nn,
    repflows,
    repformers,
    se_t_tebd_block,
)
from .base_descriptor import (
    BaseDescriptor,
)
from .dpa1 import (
    DescrptDPA1,
)
from .dpa2 import (
    DescrptDPA2,
)
from .dpa3 import (
    DescrptDPA3,
)
from .dpa4 import (
    DescrptDPA4,
)
from .hybrid import (
    DescrptHybrid,
)
from .se_atten_v2 import (
    DescrptSeAttenV2,
)
from .se_e2_a import (
    DescrptSeA,
)
from .se_r import (
    DescrptSeR,
)
from .se_t import (
    DescrptSeT,
)
from .se_t_tebd import (
    DescrptSeTTebd,
)

__all__ = [
    "BaseDescriptor",
    "DescrptDPA1",
    "DescrptDPA2",
    "DescrptDPA3",
    "DescrptDPA4",
    "DescrptHybrid",
    "DescrptSeA",
    "DescrptSeAttenV2",
    "DescrptSeR",
    "DescrptSeT",
    "DescrptSeTTebd",
]
