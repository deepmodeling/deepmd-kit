# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.descriptor.se_atten_v2 import DescrptSeAttenV2 as DescrptSeAttenV2DP

from ..common import (
    register_dpmodel_mapping,
)
from .base_descriptor import (
    BaseDescriptor,
)
from .dpa1 import (
    DescrptDPA1,
)


@BaseDescriptor.register("se_atten_v2")
class DescrptSeAttenV2(DescrptDPA1, DescrptSeAttenV2DP):
    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeAttenV2":
        obj = DescrptSeAttenV2DP.deserialize.__func__(cls, data)
        obj._refresh_tf2_trackable_lists()
        return obj


register_dpmodel_mapping(
    DescrptSeAttenV2DP,
    lambda v: DescrptSeAttenV2.deserialize(v.serialize()),
)
