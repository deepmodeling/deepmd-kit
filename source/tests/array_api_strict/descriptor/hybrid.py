# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP

from ..common import (
    to_array_api_strict_array,
)
from .base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("hybrid")
class DescrptHybrid(DescrptHybridDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"nlist_cut_idx"}:
            value = [to_array_api_strict_array(vv) for vv in value]
        elif name in {"descrpt_list"}:
            value = [BaseDescriptor.deserialize(vv.serialize()) for vv in value]

        return super().__setattr__(name, value)
