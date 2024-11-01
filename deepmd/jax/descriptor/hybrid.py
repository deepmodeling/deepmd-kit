# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("hybrid")
@flax_module
class DescrptHybrid(DescrptHybridDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"nlist_cut_idx"}:
            value = [ArrayAPIVariable(to_jax_array(vv)) for vv in value]
        elif name in {"descrpt_list"}:
            value = [BaseDescriptor.deserialize(vv.serialize()) for vv in value]

        return super().__setattr__(name, value)
