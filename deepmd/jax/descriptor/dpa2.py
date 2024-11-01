# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DescrptDPA2DP
from deepmd.dpmodel.utils.network import Identity as IdentityDP
from deepmd.dpmodel.utils.network import NativeLayer as NativeLayerDP
from deepmd.jax.common import (
    ArrayAPIVariable,
    flax_module,
    to_jax_array,
)
from deepmd.jax.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.jax.descriptor.dpa1 import (
    DescrptBlockSeAtten,
)
from deepmd.jax.descriptor.repformers import (
    DescrptBlockRepformers,
)
from deepmd.jax.descriptor.se_t_tebd import (
    DescrptBlockSeTTebd,
)
from deepmd.jax.utils.network import (
    NativeLayer,
)
from deepmd.jax.utils.type_embed import (
    TypeEmbedNet,
)


@BaseDescriptor.register("dpa2")
@flax_module
class DescrptDPA2(DescrptDPA2DP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"repinit"}:
            value = DescrptBlockSeAtten.deserialize(value.serialize())
        elif name in {"repinit_three_body"}:
            if value is not None:
                value = DescrptBlockSeTTebd.deserialize(value.serialize())
        elif name in {"repformers"}:
            value = DescrptBlockRepformers.deserialize(value.serialize())
        elif name in {"type_embedding"}:
            value = TypeEmbedNet.deserialize(value.serialize())
        elif name in {"g1_shape_tranform", "tebd_transform"}:
            if value is None:
                pass
            elif isinstance(value, NativeLayerDP):
                value = NativeLayer.deserialize(value.serialize())
            elif isinstance(value, IdentityDP):
                # IdentityDP doesn't contain any value - it's good to go
                pass
            else:
                raise ValueError(f"Unknown layer type: {type(value)}")
        return super().__setattr__(name, value)
