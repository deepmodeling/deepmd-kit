# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.repflows import (
    DescrptBlockRepflows as DescrptBlockRepflowsDP,
)
from deepmd.dpmodel.descriptor.repflows import RepFlowLayer as RepFlowLayerDP

from ..common import (
    to_array_api_strict_array,
)
from ..utils.exclude_mask import (
    PairExcludeMask,
)
from ..utils.network import (
    NativeLayer,
)


class DescrptBlockRepflows(DescrptBlockRepflowsDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_array_api_strict_array(value)
        elif name in {"layers"}:
            value = [RepFlowLayer.deserialize(layer.serialize()) for layer in value]
        elif name in {"edge_embd", "angle_embd"}:
            value = NativeLayer.deserialize(value.serialize())
        elif name in {"env_mat_edge", "env_mat_angle"}:
            # env_mat doesn't store any value
            pass
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)
        else:
            pass

        return super().__setattr__(name, value)


class RepFlowLayer(RepFlowLayerDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "node_self_mlp",
            "node_sym_linear",
            "node_edge_linear",
            "edge_self_linear",
            "a_compress_n_linear",
            "a_compress_e_linear",
            "edge_angle_linear1",
            "edge_angle_linear2",
            "angle_self_linear",
        }:
            if value is not None:
                value = NativeLayer.deserialize(value.serialize())
        elif name in {"n_residual", "e_residual", "a_residual"}:
            value = [to_array_api_strict_array(vv) for vv in value]
        else:
            pass
        return super().__setattr__(name, value)
