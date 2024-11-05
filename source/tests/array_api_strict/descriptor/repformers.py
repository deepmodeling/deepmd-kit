# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.repformers import (
    Atten2EquiVarApply as Atten2EquiVarApplyDP,
)
from deepmd.dpmodel.descriptor.repformers import Atten2Map as Atten2MapDP
from deepmd.dpmodel.descriptor.repformers import (
    Atten2MultiHeadApply as Atten2MultiHeadApplyDP,
)
from deepmd.dpmodel.descriptor.repformers import (
    DescrptBlockRepformers as DescrptBlockRepformersDP,
)
from deepmd.dpmodel.descriptor.repformers import LocalAtten as LocalAttenDP
from deepmd.dpmodel.descriptor.repformers import RepformerLayer as RepformerLayerDP

from ..common import (
    to_array_api_strict_array,
)
from ..utils.exclude_mask import (
    PairExcludeMask,
)
from ..utils.network import (
    LayerNorm,
    NativeLayer,
)


class DescrptBlockRepformers(DescrptBlockRepformersDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mean", "stddev"}:
            value = to_array_api_strict_array(value)
        elif name in {"layers"}:
            value = [RepformerLayer.deserialize(layer.serialize()) for layer in value]
        elif name == "g2_embd":
            value = NativeLayer.deserialize(value.serialize())
        elif name == "env_mat":
            # env_mat doesn't store any value
            pass
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)

        return super().__setattr__(name, value)


class Atten2Map(Atten2MapDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mapqk"}:
            value = NativeLayer.deserialize(value.serialize())
        return super().__setattr__(name, value)


class Atten2MultiHeadApply(Atten2MultiHeadApplyDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mapv", "head_map"}:
            value = NativeLayer.deserialize(value.serialize())
        return super().__setattr__(name, value)


class Atten2EquiVarApply(Atten2EquiVarApplyDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"head_map"}:
            value = NativeLayer.deserialize(value.serialize())
        return super().__setattr__(name, value)


class LocalAtten(LocalAttenDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"mapq", "mapkv", "head_map"}:
            value = NativeLayer.deserialize(value.serialize())
        return super().__setattr__(name, value)


class RepformerLayer(RepformerLayerDP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"linear1", "linear2", "g1_self_mlp", "proj_g1g2", "proj_g1g1g2"}:
            if value is not None:
                value = NativeLayer.deserialize(value.serialize())
        elif name in {"g1_residual", "g2_residual", "h2_residual"}:
            value = [to_array_api_strict_array(vv) for vv in value]
        elif name in {"attn2g_map"}:
            if value is not None:
                value = Atten2Map.deserialize(value.serialize())
        elif name in {"attn2_mh_apply"}:
            if value is not None:
                value = Atten2MultiHeadApply.deserialize(value.serialize())
        elif name in {"attn2_lm"}:
            if value is not None:
                value = LayerNorm.deserialize(value.serialize())
        elif name in {"attn2_ev_apply"}:
            if value is not None:
                value = Atten2EquiVarApply.deserialize(value.serialize())
        elif name in {"loc_attn"}:
            if value is not None:
                value = LocalAtten.deserialize(value.serialize())
        return super().__setattr__(name, value)
