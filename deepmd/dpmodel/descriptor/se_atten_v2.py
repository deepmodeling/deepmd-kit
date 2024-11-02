# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    NetworkCollection,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .dpa1 import (
    DescrptDPA1,
    NeighborGatedAttention,
)


@BaseDescriptor.register("se_atten_v2")
class DescrptSeAttenV2(DescrptDPA1):
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: Union[list[int], int],
        ntypes: int,
        neuron: list[int] = [25, 50, 100],
        axis_neuron: int = 8,
        tebd_dim: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = False,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        scaling_factor=1.0,
        normalize: bool = True,
        temperature: Optional[float] = None,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        concat_output_tebd: bool = True,
        spin: Optional[Any] = None,
        stripped_type_embedding: Optional[bool] = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: Optional[list[str]] = None,
        # consistent with argcheck, not used though
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        DescrptDPA1.__init__(
            self,
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            axis_neuron=axis_neuron,
            tebd_dim=tebd_dim,
            tebd_input_mode="strip",
            resnet_dt=resnet_dt,
            trainable=trainable,
            type_one_side=type_one_side,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=attn_mask,
            exclude_types=exclude_types,
            env_protection=env_protection,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
            smooth_type_embedding=True,
            concat_output_tebd=concat_output_tebd,
            spin=spin,
            stripped_type_embedding=stripped_type_embedding,
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            # consistent with argcheck, not used though
            seed=seed,
        )

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        obj = self.se_atten
        data = {
            "@class": "Descriptor",
            "type": "se_atten_v2",
            "@version": 2,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "ntypes": obj.ntypes,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "tebd_dim": obj.tebd_dim,
            "set_davg_zero": obj.set_davg_zero,
            "attn": obj.attn,
            "attn_layer": obj.attn_layer,
            "attn_dotr": obj.attn_dotr,
            "attn_mask": False,
            "activation_function": obj.activation_function,
            "resnet_dt": obj.resnet_dt,
            "scaling_factor": obj.scaling_factor,
            "normalize": obj.normalize,
            "temperature": obj.temperature,
            "trainable_ln": obj.trainable_ln,
            "ln_eps": obj.ln_eps,
            "type_one_side": obj.type_one_side,
            "concat_output_tebd": self.concat_output_tebd,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[obj.precision]).name,
            "embeddings": obj.embeddings.serialize(),
            "embeddings_strip": obj.embeddings_strip.serialize(),
            "attention_layers": obj.dpa1_attention.serialize(),
            "env_mat": obj.env_mat.serialize(),
            "type_embedding": self.type_embedding.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": to_numpy_array(obj["davg"]),
                "dstd": to_numpy_array(obj["dstd"]),
            },
            ## to be updated when the options are supported.
            "trainable": self.trainable,
            "spin": None,
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeAttenV2":
        """Deserialize from dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        type_embedding = data.pop("type_embedding")
        attention_layers = data.pop("attention_layers")
        data.pop("env_mat")
        embeddings_strip = data.pop("embeddings_strip")
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
        obj = cls(**data)

        obj.se_atten["davg"] = variables["davg"]
        obj.se_atten["dstd"] = variables["dstd"]
        obj.se_atten.embeddings = NetworkCollection.deserialize(embeddings)
        obj.se_atten.embeddings_strip = NetworkCollection.deserialize(embeddings_strip)
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)
        obj.se_atten.dpa1_attention = NeighborGatedAttention.deserialize(
            attention_layers
        )
        return obj
