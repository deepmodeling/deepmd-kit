# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import paddle

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pd.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pd.model.network.mlp import (
    NetworkCollection,
)
from deepmd.pd.model.network.network import (
    TypeEmbedNetConsistent,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    RESERVED_PRECISION_DICT,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .se_atten import (
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
        neuron: list = [25, 50, 100],
        axis_neuron: int = 16,
        tebd_dim: int = 8,
        set_davg_zero: bool = True,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        scaling_factor: int = 1.0,
        normalize=True,
        temperature=None,
        concat_output_tebd: bool = True,
        trainable: bool = True,
        trainable_ln: bool = True,
        ln_eps: Optional[float] = 1e-5,
        type_one_side: bool = False,
        stripped_type_embedding: Optional[bool] = None,
        seed: Optional[Union[int, list[int]]] = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: Optional[list[str]] = None,
        # not implemented
        spin=None,
        type: Optional[str] = None,
    ) -> None:
        r"""Construct smooth version of embedding net of type `se_atten_v2`.

        Parameters
        ----------
        rcut : float
            The cut-off radius :math:`r_c`
        rcut_smth : float
            From where the environment matrix should be smoothed :math:`r_s`
        sel : list[int], int
            list[int]: sel[i] specifies the maxmum number of type i atoms in the cut-off radius
            int: the total maxmum number of atoms in the cut-off radius
        ntypes : int
            Number of element types
        neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
        axis_neuron : int
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
        tebd_dim : int
            Dimension of the type embedding
        set_davg_zero : bool
            Set the shift of embedding net input to zero.
        attn : int
            Hidden dimension of the attention vectors
        attn_layer : int
            Number of attention layers
        attn_dotr : bool
            If dot the angular gate to the attention weights
        attn_mask : bool
            (Only support False to keep consistent with other backend references.)
            (Not used in this version.)
            If mask the diagonal of attention weights
        activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
        precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
        resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
        exclude_types : list[list[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
        env_protection : float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
        scaling_factor : float
            The scaling factor of normalization in calculations of attention weights.
            If `temperature` is None, the scaling of attention weights is (N_dim * scaling_factor)**0.5
        normalize : bool
            Whether to normalize the hidden vectors in attention weights calculation.
        temperature : float
            If not None, the scaling of attention weights is `temperature` itself.
        concat_output_tebd : bool
            Whether to concat type embedding at the output of the descriptor.
        trainable : bool
            If the weights of this descriptors are trainable.
        trainable_ln : bool
            Whether to use trainable shift and scale weights in layer normalization.
        ln_eps : float, Optional
            The epsilon value for layer normalization.
        type_one_side : bool
            If 'False', type embeddings of both neighbor and central atoms are considered.
            If 'True', only type embeddings of neighbor atoms are considered.
            Default is 'False'.
        stripped_type_embedding : bool, Optional
            (Deprecated, kept only for compatibility.)
            Whether to strip the type embedding into a separate embedding network.
            Setting this parameter to `True` is equivalent to setting `tebd_input_mode` to 'strip'.
            Setting it to `False` is equivalent to setting `tebd_input_mode` to 'concat'.
            The default value is `None`, which means the `tebd_input_mode` setting will be used instead.
        seed : int, Optional
            Random seed for parameter initialization.
        use_econf_tebd : bool, Optional
            Whether to use electronic configuration type embedding.
        use_tebd_bias : bool, Optional
            Whether to use bias in the type embedding layer.
        type_map : list[str], Optional
            A list of strings. Give the name to each type of atoms.
        spin
            (Only support None to keep consistent with other backend references.)
            (Not used in this version. Not-none option is not implemented.)
            The old implementation of deepspin.
        """
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
            set_davg_zero=set_davg_zero,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=attn_mask,
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            exclude_types=exclude_types,
            env_protection=env_protection,
            scaling_factor=scaling_factor,
            normalize=normalize,
            temperature=temperature,
            concat_output_tebd=concat_output_tebd,
            trainable=trainable,
            trainable_ln=trainable_ln,
            ln_eps=ln_eps,
            smooth_type_embedding=True,
            type_one_side=type_one_side,
            stripped_type_embedding=stripped_type_embedding,
            seed=seed,
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            # not implemented
            spin=spin,
            type=type,
        )

    def serialize(self) -> dict:
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
            "attn": obj.attn_dim,
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
            "precision": RESERVED_PRECISION_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "embeddings_strip": obj.filter_layers_strip.serialize(),
            "attention_layers": obj.dpa1_attention.serialize(),
            "env_mat": DPEnvMat(obj.rcut, obj.rcut_smth).serialize(),
            "type_embedding": self.type_embedding.embedding.serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            "trainable": self.trainable,
            "spin": None,
        }
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeAttenV2":
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

        def t_cvt(xx):
            return paddle.to_tensor(xx, dtype=obj.se_atten.prec, place=env.DEVICE)

        obj.type_embedding.embedding = TypeEmbedNetConsistent.deserialize(
            type_embedding
        )
        obj.se_atten["davg"] = t_cvt(variables["davg"])
        obj.se_atten["dstd"] = t_cvt(variables["dstd"])
        obj.se_atten.filter_layers = NetworkCollection.deserialize(embeddings)
        obj.se_atten.filter_layers_strip = NetworkCollection.deserialize(
            embeddings_strip
        )
        obj.se_atten.dpa1_attention = NeighborGatedAttention.deserialize(
            attention_layers
        )
        return obj
