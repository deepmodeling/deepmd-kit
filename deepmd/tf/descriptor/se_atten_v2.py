# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Optional,
)

from deepmd.utils.version import (
    check_version_compatibility,
)

from .descriptor import (
    Descriptor,
)
from .se_atten import (
    DescrptSeAtten,
)

log = logging.getLogger(__name__)


@Descriptor.register("se_atten_v2")
class DescrptSeAttenV2(DescrptSeAtten):
    r"""Smooth version 2.0 descriptor with attention.

    Parameters
    ----------
    rcut
            The cut-off radius :math:`r_c`
    rcut_smth
            From where the environment matrix should be smoothed :math:`r_s`
    sel : int
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    set_davg_zero
            Set the shift of embedding net input to zero.
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    attn
            The length of hidden vector during scale-dot attention computation.
    attn_layer
            The number of layers in attention mechanism.
    attn_dotr
            Whether to dot the relative coordinates on the attention weights as a gated scheme.
    attn_mask
            Whether to mask the diagonal in the attention weights.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: int,
        ntypes: int,
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        set_davg_zero: bool = False,
        exclude_types: List[List[int]] = [],
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        **kwargs,
    ) -> None:
        DescrptSeAtten.__init__(
            self,
            rcut,
            rcut_smth,
            sel,
            ntypes,
            neuron=neuron,
            axis_neuron=axis_neuron,
            resnet_dt=resnet_dt,
            trainable=trainable,
            seed=seed,
            type_one_side=type_one_side,
            set_davg_zero=set_davg_zero,
            exclude_types=exclude_types,
            activation_function=activation_function,
            precision=precision,
            uniform_seed=uniform_seed,
            attn=attn,
            attn_layer=attn_layer,
            attn_dotr=attn_dotr,
            attn_mask=attn_mask,
            tebd_input_mode="strip",
            smooth_type_embedding=True,
            **kwargs,
        )

    @classmethod
    def deserialize(cls, data: dict, suffix: str = ""):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Model
            The deserialized model
        """
        if cls is not DescrptSeAttenV2:
            raise NotImplementedError(f"Not implemented in class {cls.__name__}")
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        embedding_net_variables = cls.deserialize_network(
            data.pop("embeddings"), suffix=suffix
        )
        attention_layer_variables = cls.deserialize_attention_layers(
            data.pop("attention_layers"), suffix=suffix
        )
        data.pop("env_mat")
        variables = data.pop("@variables")
        type_one_side = data["type_one_side"]
        two_side_embeeding_net_variables = cls.deserialize_network_strip(
            data.pop("embeddings_strip"),
            suffix=suffix,
            type_one_side=type_one_side,
        )
        descriptor = cls(**data)
        descriptor.embedding_net_variables = embedding_net_variables
        descriptor.attention_layer_variables = attention_layer_variables
        descriptor.two_side_embeeding_net_variables = two_side_embeeding_net_variables
        descriptor.davg = variables["davg"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        descriptor.dstd = variables["dstd"].reshape(
            descriptor.ntypes, descriptor.ndescrpt
        )
        return descriptor

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        Parameters
        ----------
        suffix : str, optional
            The suffix of the scope

        Returns
        -------
        dict
            The serialized data
        """
        data = super().serialize(suffix)
        data.pop("smooth_type_embedding")
        data.pop("tebd_input_mode")
        data.update({"type": "se_atten_v2"})
        return data
