# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
)

import numpy as np

from deepmd.dpmodel.common import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.utils.network import (
    EmbeddingNet,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class TypeEmbedNet(NativeOP):
    r"""Type embedding network.

    Parameters
    ----------
    ntypes : int
        Number of atom types
    neuron : list[int]
        Number of neurons in each hidden layers of the embedding net
    resnet_dt
        Time-step `dt` in the resnet construction: y = x + dt * \phi (Wx + b)
    activation_function
        The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
        The precision of the embedding net parameters. Supported options are |PRECISION|
    trainable
        If the weights of embedding net are trainable.
    seed
        Random seed for initializing the network parameters.
    padding
        Concat the zero padding to the output, as the default embedding of empty type.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        neuron: List[int] = [],
        resnet_dt: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        trainable: bool = True,
        seed: Optional[int] = None,
        padding: bool = False,
    ) -> None:
        self.ntypes = ntypes
        self.neuron = neuron
        self.seed = seed
        self.resnet_dt = resnet_dt
        self.precision = precision
        self.activation_function = str(activation_function)
        self.trainable = trainable
        self.padding = padding
        self.embedding_net = EmbeddingNet(
            ntypes,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
        )

    def call(self) -> np.ndarray:
        """Compute the type embedding network."""
        embed = self.embedding_net(
            np.eye(self.ntypes, dtype=PRECISION_DICT[self.precision])
        )
        if self.padding:
            embed = np.pad(embed, ((0, 1), (0, 0)), mode="constant")
        return embed

    @classmethod
    def deserialize(cls, data: dict):
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
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data_cls = data.pop("@class")
        assert data_cls == "TypeEmbedNet", f"Invalid class {data_cls}"

        embedding_net = EmbeddingNet.deserialize(data.pop("embedding"))
        type_embedding_net = cls(**data)
        type_embedding_net.embedding_net = embedding_net
        return type_embedding_net

    def serialize(self) -> dict:
        """Serialize the model.

        Returns
        -------
        dict
            The serialized data
        """
        return {
            "@class": "TypeEmbedNet",
            "@version": 1,
            "ntypes": self.ntypes,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "precision": self.precision,
            "activation_function": self.activation_function,
            "trainable": self.trainable,
            "padding": self.padding,
            "embedding": self.embedding_net.serialize(),
        }
