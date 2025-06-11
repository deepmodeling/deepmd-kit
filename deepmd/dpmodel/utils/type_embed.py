# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    support_array_api,
)
from deepmd.dpmodel.common import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.utils.network import (
    EmbeddingNet,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
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
    use_econf_tebd: bool, Optional
        Whether to use electronic configuration type embedding.
    use_tebd_bias : bool, Optional
        Whether to use bias in the type embedding layer.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        *,
        ntypes: int,
        neuron: list[int],
        resnet_dt: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        padding: bool = False,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: Optional[list[str]] = None,
    ) -> None:
        self.ntypes = ntypes
        self.neuron = neuron
        self.seed = seed
        self.resnet_dt = resnet_dt
        self.precision = precision
        self.activation_function = str(activation_function)
        self.trainable = trainable
        self.padding = padding
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        embed_input_dim = ntypes
        if self.use_econf_tebd:
            self.econf_tebd, embed_input_dim = get_econf_tebd(
                self.type_map, precision=self.precision
            )
        self.embedding_net = EmbeddingNet(
            embed_input_dim,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
            seed=self.seed,
            bias=self.use_tebd_bias,
        )

    @support_array_api(version="2022.12")
    def call(self) -> np.ndarray:
        """Compute the type embedding network."""
        sample_array = self.embedding_net[0]["w"]
        xp = array_api_compat.array_namespace(sample_array)
        if not self.use_econf_tebd:
            embed = self.embedding_net(xp.eye(self.ntypes, dtype=sample_array.dtype))
        else:
            embed = self.embedding_net(self.econf_tebd)
        if self.padding:
            embed_pad = xp.zeros((1, embed.shape[-1]), dtype=embed.dtype)
            embed = xp.concat([embed, embed_pad], axis=0)
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
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data_cls = data.pop("@class")
        assert data_cls == "TypeEmbedNet", f"Invalid class {data_cls}"

        embedding_net = EmbeddingNet.deserialize(data.pop("embedding"))
        # compat with version 1
        if "use_tebd_bias" not in data:
            data["use_tebd_bias"] = True
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
            "@version": 2,
            "ntypes": self.ntypes,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "precision": self.precision,
            "activation_function": self.activation_function,
            "trainable": self.trainable,
            "padding": self.padding,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            "embedding": self.embedding_net.serialize(),
        }

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        if not self.use_econf_tebd:
            do_resnet = self.neuron[0] in [
                self.ntypes,
                self.ntypes * 2,
                len(type_map),
                len(type_map) * 2,
            ]
            assert not do_resnet or self.activation_function == "Linear", (
                "'activation_function' must be 'Linear' when performing type changing on resnet structure!"
            )
            first_layer_matrix = self.embedding_net.layers[0].w
            eye_vector = np.eye(self.ntypes, dtype=PRECISION_DICT[self.precision])
            # preprocess for resnet connection
            if self.neuron[0] == self.ntypes:
                first_layer_matrix += eye_vector
            elif self.neuron[0] == self.ntypes * 2:
                first_layer_matrix += np.concatenate([eye_vector, eye_vector], axis=-1)

            # randomly initialize params for the unseen types
            rng = np.random.default_rng()
            if has_new_type:
                extend_type_params = rng.random(
                    [len(type_map), first_layer_matrix.shape[-1]],
                    dtype=first_layer_matrix.dtype,
                )
                first_layer_matrix = np.concatenate(
                    [first_layer_matrix, extend_type_params], axis=0
                )

            first_layer_matrix = first_layer_matrix[remap_index]
            new_ntypes = len(type_map)
            eye_vector = np.eye(new_ntypes, dtype=PRECISION_DICT[self.precision])

            if self.neuron[0] == new_ntypes:
                first_layer_matrix -= eye_vector
            elif self.neuron[0] == new_ntypes * 2:
                first_layer_matrix -= np.concatenate([eye_vector, eye_vector], axis=-1)

            self.embedding_net.layers[0].num_in = new_ntypes
            self.embedding_net.layers[0].w = first_layer_matrix
        else:
            self.econf_tebd, embed_input_dim = get_econf_tebd(
                type_map, precision=self.precision
            )
        self.type_map = type_map
        self.ntypes = len(type_map)


def get_econf_tebd(type_map, precision: str = "default"):
    from deepmd.utils.econf_embd import (
        ECONF_DIM,
    )
    from deepmd.utils.econf_embd import (
        normalized_electronic_configuration_embedding as electronic_configuration_embedding,
    )
    from deepmd.utils.econf_embd import type_map as periodic_table

    assert type_map is not None, (
        "When using electronic configuration type embedding, type_map must be provided!"
    )

    missing_types = [t for t in type_map if t not in periodic_table]
    assert not missing_types, (
        "When using electronic configuration type embedding, "
        "all element in type_map should be in periodic table! "
        f"Found these invalid elements: {missing_types}"
    )
    econf_tebd = np.array(
        [electronic_configuration_embedding[kk] for kk in type_map],
        dtype=PRECISION_DICT[precision],
    )
    embed_input_dim = ECONF_DIM
    return econf_tebd, embed_input_dim
