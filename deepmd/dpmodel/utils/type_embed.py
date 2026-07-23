# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
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


def _array_device_or_none(array: Array) -> Any:
    try:
        return array_api_compat.device(array)
    except AttributeError:
        return None


def remap_atype_to_padding(atype: Array, ntypes_with_padding: int) -> Array:
    """Map negative placeholder types to a table's final padding row."""
    xp = array_api_compat.array_namespace(atype)
    return xp.where(
        atype >= 0,
        atype,
        xp.full_like(atype, ntypes_with_padding - 1),
    )


def take_type_embedding(type_embedding: Array, atype: Array) -> Array:
    """Gather type embeddings, mapping virtual atom types to the padding row.

    Descriptor type-embedding tables append an all-zero final row for virtual
    atoms. Negative placeholder types must select that row explicitly because
    negative gather indices either wrap or fail depending on the array backend.
    """
    # The caller's atom-type array determines the active backend. Model
    # conversion keeps the embedding table in that same namespace while
    # preserving trainable tensors and their gradients.
    xp = array_api_compat.array_namespace(atype)
    safe_atype = remap_atype_to_padding(atype, type_embedding.shape[0])
    return xp.take(type_embedding, xp.astype(safe_atype, xp.int64), axis=0)


class TypeEmbedNet(NativeOP):
    r"""Type embedding network.

    Each atom type :math:`t` is represented by a one-hot vector
    :math:`\mathbf e_t` (or an electronic-configuration vector), then mapped
    by an embedding network :math:`\mathcal N`:

    .. math::

       \mathbf T_t=\mathcal N(\mathbf e_t).

    If ``padding`` is enabled, an additional all-zero row represents padded
    neighbor-list entries.

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
        seed: int | list[int] | None = None,
        padding: bool = False,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: list[str] | None = None,
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
            trainable=trainable,
        )

    def call(self) -> Array:
        r"""Return all type embeddings :math:`\mathbf T_t=\mathcal N(\mathbf e_t)`."""
        sample_array = self.embedding_net[0]["w"]
        xp = array_api_compat.array_namespace(sample_array)
        if not self.use_econf_tebd:
            embed = self.embedding_net(
                xp.eye(
                    self.ntypes,
                    dtype=sample_array.dtype,
                    device=_array_device_or_none(sample_array),
                )
            )
        else:
            embed = self.embedding_net(self.econf_tebd)
        if self.padding:
            embed_pad = xp.zeros(
                (1, embed.shape[-1]),
                dtype=embed.dtype,
                device=_array_device_or_none(embed),
            )
            embed = xp.concat([embed, embed_pad], axis=0)
        return embed

    @classmethod
    def deserialize(cls, data: dict) -> "TypeEmbedNet":
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
        self, type_map: list[str], model_with_new_type_stat: Any = None
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
            # Use array_api_compat to handle both numpy and torch
            xp = array_api_compat.array_namespace(first_layer_matrix)
            eye_vector = xp.eye(
                self.ntypes,
                dtype=first_layer_matrix.dtype,
                device=array_api_compat.device(first_layer_matrix),
            )
            # preprocess for resnet connection
            if self.neuron[0] == self.ntypes:
                first_layer_matrix = first_layer_matrix + eye_vector
            elif self.neuron[0] == self.ntypes * 2:
                first_layer_matrix = first_layer_matrix + xp.concat(
                    [eye_vector, eye_vector], axis=-1
                )

            # randomly initialize params for the unseen types
            if has_new_type:
                # Create random params with same dtype and device as first_layer_matrix
                extend_type_params = np.random.default_rng().random(
                    [len(type_map), first_layer_matrix.shape[-1]],
                )
                extend_type_params = xp.asarray(
                    extend_type_params,
                    dtype=first_layer_matrix.dtype,
                    device=array_api_compat.device(first_layer_matrix),
                )
                first_layer_matrix = xp.concat(
                    [first_layer_matrix, extend_type_params], axis=0
                )

            first_layer_matrix = first_layer_matrix[remap_index]
            new_ntypes = len(type_map)
            eye_vector = xp.eye(
                new_ntypes,
                dtype=first_layer_matrix.dtype,
                device=array_api_compat.device(first_layer_matrix),
            )

            if self.neuron[0] == new_ntypes:
                first_layer_matrix = first_layer_matrix - eye_vector
            elif self.neuron[0] == new_ntypes * 2:
                first_layer_matrix = first_layer_matrix - xp.concat(
                    [eye_vector, eye_vector], axis=-1
                )

            self.embedding_net.layers[0].num_in = new_ntypes
            self.embedding_net.layers[0].w = first_layer_matrix
        else:
            self.econf_tebd, embed_input_dim = get_econf_tebd(
                type_map, precision=self.precision
            )
        self.type_map = type_map
        self.ntypes = len(type_map)


def get_econf_tebd(
    type_map: list[str], precision: str = "default"
) -> tuple[Array, int]:
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
