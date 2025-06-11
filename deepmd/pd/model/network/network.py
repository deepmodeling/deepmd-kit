# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import paddle
import paddle.nn as nn

from deepmd.dpmodel.utils.type_embed import (
    get_econf_tebd,
)
from deepmd.pd.model.network.mlp import (
    EmbeddingNet,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.utils import (
    to_paddle_tensor,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def Tensor(*shape):
    return paddle.empty(shape, dtype=env.GLOBAL_PD_FLOAT_PRECISION).to(
        device=env.DEVICE
    )


class TypeEmbedNet(nn.Layer):
    def __init__(
        self,
        type_nums,
        embed_dim,
        bavg=0.0,
        stddev=1.0,
        precision="default",
        seed: Optional[Union[int, list[int]]] = None,
        use_econf_tebd=False,
        use_tebd_bias: bool = False,
        type_map=None,
    ) -> None:
        """Construct a type embedding net."""
        super().__init__()
        self.type_nums = type_nums
        self.embed_dim = embed_dim
        self.bavg = bavg
        self.stddev = stddev
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.embedding = TypeEmbedNetConsistent(
            ntypes=self.type_nums,
            neuron=[self.embed_dim],
            padding=True,
            activation_function="Linear",
            use_econf_tebd=use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            precision=precision,
            seed=seed,
        )
        # init.normal_(self.embedding.weight[:-1], mean=bavg, std=stddev)

    def forward(self, atype):
        """
        Args:
            atype: Type of each input, [nframes, nloc] or [nframes, nloc, nnei].

        Returns
        -------
        type_embedding:

        """
        return self.embedding(atype.place)[atype]

    def get_full_embedding(self, device: str):
        """
        Get the type embeddings of all types.

        Parameters
        ----------
        device : str
            The device on which to perform the computation.

        Returns
        -------
        type_embedding : paddle.Tensor
            The full type embeddings of all types. The last index corresponds to the zero padding.
            Shape: (ntypes + 1) x tebd_dim
        """
        return self.embedding(device)

    def share_params(self, base_class, shared_level, resume=False) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only TypeEmbedNet of the same type can share params!"
        )
        if shared_level == 0:
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._sub_layers:
                self._sub_layers[item] = base_class._sub_layers[item]
        else:
            raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        self.embedding.change_type_map(type_map=type_map)


class TypeEmbedNetConsistent(nn.Layer):
    r"""Type embedding network that is consistent with other backends.

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
        """Construct a type embedding net."""
        super().__init__()
        self.ntypes = ntypes
        self.neuron = neuron
        self.seed = seed
        self.resnet_dt = resnet_dt
        self.precision = precision
        self.prec = env.PRECISION_DICT[self.precision]
        self.activation_function = str(activation_function)
        self.trainable = trainable
        self.padding = padding
        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.econf_tebd = None
        embed_input_dim = ntypes
        if self.use_econf_tebd:
            econf_tebd, embed_input_dim = get_econf_tebd(
                self.type_map, precision=self.precision
            )
            self.econf_tebd = to_paddle_tensor(econf_tebd)
        self.embedding_net = EmbeddingNet(
            embed_input_dim,
            self.neuron,
            self.activation_function,
            self.resnet_dt,
            self.precision,
            self.seed,
            bias=self.use_tebd_bias,
        )
        for param in self.parameters():
            param.stop_gradient = not trainable

    def forward(self, device: str):
        """Caulate type embedding network.

        Returns
        -------
        type_embedding: paddle.Tensor
            Type embedding network.
        """
        if not self.use_econf_tebd:
            embed = self.embedding_net(
                paddle.eye(self.ntypes, dtype=self.prec).to(device=device)
            )
        else:
            assert self.econf_tebd is not None
            embed = self.embedding_net(self.econf_tebd.to(device))
        if self.padding:
            embed = paddle.concat(
                [
                    embed,
                    paddle.zeros([1, embed.shape[1]], dtype=self.prec).to(
                        device=device
                    ),
                ]
            )
        return embed

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
            first_layer_matrix = self.embedding_net.layers[0].matrix
            eye_vector = paddle.eye(self.ntypes, dtype=self.prec).to(
                device=first_layer_matrix.place
            )
            # preprocess for resnet connection
            if self.neuron[0] == self.ntypes:
                first_layer_matrix += eye_vector
            elif self.neuron[0] == self.ntypes * 2:
                first_layer_matrix += paddle.concat([eye_vector, eye_vector], axis=-1)

            # randomly initialize params for the unseen types
            if has_new_type:
                extend_type_params = paddle.rand(
                    [len(type_map), first_layer_matrix.shape[-1]],
                    dtype=first_layer_matrix.dtype,
                ).to(device=first_layer_matrix.place)
                first_layer_matrix = paddle.concat(
                    [first_layer_matrix, extend_type_params], axis=0
                )

            first_layer_matrix = first_layer_matrix[remap_index]
            new_ntypes = len(type_map)
            eye_vector = paddle.eye(new_ntypes, dtype=self.prec).to(
                device=first_layer_matrix.place
            )

            if self.neuron[0] == new_ntypes:
                first_layer_matrix -= eye_vector
            elif self.neuron[0] == new_ntypes * 2:
                first_layer_matrix -= paddle.concat([eye_vector, eye_vector], axis=-1)

            self.embedding_net.layers[0].num_in = new_ntypes
            self.embedding_net.layers[0].matrix = self.create_parameter(
                first_layer_matrix.shape,
                dtype=first_layer_matrix.dtype,
                default_initializer=nn.initializer.Assign(first_layer_matrix),
            )
        else:
            econf_tebd, embed_input_dim = get_econf_tebd(
                type_map, precision=self.precision
            )
            self.econf_tebd = to_paddle_tensor(econf_tebd)
        self.type_map = type_map
        self.ntypes = len(type_map)

    @classmethod
    def deserialize(cls, data: dict):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        TypeEmbedNetConsistent
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
