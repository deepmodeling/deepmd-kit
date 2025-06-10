# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Final,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from deepmd.dpmodel.utils.type_embed import (
    get_econf_tebd,
)
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
    to_torch_tensor,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def Tensor(*shape):
    return torch.empty(shape, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)


class SimpleLinear(nn.Module):
    use_timestep: Final[bool]

    def __init__(
        self,
        num_in,
        num_out,
        bavg=0.0,
        stddev=1.0,
        use_timestep=False,
        activate=None,
        bias: bool = True,
    ) -> None:
        """Construct a linear layer.

        Args:
        - num_in: Width of input tensor.
        - num_out: Width of output tensor.
        - use_timestep: Apply time-step to weight.
        - activate: type of activate func.
        """
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.use_timestep = use_timestep
        self.activate = ActivationFn(activate)

        self.matrix = nn.Parameter(data=Tensor(num_in, num_out))
        nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        if bias:
            self.bias = nn.Parameter(data=Tensor(1, num_out))
            nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        else:
            self.bias = None
        if self.use_timestep:
            self.idt = nn.Parameter(data=Tensor(1, num_out))
            nn.init.normal_(self.idt.data, mean=0.1, std=0.001)

    def forward(self, inputs):
        """Return X*W+b."""
        xw = torch.matmul(inputs, self.matrix)
        hidden = xw + self.bias if self.bias is not None else xw
        hidden = self.activate(hidden)
        if self.use_timestep:
            hidden = hidden * self.idt
        return hidden


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ) -> None:
        super().__init__(
            d_in,
            d_out,
            bias=bias,
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0) -> None:
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _glorot_uniform_init(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True) -> None:
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self) -> None:
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")


class NonLinearHead(nn.Module):
    def __init__(self, input_dim, out_dim, activation_fn, hidden=None) -> None:
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = SimpleLinear(input_dim, hidden, activate=activation_fn)
        self.linear2 = SimpleLinear(hidden, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None) -> None:
        super().__init__()
        self.dense = SimpleLinear(embed_dim, embed_dim)
        self.activation_fn = ActivationFn(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)

        if weight is None:
            weight = nn.Linear(
                embed_dim, output_dim, bias=False, dtype=env.GLOBAL_PT_FLOAT_PRECISION
            ).weight
        self.weight = weight
        self.bias = nn.Parameter(
            torch.zeros(output_dim, dtype=env.GLOBAL_PT_FLOAT_PRECISION)  # pylint: disable=no-explicit-dtype,no-explicit-device
        )

    def forward(self, features, masked_tokens: Optional[torch.Tensor] = None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ResidualDeep(nn.Module):
    def __init__(
        self, type_id, embedding_width, neuron, bias_atom_e, out_dim=1, resnet_dt=False
    ) -> None:
        """Construct a filter on the given element as neighbor.

        Args:
        - typei: Element ID.
        - embedding_width: Embedding width per atom.
        - neuron: Number of neurons in each hidden layers of the embedding net.
        - resnet_dt: Using time-step in the ResNet construction.
        """
        super().__init__()
        self.type_id = type_id
        self.neuron = [embedding_width, *neuron]
        self.out_dim = out_dim

        deep_layers = []
        for ii in range(1, len(self.neuron)):
            one = SimpleLinear(
                num_in=self.neuron[ii - 1],
                num_out=self.neuron[ii],
                use_timestep=(
                    resnet_dt and ii > 1 and self.neuron[ii - 1] == self.neuron[ii]
                ),
                activate="tanh",
            )
            deep_layers.append(one)
        self.deep_layers = nn.ModuleList(deep_layers)
        if not env.ENERGY_BIAS_TRAINABLE:
            bias_atom_e = 0
        self.final_layer = SimpleLinear(self.neuron[-1], self.out_dim, bias_atom_e)

    def forward(self, inputs):
        """Calculate decoded embedding for each atom.

        Args:
        - inputs: Embedding net output per atom. Its shape is [nframes*nloc, self.embedding_width].

        Returns
        -------
        - `torch.Tensor`: Output layer with shape [nframes*nloc, self.neuron[-1]].
        """
        outputs = inputs
        for idx, linear in enumerate(self.deep_layers):
            if idx > 0 and linear.num_in == linear.num_out:
                outputs = outputs + linear(outputs)
            else:
                outputs = linear(outputs)
        outputs = self.final_layer(outputs)
        return outputs


class TypeEmbedNet(nn.Module):
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
        # nn.init.normal_(self.embedding.weight[:-1], mean=bavg, std=stddev)

    def forward(self, atype):
        """
        Args:
            atype: Type of each input, [nframes, nloc] or [nframes, nloc, nnei].

        Returns
        -------
        type_embedding:

        """
        return torch.embedding(self.embedding(atype.device), atype)

    def get_full_embedding(self, device: torch.device):
        """
        Get the type embeddings of all types.

        Parameters
        ----------
        device : torch.device
            The device on which to perform the computation.

        Returns
        -------
        type_embedding : torch.Tensor
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
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        else:
            raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        self.embedding.change_type_map(type_map=type_map)


class TypeEmbedNetConsistent(nn.Module):
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
            self.econf_tebd = to_torch_tensor(econf_tebd)
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
            param.requires_grad = trainable

    def forward(self, device: torch.device):
        """Caulate type embedding network.

        Returns
        -------
        type_embedding: torch.Tensor
            Type embedding network.
        """
        if not self.use_econf_tebd:
            embed = self.embedding_net(
                torch.eye(self.ntypes, dtype=self.prec, device=device)
            )
        else:
            assert self.econf_tebd is not None
            embed = self.embedding_net(self.econf_tebd.to(device))
        if self.padding:
            embed = torch.cat(
                [embed, torch.zeros(1, embed.shape[1], dtype=self.prec, device=device)]
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
            first_layer_matrix = self.embedding_net.layers[0].matrix.data
            eye_vector = torch.eye(
                self.ntypes, dtype=self.prec, device=first_layer_matrix.device
            )
            # preprocess for resnet connection
            if self.neuron[0] == self.ntypes:
                first_layer_matrix += eye_vector
            elif self.neuron[0] == self.ntypes * 2:
                first_layer_matrix += torch.concat([eye_vector, eye_vector], dim=-1)

            # randomly initialize params for the unseen types
            if has_new_type:
                extend_type_params = torch.rand(
                    [len(type_map), first_layer_matrix.shape[-1]],
                    device=first_layer_matrix.device,
                    dtype=first_layer_matrix.dtype,
                )
                first_layer_matrix = torch.cat(
                    [first_layer_matrix, extend_type_params], dim=0
                )

            first_layer_matrix = first_layer_matrix[remap_index]
            new_ntypes = len(type_map)
            eye_vector = torch.eye(
                new_ntypes, dtype=self.prec, device=first_layer_matrix.device
            )

            if self.neuron[0] == new_ntypes:
                first_layer_matrix -= eye_vector
            elif self.neuron[0] == new_ntypes * 2:
                first_layer_matrix -= torch.concat([eye_vector, eye_vector], dim=-1)

            self.embedding_net.layers[0].num_in = new_ntypes
            self.embedding_net.layers[0].matrix = nn.Parameter(data=first_layer_matrix)
        else:
            econf_tebd, embed_input_dim = get_econf_tebd(
                type_map, precision=self.precision
            )
            self.econf_tebd = to_torch_tensor(econf_tebd)
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
