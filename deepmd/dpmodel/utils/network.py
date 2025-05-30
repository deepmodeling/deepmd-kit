# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native DP model format for multiple backends.

See issue #2982 for more information.
"""

import itertools
from typing import (
    Callable,
    ClassVar,
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    support_array_api,
    xp_add_at,
    xp_bincount,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def sigmoid_t(x: np.ndarray) -> np.ndarray:
    """Sigmoid."""
    if array_api_compat.is_jax_array(x):
        from deepmd.jax.env import (
            jax,
        )

        # see https://github.com/jax-ml/jax/discussions/15617
        return jax.nn.sigmoid(x)
    xp = array_api_compat.array_namespace(x)
    return 1 / (1 + xp.exp(-x))


class Identity(NativeOP):
    def __init__(self) -> None:
        super().__init__()

    def call(self, x: np.ndarray) -> np.ndarray:
        """The Identity operation layer."""
        return x

    def serialize(self) -> dict:
        return {
            "@class": "Identity",
            "@version": 1,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Identity":
        return Identity()


class NativeLayer(NativeOP):
    """Native representation of a layer.

    Parameters
    ----------
    w : np.ndarray, optional
        The weights of the layer.
    b : np.ndarray, optional
        The biases of the layer.
    idt : np.ndarray, optional
        The identity matrix of the layer.
    activation_function : str, optional
        The activation function of the layer.
    resnet : bool, optional
        Whether the layer is a residual layer.
    precision : str, optional
        The precision of the layer.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        num_in,
        num_out,
        bias: bool = True,
        use_timestep: bool = False,
        activation_function: Optional[str] = None,
        resnet: bool = False,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        prec = PRECISION_DICT[precision.lower()]
        self.precision = precision
        # only use_timestep when skip connection is established.
        use_timestep = use_timestep and (num_out == num_in or num_out == num_in * 2)
        rng = np.random.default_rng(seed)
        scale_factor = 1.0 / np.sqrt(num_out + num_in)
        self.w = rng.normal(size=(num_in, num_out), scale=scale_factor).astype(prec)
        self.b = (
            rng.normal(size=(num_out,), scale=scale_factor).astype(prec)
            if bias
            else None
        )
        self.idt = (
            rng.normal(size=(num_out,), scale=scale_factor).astype(prec)
            if use_timestep
            else None
        )
        self.activation_function = (
            activation_function if activation_function is not None else "none"
        )
        self.resnet = resnet
        self.check_type_consistency()
        self.check_shape_consistency()

    def serialize(self) -> dict:
        """Serialize the layer to a dict.

        Returns
        -------
        dict
            The serialized layer.
        """
        data = {
            "w": to_numpy_array(self.w),
            "b": to_numpy_array(self.b),
            "idt": to_numpy_array(self.idt),
        }
        return {
            "@class": "Layer",
            "@version": 1,
            "bias": self.b is not None,
            "use_timestep": self.idt is not None,
            "activation_function": self.activation_function,
            "resnet": self.resnet,
            # make deterministic
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "@variables": data,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NativeLayer":
        """Deserialize the layer from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        variables = data.pop("@variables")
        assert variables["w"] is not None and len(variables["w"].shape) == 2
        num_in, num_out = variables["w"].shape
        obj = cls(
            num_in,
            num_out,
            **data,
        )
        w, b, idt = (
            variables["w"],
            variables.get("b", None),
            variables.get("idt", None),
        )
        if b is not None:
            b = b.ravel()
        if idt is not None:
            idt = idt.ravel()
        obj.w = w
        obj.b = b
        obj.idt = idt
        obj.check_shape_consistency()
        return obj

    def check_shape_consistency(self) -> None:
        if self.b is not None and self.w.shape[1] != self.b.shape[0]:
            raise ValueError(
                f"dim 1 of w {self.w.shape[1]} is not equal to shape "
                f"of b {self.b.shape[0]}",
            )
        if self.idt is not None and self.w.shape[1] != self.idt.shape[0]:
            raise ValueError(
                f"dim 1 of w {self.w.shape[1]} is not equal to shape "
                f"of idt {self.idt.shape[0]}",
            )

    def check_type_consistency(self) -> None:
        precision = self.precision

        def check_var(var) -> None:
            if var is not None:
                # array api standard doesn't provide a API to get the dtype name
                # this is really hacked
                dtype_name = str(var.dtype).split(".")[-1]
                # assertion "float64" == "double" would fail
                assert PRECISION_DICT[dtype_name] is PRECISION_DICT[precision]

        check_var(self.w)
        check_var(self.b)
        check_var(self.idt)

    def __setitem__(self, key, value) -> None:
        if key in ("w", "matrix"):
            self.w = value
        elif key in ("b", "bias"):
            self.b = value
        elif key == "idt":
            self.idt = value
        elif key == "activation_function":
            self.activation_function = value
        elif key == "resnet":
            self.resnet = value
        elif key == "precision":
            self.precision = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("w", "matrix"):
            return self.w
        elif key in ("b", "bias"):
            return self.b
        elif key == "idt":
            return self.idt
        elif key == "activation_function":
            return self.activation_function
        elif key == "resnet":
            return self.resnet
        elif key == "precision":
            return self.precision
        else:
            raise KeyError(key)

    def dim_in(self) -> int:
        return self.w.shape[0]

    def dim_out(self) -> int:
        return self.w.shape[1]

    @support_array_api(version="2022.12")
    def call(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : np.ndarray
            The input.

        Returns
        -------
        np.ndarray
            The output.
        """
        if self.w is None or self.activation_function is None:
            raise ValueError("w, b, and activation_function must be set")
        xp = array_api_compat.array_namespace(x)
        fn = get_activation_fn(self.activation_function)
        y = (
            xp.matmul(x, self.w[...]) + self.b[...]
            if self.b is not None
            else xp.matmul(x, self.w[...])
        )
        if y.dtype != x.dtype:
            # workaround for bfloat16
            # https://github.com/jax-ml/ml_dtypes/issues/235
            y = xp.astype(y, x.dtype)
        y = fn(y)
        if self.idt is not None:
            y *= self.idt
        if self.resnet and self.w.shape[1] == self.w.shape[0]:
            y += x
        elif self.resnet and self.w.shape[1] == 2 * self.w.shape[0]:
            y += xp.concat([x, x], axis=-1)
        return y


@support_array_api(version="2022.12")
def get_activation_fn(activation_function: str) -> Callable[[np.ndarray], np.ndarray]:
    activation_function = activation_function.lower()
    if activation_function == "tanh":

        def fn(x):
            xp = array_api_compat.array_namespace(x)
            return xp.tanh(x)

        return fn
    elif activation_function == "relu":

        def fn(x):
            xp = array_api_compat.array_namespace(x)
            # https://stackoverflow.com/a/47936476/9567349
            return x * xp.astype(x > 0, x.dtype)

        return fn
    elif activation_function in ("gelu", "gelu_tf"):

        def fn(x):
            xp = array_api_compat.array_namespace(x)
            # generated by GitHub Copilot
            return (
                0.5
                * x
                * (1 + xp.tanh(xp.sqrt(xp.asarray(2 / xp.pi)) * (x + 0.044715 * x**3)))
            )

        return fn
    elif activation_function == "relu6":

        def fn(x):
            xp = array_api_compat.array_namespace(x)
            # generated by GitHub Copilot
            return xp.where(
                x < 0, xp.full_like(x, 0), xp.where(x > 6, xp.full_like(x, 6), x)
            )

        return fn
    elif activation_function == "softplus":

        def fn(x):
            xp = array_api_compat.array_namespace(x)
            # generated by GitHub Copilot
            return xp.log(1 + xp.exp(x))

        return fn
    elif activation_function == "sigmoid":

        def fn(x):
            # generated by GitHub Copilot
            return sigmoid_t(x)

        return fn
    elif activation_function == "silu":

        def fn(x):
            # generated by GitHub Copilot
            return x * sigmoid_t(x)

        return fn
    elif activation_function.startswith("silut") or activation_function.startswith(
        "custom_silu"
    ):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def silu(x):
            return x * sigmoid(x)

        def silu_grad(x):
            sig = sigmoid(x)
            return sig + x * sig * (1 - sig)

        threshold = (
            float(activation_function.split(":")[-1])
            if ":" in activation_function
            else 3.0
        )
        slope = float(silu_grad(threshold))
        const = float(silu(threshold))

        def fn(x):
            xp = array_api_compat.array_namespace(x)
            return xp.where(
                x < threshold,
                x * sigmoid_t(x),
                xp.tanh(slope * (x - threshold)) + const,
            )

        return fn
    elif activation_function.lower() in ("none", "linear"):

        def fn(x):
            return x

        return fn
    else:
        raise NotImplementedError(activation_function)


class LayerNorm(NativeLayer):
    """Implementation of Layer Normalization layer.

    Parameters
    ----------
    num_in : int
        The input dimension of the layer.
    eps : float, optional
        A small value added to prevent division by zero in calculations.
    uni_init : bool, optional
        If initialize the weights to be zeros and ones.
    trainable : bool, optional
        If the weights are trainable.
    precision : str, optional
        The precision of the layer.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        num_in: int,
        eps: float = 1e-5,
        uni_init: bool = True,
        trainable: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        self.eps = eps
        self.uni_init = uni_init
        self.num_in = num_in
        super().__init__(
            num_in=1,
            num_out=num_in,
            bias=True,
            use_timestep=False,
            activation_function=None,
            resnet=False,
            precision=precision,
            seed=seed,
        )
        xp = array_api_compat.array_namespace(self.w, self.b)
        self.w = xp.squeeze(self.w, 0)  # keep the weight shape to be [num_in]
        if self.uni_init:
            self.w = xp.ones_like(self.w)
            self.b = xp.zeros_like(self.b)
        # only to keep consistent with other backends
        self.trainable = trainable

    def serialize(self) -> dict:
        """Serialize the layer to a dict.

        Returns
        -------
        dict
            The serialized layer.
        """
        data = {
            "w": to_numpy_array(self.w),
            "b": to_numpy_array(self.b),
        }
        return {
            "@class": "LayerNorm",
            "@version": 1,
            "eps": self.eps,
            "trainable": self.trainable,
            "precision": self.precision,
            "@variables": data,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "LayerNorm":
        """Deserialize the layer from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        variables = data.pop("@variables")
        if variables["w"] is not None:
            assert len(variables["w"].shape) == 1
        if variables["b"] is not None:
            assert len(variables["b"].shape) == 1
        (num_in,) = variables["w"].shape
        obj = cls(
            num_in,
            **data,
        )
        (obj.w,) = (variables["w"],)
        (obj.b,) = (variables["b"],)
        obj._check_shape_consistency()
        return obj

    def _check_shape_consistency(self) -> None:
        if self.b is not None and self.w.shape[0] != self.b.shape[0]:
            raise ValueError(
                f"dim 1 of w {self.w.shape[0]} is not equal to shape "
                f"of b {self.b.shape[0]}",
            )

    def __setitem__(self, key, value) -> None:
        if key in ("w", "matrix"):
            self.w = value
        elif key in ("b", "bias"):
            self.b = value
        elif key == "trainable":
            self.trainable = value
        elif key == "precision":
            self.precision = value
        elif key == "eps":
            self.eps = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("w", "matrix"):
            return self.w
        elif key in ("b", "bias"):
            return self.b
        elif key == "trainable":
            return self.trainable
        elif key == "precision":
            return self.precision
        elif key == "eps":
            return self.eps
        else:
            raise KeyError(key)

    def dim_out(self) -> int:
        return self.w.shape[0]

    def call(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : np.ndarray
            The input.

        Returns
        -------
        np.ndarray
            The output.
        """
        y = self.layer_norm_numpy(x, (self.num_in,), self.w, self.b, self.eps)
        return y

    @staticmethod
    def layer_norm_numpy(x, shape, weight=None, bias=None, eps=1e-5):
        xp = array_api_compat.array_namespace(x)
        # mean and variance
        mean = xp.mean(x, axis=tuple(range(-len(shape), 0)), keepdims=True)
        var = xp.var(x, axis=tuple(range(-len(shape), 0)), keepdims=True)
        # normalize
        x_normalized = (x - mean) / xp.sqrt(var + eps)
        # shift and scale
        if weight is not None and bias is not None:
            x_normalized = x_normalized * weight + bias
        return x_normalized


def make_multilayer_network(T_NetworkLayer, ModuleBase):
    class NN(ModuleBase):
        """Native representation of a neural network.

        Parameters
        ----------
        layers : list[NativeLayer], optional
            The layers of the network.
        """

        def __init__(self, layers: Optional[list[dict]] = None) -> None:
            super().__init__()
            if layers is None:
                layers = []
            self.layers = [T_NetworkLayer.deserialize(layer) for layer in layers]
            self.check_shape_consistency()

        def serialize(self) -> dict:
            """Serialize the network to a dict.

            Returns
            -------
            dict
                The serialized network.
            """
            return {
                "@class": "NN",
                "@version": 1,
                "layers": [layer.serialize() for layer in self.layers],
            }

        @classmethod
        def deserialize(cls, data: dict) -> "NN":
            """Deserialize the network from a dict.

            Parameters
            ----------
            data : dict
                The dict to deserialize from.
            """
            data = data.copy()
            check_version_compatibility(data.pop("@version", 1), 1, 1)
            data.pop("@class", None)
            return cls(data["layers"])

        def __getitem__(self, key):
            assert isinstance(key, int)
            return self.layers[key]

        def __setitem__(self, key, value) -> None:
            assert isinstance(key, int)
            self.layers[key] = value

        def check_shape_consistency(self) -> None:
            for ii in range(len(self.layers) - 1):
                if self.layers[ii].dim_out() != self.layers[ii + 1].dim_in():
                    raise ValueError(
                        f"the dim of layer {ii} output {self.layers[ii].dim_out} ",
                        f"does not match the dim of layer {ii + 1} ",
                        f"output {self.layers[ii].dim_out}",
                    )

        def call(self, x):
            """Forward pass.

            Parameters
            ----------
            x : np.ndarray
                The input.

            Returns
            -------
            np.ndarray
                The output.
            """
            for layer in self.layers:
                x = layer(x)
            return x

        def clear(self) -> None:
            """Clear the network parameters to zero."""
            for layer in self.layers:
                xp = array_api_compat.array_namespace(layer.w)
                layer.w = xp.zeros_like(layer.w)
                if layer.b is not None:
                    layer.b = xp.zeros_like(layer.b)
                if layer.idt is not None:
                    layer.idt = xp.zeros_like(layer.idt)

    return NN


NativeNet = make_multilayer_network(NativeLayer, NativeOP)


def make_embedding_network(T_Network, T_NetworkLayer):
    class EN(T_Network):
        """The embedding network.

        Parameters
        ----------
        in_dim
            Input dimension.
        neuron
            The number of neurons in each layer. The output dimension
            is the same as the dimension of the last layer.
        activation_function
            The activation function.
        resnet_dt
            Use time step at the resnet architecture.
        precision
            Floating point precision for the model parameters.
        seed : int, optional
            Random seed.
        bias : bool, Optional
            Whether to use bias in the embedding layer.
        """

        def __init__(
            self,
            in_dim,
            neuron: list[int] = [24, 48, 96],
            activation_function: str = "tanh",
            resnet_dt: bool = False,
            precision: str = DEFAULT_PRECISION,
            seed: Optional[Union[int, list[int]]] = None,
            bias: bool = True,
        ) -> None:
            layers = []
            i_in = in_dim
            for idx, ii in enumerate(neuron):
                i_ot = ii
                layers.append(
                    T_NetworkLayer(
                        i_in,
                        i_ot,
                        bias=bias,
                        use_timestep=resnet_dt,
                        activation_function=activation_function,
                        resnet=True,
                        precision=precision,
                        seed=child_seed(seed, idx),
                    ).serialize()
                )
                i_in = i_ot
            super().__init__(layers)
            self.in_dim = in_dim
            self.neuron = neuron
            self.activation_function = activation_function
            self.resnet_dt = resnet_dt
            self.precision = precision
            self.bias = bias

        def serialize(self) -> dict:
            """Serialize the network to a dict.

            Returns
            -------
            dict
                The serialized network.
            """
            return {
                "@class": "EmbeddingNetwork",
                "@version": 2,
                "in_dim": self.in_dim,
                "neuron": self.neuron.copy(),
                "activation_function": self.activation_function,
                "resnet_dt": self.resnet_dt,
                "bias": self.bias,
                # make deterministic
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "layers": [layer.serialize() for layer in self.layers],
            }

        @classmethod
        def deserialize(cls, data: dict) -> "EmbeddingNet":
            """Deserialize the network from a dict.

            Parameters
            ----------
            data : dict
                The dict to deserialize from.
            """
            data = data.copy()
            check_version_compatibility(data.pop("@version", 1), 2, 1)
            data.pop("@class", None)
            layers = data.pop("layers")
            obj = cls(**data)
            super(EN, obj).__init__(layers)
            return obj

    return EN


EmbeddingNet = make_embedding_network(NativeNet, NativeLayer)


def make_fitting_network(T_EmbeddingNet, T_Network, T_NetworkLayer):
    class FN(T_EmbeddingNet):
        """The fitting network. It may be implemented as an embedding
        net connected with a linear output layer.

        Parameters
        ----------
        in_dim
            Input dimension.
        out_dim
            Output dimension
        neuron
            The number of neurons in each hidden layer.
        activation_function
            The activation function.
        resnet_dt
            Use time step at the resnet architecture.
        precision
            Floating point precision for the model parameters.
        bias_out
            The last linear layer has bias.
        seed : int, optional
            Random seed.
        """

        def __init__(
            self,
            in_dim,
            out_dim,
            neuron: list[int] = [24, 48, 96],
            activation_function: str = "tanh",
            resnet_dt: bool = False,
            precision: str = DEFAULT_PRECISION,
            bias_out: bool = True,
            seed: Optional[Union[int, list[int]]] = None,
        ) -> None:
            super().__init__(
                in_dim,
                neuron=neuron,
                activation_function=activation_function,
                resnet_dt=resnet_dt,
                precision=precision,
                seed=seed,
            )
            i_in = neuron[-1] if len(neuron) > 0 else in_dim
            i_ot = out_dim
            self.layers.append(
                T_NetworkLayer(
                    i_in,
                    i_ot,
                    bias=bias_out,
                    use_timestep=False,
                    activation_function=None,
                    resnet=False,
                    precision=precision,
                    seed=child_seed(seed, len(neuron)),
                )
            )
            self.out_dim = out_dim
            self.bias_out = bias_out

        def serialize(self) -> dict:
            """Serialize the network to a dict.

            Returns
            -------
            dict
                The serialized network.
            """
            return {
                "@class": "FittingNetwork",
                "@version": 1,
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "neuron": self.neuron.copy(),
                "activation_function": self.activation_function,
                "resnet_dt": self.resnet_dt,
                "precision": self.precision,
                "bias_out": self.bias_out,
                "layers": [layer.serialize() for layer in self.layers],
            }

        @classmethod
        def deserialize(cls, data: dict) -> "FittingNet":
            """Deserialize the network from a dict.

            Parameters
            ----------
            data : dict
                The dict to deserialize from.
            """
            data = data.copy()
            check_version_compatibility(data.pop("@version", 1), 1, 1)
            data.pop("@class", None)
            layers = data.pop("layers")
            obj = cls(**data)
            T_Network.__init__(obj, layers)
            return obj

    return FN


FittingNet = make_fitting_network(EmbeddingNet, NativeNet, NativeLayer)


class NetworkCollection:
    """A collection of networks for multiple elements.

    The number of dimensions for types might be 0, 1, or 2.
    - 0: embedding or fitting with type embedding, in ()
    - 1: embedding with type_one_side, or fitting, in (type_i)
    - 2: embedding without type_one_side, in (type_i, type_j)

    Parameters
    ----------
    ndim : int
        The number of dimensions.
    network_type : str, optional
        The type of the network.
    networks : dict, optional
        The networks to initialize with.
    """

    # subclass may override this
    NETWORK_TYPE_MAP: ClassVar[dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(
        self,
        ndim: int,
        ntypes: int,
        network_type: str = "network",
        networks: list[Union[NativeNet, dict]] = [],
    ) -> None:
        self.ndim = ndim
        self.ntypes = ntypes
        self.network_type = self.NETWORK_TYPE_MAP[network_type]
        self._networks = [None for ii in range(ntypes**ndim)]
        for ii, network in enumerate(networks):
            self[ii] = network
        if len(networks):
            self.check_completeness()

    def check_completeness(self) -> None:
        """Check whether the collection is complete.

        Raises
        ------
        RuntimeError
            If the collection is incomplete.
        """
        for tt in itertools.product(range(self.ntypes), repeat=self.ndim):
            if self[tuple(tt)] is None:
                raise RuntimeError(f"network for {tt} not found")

    def _convert_key(self, key):
        if isinstance(key, int):
            idx = key
        else:
            if isinstance(key, tuple):
                pass
            elif isinstance(key, str):
                key = tuple([int(tt) for tt in key.split("_")[1:]])
            else:
                raise TypeError(key)
            assert isinstance(key, tuple)
            assert len(key) == self.ndim
            idx = sum([tt * self.ntypes**ii for ii, tt in enumerate(key)])
        return idx

    def __getitem__(self, key):
        return self._networks[self._convert_key(key)]

    def __setitem__(self, key, value) -> None:
        if isinstance(value, self.network_type):
            pass
        elif isinstance(value, dict):
            value = self.network_type.deserialize(value)
        else:
            raise TypeError(value)
        self._networks[self._convert_key(key)] = value

    def serialize(self) -> dict:
        """Serialize the networks to a dict.

        Returns
        -------
        dict
            The serialized networks.
        """
        network_type_map_inv = {v: k for k, v in self.NETWORK_TYPE_MAP.items()}
        network_type_name = network_type_map_inv[self.network_type]
        return {
            "@class": "NetworkCollection",
            "@version": 1,
            "ndim": self.ndim,
            "ntypes": self.ntypes,
            "network_type": network_type_name,
            "networks": [nn.serialize() for nn in self._networks],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NetworkCollection":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        return cls(**data)


def aggregate(
    data: np.ndarray,
    owners: np.ndarray,
    average=True,
    num_owner=None,
):
    """
    Aggregate rows in data by specifying the owners.

    Parameters
    ----------
    data : data tensor to aggregate [n_row, feature_dim]
    owners : specify the owner of each row [n_row, 1]
    average : if True, average the rows, if False, sum the rows.
        Default = True
    num_owner : the number of owners, this is needed if the
        max idx of owner is not presented in owners tensor
        Default = None

    Returns
    -------
    output: [num_owner, feature_dim]
    """
    xp = array_api_compat.array_namespace(data, owners)
    bin_count = xp_bincount(owners)
    bin_count = xp.where(bin_count == 0, xp.ones_like(bin_count), bin_count)

    if num_owner is not None and bin_count.shape[0] != num_owner:
        difference = num_owner - bin_count.shape[0]
        bin_count = xp.concat([bin_count, xp.ones(difference, dtype=bin_count.dtype)])

    output = xp.zeros((bin_count.shape[0], data.shape[1]), dtype=data.dtype)
    output = xp_add_at(output, owners, data)

    if average:
        output = xp.transpose(xp.transpose(output) / bin_count)

    return output


def get_graph_index(
    nlist: np.ndarray,
    nlist_mask: np.ndarray,
    a_nlist_mask: np.ndarray,
    nall: int,
    use_loc_mapping: bool = True,
):
    """
    Get the index mapping for edge graph and angle graph, ready in `aggregate` or `index_select`.

    Parameters
    ----------
    nlist : nf x nloc x nnei
        Neighbor list. (padded neis are set to 0)
    nlist_mask : nf x nloc x nnei
        Masks of the neighbor list. real nei 1 otherwise 0
    a_nlist_mask : nf x nloc x a_nnei
        Masks of the neighbor list for angle. real nei 1 otherwise 0
    nall
        The number of extended atoms.
    use_loc_mapping
        Whether to use local atom index mapping in training or non-parallel inference.
        When True, local indexing and mapping are applied to neighbor lists and embeddings during descriptor computation.

    Returns
    -------
    edge_index : n_edge x 2
        n2e_index : n_edge
            Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
        n_ext2e_index : n_edge
            Broadcast indices from extended node(j) to edge(ij).
    angle_index : n_angle x 3
        n2a_index : n_angle
            Broadcast indices from extended node(j) to angle(ijk).
        eij2a_index : n_angle
            Broadcast indices from extended edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
        eik2a_index : n_angle
            Broadcast indices from extended edge(ik) to angle(ijk).
    """
    xp = array_api_compat.array_namespace(nlist, nlist_mask, a_nlist_mask)

    nf, nloc, nnei = nlist.shape
    _, _, a_nnei = a_nlist_mask.shape

    a_nlist_mask_3d = xp.logical_and(
        a_nlist_mask[:, :, :, None], a_nlist_mask[:, :, None, :]
    )

    n_edge = int(xp.sum(xp.astype(nlist_mask, xp.int32)))

    # following: get n2e_index, n_ext2e_index, n2a_index, eij2a_index, eik2a_index

    # 1. atom graph
    # node(i) to edge(ij) index_select; edge(ij) to node aggregate
    nlist_loc_index = xp.arange(nf * nloc, dtype=nlist.dtype)
    # nf x nloc x nnei
    n2e_index = xp.broadcast_to(
        xp.reshape(nlist_loc_index, (nf, nloc, 1)), (nf, nloc, nnei)
    )
    # n_edge
    n2e_index = n2e_index[xp.astype(nlist_mask, xp.bool)]

    # node_ext(j) to edge(ij) index_select
    frame_shift = xp.arange(nf, dtype=nlist.dtype) * (
        nall if not use_loc_mapping else nloc
    )
    shifted_nlist = nlist + frame_shift[:, xp.newaxis, xp.newaxis]
    # n_edge
    n_ext2e_index = shifted_nlist[xp.astype(nlist_mask, xp.bool)]

    # 2. edge graph
    # node(i) to angle(ijk) index_select
    n2a_index = xp.broadcast_to(
        xp.reshape(nlist_loc_index, (nf, nloc, 1, 1)), (nf, nloc, a_nnei, a_nnei)
    )
    # n_angle
    n2a_index = n2a_index[a_nlist_mask_3d]

    # edge(ij) to angle(ijk) index_select; angle(ijk) to edge(ij) aggregate
    edge_id = xp.arange(n_edge, dtype=nlist.dtype)
    edge_index = xp.zeros((nf, nloc, nnei), dtype=nlist.dtype)
    if array_api_compat.is_jax_array(nlist):
        # JAX doesn't support in-place item assignment
        edge_index = edge_index.at[xp.astype(nlist_mask, xp.bool)].set(edge_id)
    else:
        edge_index[xp.astype(nlist_mask, xp.bool)] = edge_id
    # only cut a_nnei neighbors, to avoid nnei x nnei
    edge_index = edge_index[:, :, :a_nnei]
    edge_index_ij = xp.broadcast_to(
        edge_index[:, :, :, xp.newaxis], (nf, nloc, a_nnei, a_nnei)
    )
    # n_angle
    eij2a_index = edge_index_ij[a_nlist_mask_3d]

    # edge(ik) to angle(ijk) index_select
    edge_index_ik = xp.broadcast_to(
        edge_index[:, :, xp.newaxis, :], (nf, nloc, a_nnei, a_nnei)
    )
    # n_angle
    eik2a_index = edge_index_ik[a_nlist_mask_3d]

    edge_index_result = xp.stack([n2e_index, n_ext2e_index], axis=-1)
    angle_index_result = xp.stack([n2a_index, eij2a_index, eik2a_index], axis=-1)

    return edge_index_result, angle_index_result
