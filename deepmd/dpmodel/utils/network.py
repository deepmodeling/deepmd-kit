# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native DP model format for multiple backends.

See issue #2982 for more information.
"""
import copy
import itertools
import json
from datetime import (
    datetime,
)
from typing import (
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)

import h5py
import numpy as np

from deepmd.utils.version import (
    check_version_compatibility,
)

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)


def traverse_model_dict(model_obj, callback: callable, is_variable: bool = False):
    """Traverse a model dict and call callback on each variable.

    Parameters
    ----------
    model_obj : object
        The model object to traverse.
    callback : callable
        The callback function to call on each variable.
    is_variable : bool, optional
        Whether the current node is a variable.

    Returns
    -------
    object
        The model object after traversing.
    """
    if isinstance(model_obj, dict):
        for kk, vv in model_obj.items():
            model_obj[kk] = traverse_model_dict(
                vv, callback, is_variable=is_variable or kk == "@variables"
            )
    elif isinstance(model_obj, list):
        for ii, vv in enumerate(model_obj):
            model_obj[ii] = traverse_model_dict(vv, callback, is_variable=is_variable)
    elif model_obj is None:
        return model_obj
    elif is_variable:
        model_obj = callback(model_obj)
    return model_obj


class Counter:
    """A callable counter.

    Examples
    --------
    >>> counter = Counter()
    >>> counter()
    0
    >>> counter()
    1
    """

    def __init__(self):
        self.count = -1

    def __call__(self):
        self.count += 1
        return self.count


# TODO: should be moved to otherwhere...
def save_dp_model(filename: str, model_dict: dict) -> None:
    """Save a DP model to a file in the native format.

    Parameters
    ----------
    filename : str
        The filename to save to.
    model_dict : dict
        The model dict to save.
    """
    model_dict = model_dict.copy()
    variable_counter = Counter()
    with h5py.File(filename, "w") as f:
        model_dict = traverse_model_dict(
            model_dict,
            lambda x: f.create_dataset(
                f"variable_{variable_counter():04d}", data=x
            ).name,
        )
        save_dict = {
            "software": "deepmd-kit",
            "version": __version__,
            # use UTC+0 time
            "time": str(datetime.utcnow()),
            **model_dict,
        }
        f.attrs["json"] = json.dumps(save_dict, separators=(",", ":"))


def load_dp_model(filename: str) -> dict:
    """Load a DP model from a file in the native format.

    Parameters
    ----------
    filename : str
        The filename to load from.

    Returns
    -------
    dict
        The loaded model dict, including meta information.
    """
    with h5py.File(filename, "r") as f:
        model_dict = json.loads(f.attrs["json"])
        model_dict = traverse_model_dict(model_dict, lambda x: f[x][()].copy())
    return model_dict


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
    ) -> None:
        prec = PRECISION_DICT[precision.lower()]
        self.precision = precision
        # only use_timestep when skip connection is established.
        use_timestep = use_timestep and (num_out == num_in or num_out == num_in * 2)
        rng = np.random.default_rng()
        self.w = rng.normal(size=(num_in, num_out)).astype(prec)
        self.b = rng.normal(size=(num_out,)).astype(prec) if bias else None
        self.idt = rng.normal(size=(num_out,)).astype(prec) if use_timestep else None
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
            "w": self.w,
            "b": self.b,
            "idt": self.idt,
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
        data = copy.deepcopy(data)
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
        obj.w, obj.b, obj.idt = (
            variables["w"],
            variables.get("b", None),
            variables.get("idt", None),
        )
        obj.check_shape_consistency()
        return obj

    def check_shape_consistency(self):
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

    def check_type_consistency(self):
        precision = self.precision

        def check_var(var):
            if var is not None:
                # assertion "float64" == "double" would fail
                assert PRECISION_DICT[var.dtype.name] is PRECISION_DICT[precision]

        check_var(self.w)
        check_var(self.b)
        check_var(self.idt)

    def __setitem__(self, key, value):
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
        if self.activation_function == "tanh":
            fn = np.tanh
        elif self.activation_function.lower() == "none":

            def fn(x):
                return x
        else:
            raise NotImplementedError(self.activation_function)
        y = (
            np.matmul(x, self.w) + self.b
            if self.b is not None
            else np.matmul(x, self.w)
        )
        y = fn(y)
        if self.idt is not None:
            y *= self.idt
        if self.resnet and self.w.shape[1] == self.w.shape[0]:
            y += x
        elif self.resnet and self.w.shape[1] == 2 * self.w.shape[0]:
            y += np.concatenate([x, x], axis=-1)
        return y


def make_multilayer_network(T_NetworkLayer, ModuleBase):
    class NN(ModuleBase):
        """Native representation of a neural network.

        Parameters
        ----------
        layers : list[NativeLayer], optional
            The layers of the network.
        """

        def __init__(self, layers: Optional[List[dict]] = None) -> None:
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

        def __setitem__(self, key, value):
            assert isinstance(key, int)
            self.layers[key] = value

        def check_shape_consistency(self):
            for ii in range(len(self.layers) - 1):
                if self.layers[ii].dim_out() != self.layers[ii + 1].dim_in():
                    raise ValueError(
                        f"the dim of layer {ii} output {self.layers[ii].dim_out} ",
                        f"does not match the dim of layer {ii+1} ",
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

        def clear(self):
            """Clear the network parameters to zero."""
            for layer in self.layers:
                layer.w.fill(0.0)
                if layer.b is not None:
                    layer.b.fill(0.0)
                if layer.idt is not None:
                    layer.idt.fill(0.0)

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
            Floating point precision for the model paramters.

        """

        def __init__(
            self,
            in_dim,
            neuron: List[int] = [24, 48, 96],
            activation_function: str = "tanh",
            resnet_dt: bool = False,
            precision: str = DEFAULT_PRECISION,
        ):
            layers = []
            i_in = in_dim
            for idx, ii in enumerate(neuron):
                i_ot = ii
                layers.append(
                    T_NetworkLayer(
                        i_in,
                        i_ot,
                        bias=True,
                        use_timestep=resnet_dt,
                        activation_function=activation_function,
                        resnet=True,
                        precision=precision,
                    ).serialize()
                )
                i_in = i_ot
            super().__init__(layers)
            self.in_dim = in_dim
            self.neuron = neuron
            self.activation_function = activation_function
            self.resnet_dt = resnet_dt
            self.precision = precision

        def serialize(self) -> dict:
            """Serialize the network to a dict.

            Returns
            -------
            dict
                The serialized network.
            """
            return {
                "@class": "EmbeddingNetwork",
                "@version": 1,
                "in_dim": self.in_dim,
                "neuron": self.neuron.copy(),
                "activation_function": self.activation_function,
                "resnet_dt": self.resnet_dt,
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
            data = copy.deepcopy(data)
            check_version_compatibility(data.pop("@version", 1), 1, 1)
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
            Floating point precision for the model paramters.
        bias_out
            The last linear layer has bias.

        """

        def __init__(
            self,
            in_dim,
            out_dim,
            neuron: List[int] = [24, 48, 96],
            activation_function: str = "tanh",
            resnet_dt: bool = False,
            precision: str = DEFAULT_PRECISION,
            bias_out: bool = True,
        ):
            super().__init__(
                in_dim,
                neuron=neuron,
                activation_function=activation_function,
                resnet_dt=resnet_dt,
                precision=precision,
            )
            i_in, i_ot = neuron[-1], out_dim
            self.layers.append(
                T_NetworkLayer(
                    i_in,
                    i_ot,
                    bias=bias_out,
                    use_timestep=False,
                    activation_function=None,
                    resnet=False,
                    precision=precision,
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
            data = copy.deepcopy(data)
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

    The number of dimesions for types might be 0, 1, or 2.
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
    NETWORK_TYPE_MAP: ClassVar[Dict[str, type]] = {
        "network": NativeNet,
        "embedding_network": EmbeddingNet,
        "fitting_network": FittingNet,
    }

    def __init__(
        self,
        ndim: int,
        ntypes: int,
        network_type: str = "network",
        networks: List[Union[NativeNet, dict]] = [],
    ):
        self.ndim = ndim
        self.ntypes = ntypes
        self.network_type = self.NETWORK_TYPE_MAP[network_type]
        self._networks = [None for ii in range(ntypes**ndim)]
        for ii, network in enumerate(networks):
            self[ii] = network
        if len(networks):
            self.check_completeness()

    def check_completeness(self):
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

    def __setitem__(self, key, value):
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
