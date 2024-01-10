# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native DP model format for multiple backends.

See issue #2982 for more information.
"""
import itertools
import json
from typing import (
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)

import h5py
import numpy as np

try:
    from deepmd_utils._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .common import (
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


def save_dp_model(filename: str, model_dict: dict, extra_info: Optional[dict] = None):
    """Save a DP model to a file in the native format.

    Parameters
    ----------
    filename : str
        The filename to save to.
    model_dict : dict
        The model dict to save.
    extra_info : dict, optional
        Extra meta information to save.
    """
    model_dict = model_dict.copy()
    variable_counter = Counter()
    if extra_info is not None:
        extra_info = extra_info.copy()
    else:
        extra_info = {}
    with h5py.File(filename, "w") as f:
        model_dict = traverse_model_dict(
            model_dict,
            lambda x: f.create_dataset(
                f"variable_{variable_counter():04d}", data=x
            ).name,
        )
        save_dict = {
            "model": model_dict,
            "software": "deepmd-kit",
            "version": __version__,
            **extra_info,
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
        w: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        idt: Optional[np.ndarray] = None,
        activation_function: Optional[str] = None,
        resnet: bool = False,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        prec = PRECISION_DICT[precision.lower()]
        self.precision = precision
        self.w = w.astype(prec) if w is not None else None
        self.b = b.astype(prec) if b is not None else None
        self.idt = idt.astype(prec) if idt is not None else None
        self.activation_function = activation_function
        self.resnet = resnet
        self.check_type_consistency()

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
        }
        if self.idt is not None:
            data["idt"] = self.idt
        return {
            "activation_function": self.activation_function,
            "resnet": self.resnet,
            "precision": self.precision,
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
        precision = data.get("precision", DEFAULT_PRECISION)
        return cls(
            w=data["@variables"]["w"],
            b=data["@variables"].get("b", None),
            idt=data["@variables"].get("idt", None),
            activation_function=data["activation_function"],
            resnet=data.get("resnet", False),
            precision=precision,
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


class NativeNet(NativeOP):
    """Native representation of a neural network.

    Parameters
    ----------
    layers : list[NativeLayer], optional
        The layers of the network.
    """

    def __init__(self, layers: Optional[List[dict]] = None) -> None:
        if layers is None:
            layers = []
        self.layers = [NativeLayer.deserialize(layer) for layer in layers]

    def serialize(self) -> dict:
        """Serialize the network to a dict.

        Returns
        -------
        dict
            The serialized network.
        """
        return {"layers": [layer.serialize() for layer in self.layers]}

    @classmethod
    def deserialize(cls, data: dict) -> "NativeNet":
        """Deserialize the network from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        return cls(data["layers"])

    def __getitem__(self, key):
        assert isinstance(key, int)
        if len(self.layers) <= key:
            self.layers.extend([NativeLayer()] * (key - len(self.layers) + 1))
        return self.layers[key]

    def __setitem__(self, key, value):
        assert isinstance(key, int)
        if len(self.layers) <= key:
            self.layers.extend([NativeLayer()] * (key - len(self.layers) + 1))
        self.layers[key] = value

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
        for layer in self.layers:
            x = layer.call(x)
        return x


class EmbeddingNet(NativeNet):
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
        rng = np.random.default_rng()
        for idx, ii in enumerate(neuron):
            i_ot = ii
            layers.append(
                NativeLayer(
                    rng.normal(size=(i_in, i_ot)),
                    b=rng.normal(size=(ii)),
                    idt=rng.normal(size=(ii)) if resnet_dt else None,
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
            "in_dim": self.in_dim,
            "neuron": self.neuron.copy(),
            "activation_function": self.activation_function,
            "resnet_dt": self.resnet_dt,
            "precision": self.precision,
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
        layers = data.pop("layers")
        obj = cls(**data)
        super(EmbeddingNet, obj).__init__(layers)
        return obj


class NetworkCollection:
    """A collection of networks for multiple elements.

    The number of dimesions for types might be 0, 1, or 2.
    - 0: embedding or fitting with type embedding, in ()
    - 1: embedding with type_one_side, or fitting, in (type_i)
    - 2: embedding without type_one_side, in (type_i, type_j)

    Serialized keys are in the form of type, type_i, type_i_j, ...

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
    }

    def __init__(
        self,
        ndim: int,
        ntypes: int,
        network_type: str = "network",
        networks: Dict[Union[str, tuple], Union[NativeNet, dict]] = {},
    ):
        self.ndim = ndim
        self.ntypes = ntypes
        self.network_type = self.NETWORK_TYPE_MAP[network_type]
        self._networks = {}
        for kk, vv in networks.items():
            self[kk] = vv

    def check_completeness(self):
        """Check whether the collection is complete.

        Raises
        ------
        RuntimeError
            If the collection is incomplete.
        """
        for tt in itertools.product(range(self.ntypes), repeat=self.ndim):
            if tuple(tt) not in self._networks:
                raise RuntimeError(f"network for {tt} not found")

    def _convert_key(self, key):
        if isinstance(key, tuple):
            pass
        elif isinstance(key, str):
            key = tuple([int(tt) for tt in key.split("_")[1:]])
        else:
            raise TypeError(key)
        assert isinstance(key, tuple)
        assert len(key) == self.ndim
        return key

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
        network_type_map_inv = inv_map = {
            v: k for k, v in self.NETWORK_TYPE_MAP.items()
        }
        network_type_name = network_type_map_inv[self.network_type]
        return {
            "ndim": self.ndim,
            "ntypes": self.ntypes,
            "network_type": network_type_name,
            "networks": {
                ("_".join(["type"] + [str(tt) for tt in key])): value.serialize()
                for key, value in self._networks.items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NetworkCollection":
        """Deserialize the networks from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        return cls(**data)
