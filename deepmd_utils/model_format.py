# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native DP model format for multiple backends."""
import json
from typing import (
    List,
    Optional,
)

import h5py
import numpy as np

try:
    from deepmd_utils._version import version as __version__
except ImportError:
    __version__ = "unknown"


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
        print(model_dict)
        model_dict = traverse_model_dict(model_dict, lambda x: f[x][()].copy())
    return model_dict


class NativeLayer:
    """Native representation of a layer.

    Parameters
    ----------
    w : np.ndarray, optional
        The weights of the layer.
    b : np.ndarray, optional
        The biases of the layer.
    idt : np.ndarray, optional
        The identity matrix of the layer.
    """

    def __init__(
        self,
        w: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        idt: Optional[np.ndarray] = None,
    ) -> None:
        self.w = w
        self.b = b
        self.idt = idt

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
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "NativeLayer":
        """Deserialize the layer from a dict.

        Parameters
        ----------
        data : dict
            The dict to deserialize from.
        """
        return cls(data["w"], data["b"], data.get("idt", None))

    def __setitem__(self, key, value):
        if key in ("w", "matrix"):
            self.w = value
        elif key in ("b", "bias"):
            self.b = value
        elif key == "idt":
            self.idt = value
        else:
            raise KeyError(key)


class NativeNet:
    """Native representation of a neural network.

    Parameters
    ----------
    layers : list[NativeLayer], optional
        The layers of the network.
    """

    def __init__(self, layers: Optional[List[NativeLayer]] = None) -> None:
        if layers is None:
            layers = []
        self.layers = layers

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
        return cls([NativeLayer.deserialize(layer) for layer in data["layers"]])

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
