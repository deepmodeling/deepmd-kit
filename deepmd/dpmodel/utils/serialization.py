# SPDX-License-Identifier: LGPL-3.0-or-later
import datetime
import json
from pathlib import (
    Path,
)
from typing import (
    Callable,
)

import h5py
import numpy as np
import yaml

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"


def traverse_model_dict(model_obj, callback: Callable, is_variable: bool = False):
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
        if model_obj.get("@is_variable", False):
            return callback(model_obj)
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

    def __init__(self) -> None:
        self.count = -1

    def __call__(self):
        self.count += 1
        return self.count


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
    filename_extension = Path(filename).suffix
    extra_dict = {
        "software": "deepmd-kit",
        "version": __version__,
        # use UTC+0 time
        "time": str(datetime.datetime.now(tz=datetime.timezone.utc)),
    }
    if filename_extension in (".dp", ".hlo"):
        variable_counter = Counter()
        with h5py.File(filename, "w") as f:
            model_dict = traverse_model_dict(
                model_dict,
                lambda x: f.create_dataset(
                    f"variable_{variable_counter():04d}", data=x
                ).name,
            )
            save_dict = {
                **extra_dict,
                **model_dict,
            }
            f.attrs["json"] = json.dumps(save_dict, separators=(",", ":"))
    elif filename_extension in {".yaml", ".yml"}:
        model_dict = traverse_model_dict(
            model_dict,
            lambda x: {
                "@class": "np.ndarray",
                "@is_variable": True,
                "@version": 1,
                "dtype": x.dtype.name,
                "value": x.tolist(),
            }
            if isinstance(x, np.ndarray)
            else x,
        )
        with open(filename, "w") as f:
            yaml.safe_dump(
                {
                    **extra_dict,
                    **model_dict,
                },
                f,
            )
    else:
        raise ValueError(f"Unknown filename extension: {filename_extension}")


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
    filename_extension = Path(filename).suffix
    if filename_extension in {".dp", ".hlo"}:
        with h5py.File(filename, "r") as f:
            model_dict = json.loads(f.attrs["json"])
            model_dict = traverse_model_dict(model_dict, lambda x: f[x][()].copy())
    elif filename_extension in {".yaml", ".yml"}:

        def convert_numpy_ndarray(x):
            if isinstance(x, dict) and x.get("@class") == "np.ndarray":
                dtype = np.dtype(x["dtype"])
                value = np.asarray(x["value"], dtype=dtype)
                return value
            return x

        with open(filename) as f:
            model_dict = yaml.safe_load(f)
            model_dict = traverse_model_dict(
                model_dict,
                convert_numpy_ndarray,
            )
    else:
        raise ValueError(f"Unknown filename extension: {filename_extension}")
    return model_dict
