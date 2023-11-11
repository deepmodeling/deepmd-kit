# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native DP model format for multiple backends."""
import json
from typing import (
    Optional,
)

import h5py

try:
    from deepmd_utils._version import version as __version__
except ImportError:
    __version__ = "unknown"


def traverse_model_dict(model_dict: dict, callback: callable):
    """Traverse a model dict and call callback on each variable.

    Parameters
    ----------
    model_dict : dict
        The model dict to traverse.
    callback : callable
        The callback function to call on each variable.
    """
    for kk, vv in model_dict.items():
        if isinstance(vv, dict):
            if kk == "@variables":
                variable_dict = vv.copy()
                for k2, v2 in variable_dict.items():
                    variable_dict[k2] = callback(v2)
                model_dict[kk] = variable_dict
            else:
                traverse_model_dict(vv, callback)


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
        traverse_model_dict(
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
        traverse_model_dict(model_dict, lambda x: f[x][()].copy())
    return model_dict
