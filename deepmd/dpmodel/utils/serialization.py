# SPDX-License-Identifier: LGPL-3.0-or-later
import datetime
import json
from collections.abc import (
    Callable,
)
from copy import (
    deepcopy,
)
from functools import (
    cached_property,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import h5py
import numpy as np
import yaml

try:
    from deepmd._version import version as __version__
except ImportError:
    __version__ = "unknown"


def traverse_model_dict(
    model_obj: Any, callback: Callable, is_variable: bool = False
) -> Any:
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

    def __call__(self) -> int:
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
                lambda x: (
                    f.create_dataset(f"variable_{variable_counter():04d}", data=x).name
                ),
            )
            save_dict = {
                **extra_dict,
                **model_dict,
            }
            f.attrs["json"] = json.dumps(save_dict, separators=(",", ":"))
    elif filename_extension in {".yaml", ".yml"}:
        model_dict = traverse_model_dict(
            model_dict,
            lambda x: (
                {
                    "@class": "np.ndarray",
                    "@is_variable": True,
                    "@version": 1,
                    "dtype": x.dtype.name,
                    "value": x.tolist(),
                }
                if isinstance(x, np.ndarray)
                else x
            ),
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

        def convert_numpy_ndarray(x: Any) -> Any:
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


def format_big_number(x: int) -> str:
    """Format a big number with suffixes.

    Parameters
    ----------
    x : int
        The number to format.

    Returns
    -------
    str
        The formatted string.
    """
    if x >= 1_000_000_000:
        return f"{x / 1_000_000_000:.1f}B"
    elif x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x / 1_000:.1f}K"
    else:
        return str(x)


class Node:
    """A node in a serialization tree.

    Examples
    --------
    >>> model_dict = load_dp_model("model.dp")  # Example filename
    >>> root_node = Node.deserialize(model_dict["model"])
    >>> print(root_node)
    """

    def __init__(
        self,
        name: str,
        children: dict[str, "Node"],
        data: dict[str, Any],
        variables: dict[str, Any],
    ) -> None:
        self.name = name
        self.children: dict[str, Node] = children
        self.data: dict[str, Any] = data
        self.variables: dict[str, Any] = variables

    @cached_property
    def size(self) -> int:
        """Get the size of the node.

        Returns
        -------
        int
            The size of the node.
        """
        total_size = 0

        def count_variables(x: Any) -> Any:
            nonlocal total_size
            if isinstance(x, np.ndarray):
                total_size += x.size
            return x

        traverse_model_dict(
            self.variables,
            count_variables,
            is_variable=True,
        )
        for child in self.children.values():
            total_size += child.size
        return total_size

    @classmethod
    def deserialize(cls, data: Any) -> "Node":
        """Deserialize a Node from a dictionary.

        Parameters
        ----------
        data : Any
            The data to deserialize from.

        Returns
        -------
        Node
            The deserialized node.
        """
        if isinstance(data, dict):
            return cls.from_dict(data)
        elif isinstance(data, list):
            return cls.from_list(data)
        else:
            raise ValueError("Cannot deserialize Node from non-dict/list data.")

    @classmethod
    def from_dict(cls, data_dict: dict) -> "Node":
        """Create a Node from a dictionary.

        Parameters
        ----------
        data_dict : dict
            The dictionary to create the node from.

        Returns
        -------
        Node
            The created node.
        """
        class_name = data_dict.get("@class")
        type_name = data_dict.get("type")
        if class_name is not None:
            if type_name is not None:
                name = f"{class_name} {type_name}"
            else:
                name = class_name
        else:
            name = "Node"
        variables = {}
        children = {}
        data = {}
        for kk, vv in data_dict.items():
            if kk == "@variables":
                variables = deepcopy(vv)
            elif isinstance(vv, dict):
                children[kk] = cls.from_dict(vv)
            elif isinstance(vv, list):
                # drop if no children inside a list
                list_node = cls.from_list(vv)
                if len(list_node.children) > 0:
                    children[kk] = list_node
            else:
                data[kk] = vv
        return cls(name, children, data, variables)

    @classmethod
    def from_list(cls, data_list: list[Any]) -> "Node":
        """Create a Node from a list.

        Parameters
        ----------
        data_list : list
            The list to create the node from.

        Returns
        -------
        Node
            The created node.
        """
        variables = {}
        children = {}
        data = {}
        for ii, vv in enumerate(data_list):
            if isinstance(vv, dict):
                children[f"{ii:d}"] = cls.from_dict(vv)
            elif isinstance(vv, list):
                children[f"{ii:d}"] = cls.from_list(vv)
            else:
                data[f"{ii:d}"] = vv
        return cls("ListNode", children, data, variables)

    def __str__(self) -> str:
        elbow = "└──"
        pipe = "│  "
        tee = "├──"
        blank = "   "
        linebreak = "\n"
        buff = []
        buff.append(f"{self.name} (size={format_big_number(self.size)})")
        children_buff = []
        for ii, (kk, vv) in enumerate(self.children.items()):
            # add indentation
            child_repr = str(vv)
            if len(children_buff) > 0:
                # check if it is the same as the last one
                last_repr = children_buff[-1][1]
                if child_repr == last_repr:
                    # merge
                    last_kk, _ = children_buff[-1]
                    children_buff[-1] = (f"{last_kk}, {kk}", last_repr)
                    continue
            children_buff.append((kk, child_repr))

        def format_list_keys(kk: str) -> str:
            if self.name == "ListNode":
                keys = kk.split(", ")
                if len(keys) > 2:
                    return f"[{keys[0]}...{keys[-1]}]"
            return kk

        def format_value(vv: str, current_index: int) -> str:
            return vv.replace(
                linebreak,
                linebreak + (pipe if current_index < len(children_buff) - 1 else blank),
            )

        buff.extend(
            f"{tee if ii < len(children_buff) - 1 else elbow}{format_list_keys(kk)} -> {format_value(vv, ii)}"
            for ii, (kk, vv) in enumerate(children_buff)
        )
        return "\n".join(buff)
