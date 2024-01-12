# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base of plugin systems."""
# copied from https://github.com/deepmodeling/dpdata/blob/a3e76d75de53f6076254de82d18605a010dc3b00/dpdata/plugin.py

from abc import (
    ABCMeta,
)
from typing import (
    Callable,
)


class Plugin:
    """A class to register and restore plugins.

    Attributes
    ----------
    plugins : Dict[str, object]
        plugins

    Examples
    --------
    >>> plugin = Plugin()
    >>> @plugin.register("xx")
        def xxx():
            pass
    >>> print(plugin.plugins['xx'])
    """

    def __init__(self):
        self.plugins = {}

    def __add__(self, other) -> "Plugin":
        self.plugins.update(other.plugins)
        return self

    def register(self, key: str) -> Callable[[object], object]:
        """Register a plugin.

        Parameters
        ----------
        key : str
            key of the plugin

        Returns
        -------
        Callable[[object], object]
            decorator
        """

        def decorator(object: object) -> object:
            self.plugins[key] = object
            return object

        return decorator

    def get_plugin(self, key) -> object:
        """Visit a plugin by key.

        Parameters
        ----------
        key : str
            key of the plugin

        Returns
        -------
        object
            the plugin
        """
        return self.plugins[key]


class VariantMeta:
    def __call__(cls, *args, **kwargs):
        """Remove `type` and keys that starts with underline."""
        obj = cls.__new__(cls, *args, **kwargs)
        kwargs.pop("type", None)
        to_pop = []
        for kk in kwargs:
            if kk[0] == "_":
                to_pop.append(kk)
        for kk in to_pop:
            kwargs.pop(kk, None)
        obj.__init__(*args, **kwargs)
        return obj


class VariantABCMeta(VariantMeta, ABCMeta):
    pass


class PluginVariant(metaclass=VariantABCMeta):
    """A class to remove `type` from input arguments."""

    pass
