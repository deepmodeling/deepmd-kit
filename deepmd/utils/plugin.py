# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base of plugin systems."""
# copied from https://github.com/deepmodeling/dpdata/blob/a3e76d75de53f6076254de82d18605a010dc3b00/dpdata/plugin.py

import difflib
from abc import (
    ABCMeta,
)
from typing import (
    Callable,
    Dict,
    Optional,
    Type,
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
    >>> print(plugin.plugins["xx"])
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


def make_plugin_registry(name: Optional[str] = None) -> Type[object]:
    """Make a plugin registry.

    Parameters
    ----------
    name : Optional[str]
        the name of the registry for the error message, e.g. descriptor, backend, etc.

    Examples
    --------
    >>> class BaseClass(make_plugin_registry()):
            pass
    """
    if name is None:
        name = "class"

    class PR:
        __plugins = Plugin()

        @staticmethod
        def register(key: str) -> Callable[[object], object]:
            """Register a descriptor plugin.

            Parameters
            ----------
            key : str
                the key of a descriptor

            Returns
            -------
            callable[[object], object]
                the registered descriptor

            Examples
            --------
            >>> @BaseClass.register("some_class")
                class SomeClass(BaseClass):
                    pass
            """
            return PR.__plugins.register(key)

        @classmethod
        def get_class_by_type(cls, class_type: str) -> Type[object]:
            """Get the class by the plugin type."""
            if class_type in PR.__plugins.plugins:
                return PR.__plugins.plugins[class_type]
            else:
                # did you mean
                matches = difflib.get_close_matches(
                    class_type, PR.__plugins.plugins.keys()
                )
                dym_message = f"Did you mean: {matches[0]}?" if matches else ""
                raise RuntimeError(f"Unknown {name} type: {class_type}. {dym_message}")

        @classmethod
        def get_plugins(cls) -> Dict[str, Type[object]]:
            """Get all the registered plugins."""
            return PR.__plugins.plugins

    return PR
