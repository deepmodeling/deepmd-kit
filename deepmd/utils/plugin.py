# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base of plugin systems."""
# copied from https://github.com/deepmodeling/dpdata/blob/a3e76d75de53f6076254de82d18605a010dc3b00/dpdata/plugin.py

import difflib
from abc import (
    ABCMeta,
)
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)


class Plugin:
    """A class to register and restore plugins.

    Attributes
    ----------
    plugins : dict[str, object]
        plugins

    Examples
    --------
    >>> plugin = Plugin()
    >>> @plugin.register("xx")
        def xxx():
            pass
    >>> print(plugin.plugins["xx"])
    """

    def __init__(self) -> None:
        self.plugins = {}

    def __add__(self, other: "Plugin") -> "Plugin":
        self.plugins.update(other.plugins)
        return self

    def register(
        self, key: str, alias: list[str] | None = None
    ) -> Callable[[object], object]:
        """Register a plugin.

        Parameters
        ----------
        key : str
            Primary key of the plugin.
        alias : list[str], optional
            Alternative keys for the plugin.

        Returns
        -------
        Callable[[object], object]
            decorator
        """

        def decorator(object: object) -> object:
            self.plugins[key] = object
            if alias:
                for alias_key in alias:
                    self.plugins[alias_key] = object
            return object

        return decorator

    def get_plugin(self, key: str) -> object:
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
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Remove `type` and keys that starts with underline."""
        obj = self.__new__(self, *args, **kwargs)
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


def make_plugin_registry(name: str | None = None) -> type[object]:
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
        def register(
            key: str, alias: list[str] | None = None
        ) -> Callable[[object], object]:
            """Register a descriptor plugin.

            Parameters
            ----------
            key : str
                The primary key of the plugin.
            alias : list[str], optional
                Alternative keys for the plugin.

            Returns
            -------
            callable[[object], object]
                the registered descriptor

            Examples
            --------
            >>> @BaseClass.register("some_class")
                class SomeClass(BaseClass):
                    pass
            >>> @BaseClass.register("some_class", alias=["alias1", "alias2"])
                class SomeClass(BaseClass):
                    pass
            """
            return PR.__plugins.register(key, alias=alias)

        @classmethod
        def get_class_by_type(cls, class_type: str) -> type[object]:
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
        def get_plugins(cls) -> dict[str, type[object]]:
            """Get all the registered plugins."""
            return PR.__plugins.plugins

    return PR
