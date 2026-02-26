# SPDX-License-Identifier: LGPL-3.0-or-later
from functools import (
    wraps,
)
from typing import (
    Any,
    TypeVar,
    overload,
)

import numpy as np

from deepmd.jax.env import (
    jnp,
    nnx,
)


@overload
def to_jax_array(array: np.ndarray) -> jnp.ndarray: ...


@overload
def to_jax_array(array: None) -> None: ...


def to_jax_array(array: np.ndarray | None) -> jnp.ndarray | None:
    """Convert a numpy array to a JAX array.

    Parameters
    ----------
    array : np.ndarray
        The numpy array to convert.

    Returns
    -------
    jnp.ndarray
        The JAX tensor.
    """
    if array is None:
        return None
    return jnp.array(array)


T = TypeVar("T")


def flax_module(
    module: type[T],
) -> type[T]:  # runtime: actually returns type[T & nnx.Module]
    """Convert a NativeOP to a Flax module.

    Parameters
    ----------
    module : type[NativeOP]
        The NativeOP to convert.

    Returns
    -------
    type[flax.nnx.Module]
        The Flax module.

    Examples
    --------
    >>> @flax_module
    ... class MyModule(NativeOP):
    ...     pass
    """
    metas = set()
    if not issubclass(type(nnx.Module), type(module)):
        metas.add(type(module))
    if not issubclass(type(module), type(nnx.Module)):
        metas.add(type(nnx.Module))

    class MixedMetaClass(*metas):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return type(nnx.Module).__call__(self, *args, **kwargs)

    @wraps(module, updated=())
    class FlaxModule(module, nnx.Module, metaclass=MixedMetaClass):
        def __init_subclass__(cls, **kwargs: Any) -> None:
            return super().__init_subclass__(**kwargs)

        def __setattr__(self, name: str, value: Any) -> None:
            return super().__setattr__(name, value)

    return FlaxModule


class ArrayAPIVariable(nnx.Variable):
    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.value.__array__(*args, **kwargs)

    def __array_namespace__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__array_namespace__(*args, **kwargs)

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self, *args: Any, **kwargs: Any) -> Any:
        return self.value.__dlpack_device__(*args, **kwargs)
