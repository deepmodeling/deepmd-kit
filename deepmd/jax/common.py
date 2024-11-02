# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Optional,
    overload,
)

import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.jax.env import (
    jnp,
    nnx,
)


@overload
def to_jax_array(array: np.ndarray) -> jnp.ndarray: ...


@overload
def to_jax_array(array: None) -> None: ...


def to_jax_array(array: Optional[np.ndarray]) -> Optional[jnp.ndarray]:
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


def flax_module(
    module: NativeOP,
) -> nnx.Module:
    """Convert a NativeOP to a Flax module.

    Parameters
    ----------
    module : NativeOP
        The NativeOP to convert.

    Returns
    -------
    flax.nnx.Module
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
        def __call__(self, *args, **kwargs):
            return type(nnx.Module).__call__(self, *args, **kwargs)

    class FlaxModule(module, nnx.Module, metaclass=MixedMetaClass):
        def __init_subclass__(cls, **kwargs) -> None:
            return super().__init_subclass__(**kwargs)

        def __setattr__(self, name: str, value: Any) -> None:
            return super().__setattr__(name, value)

    return FlaxModule


class ArrayAPIVariable(nnx.Variable):
    def __array__(self, *args, **kwargs):
        return self.value.__array__(*args, **kwargs)

    def __array_namespace__(self, *args, **kwargs):
        return self.value.__array_namespace__(*args, **kwargs)

    def __dlpack__(self, *args, **kwargs):
        return self.value.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self, *args, **kwargs):
        return self.value.__dlpack_device__(*args, **kwargs)


def scatter_sum(input, dim, index: jnp.ndarray, src: jnp.ndarray) -> jnp.ndarray:
    """Reduces all values from the src tensor to the indices specified in the index tensor."""
    idx = jnp.arange(input.size, dtype=jnp.int64).reshape(input.shape)
    new_idx = jnp.take_along_axis(idx, index, axis=dim).ravel()
    shape = input.shape
    input = input.ravel()
    input = input.at[new_idx].add(src.ravel())
    return input.reshape(shape)
