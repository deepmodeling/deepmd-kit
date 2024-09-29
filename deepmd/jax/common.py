# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
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
        pass

    class FlaxModule(module, nnx.Module, metaclass=MixedMetaClass):
        pass

    return FlaxModule
