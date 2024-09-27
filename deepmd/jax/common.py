# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Union,
    overload,
)

import numpy as np

from deepmd.jax.env import (
    jnp,
)


@overload
def to_jax_array(array: np.ndarray) -> jnp.ndarray: ...


@overload
def to_jax_array(array: None) -> None: ...


def to_jax_array(array: Union[np.ndarray]) -> Union[jnp.ndarray]:
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
