# SPDX-License-Identifier: LGPL-3.0-or-later

import array_api_strict
import numpy as np


def to_array_api_strict_array(array: np.ndarray | None):
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
    return array_api_strict.asarray(array)
