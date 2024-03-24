# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Tuple,
    Union,
)

import numpy as np

_RANDOM_GENERATOR = np.random.RandomState()


def choice(
    a: Union[np.ndarray, int],
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    replace: bool = True,
    p: Optional[np.ndarray] = None,
):
    """Generates a random sample from a given 1-D array.

    Parameters
    ----------
    a : 1-D array-like or int
        If an ndarray, a random sample is generated from its elements. If an int,
        the random sample is generated as if it were np.arange(a)
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples
        are drawn. Default is None, in which case a single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement. Default is True, meaning
        that a value of a can be selected multiple times.
    p : 1-D array-like, optional
        The probabilities associated with each entry in a. If not given, the sample
        assumes a uniform distribution over all entries in a.

    Returns
    -------
    np.ndarray
        arrays with results and their shapes
    """
    return _RANDOM_GENERATOR.choice(a, size=size, replace=replace, p=p)


def random(size=None):
    """Return random floats in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    size
        Output shape.

    Returns
    -------
    np.ndarray
        Arrays with results and their shapes.
    """
    return _RANDOM_GENERATOR.random_sample(size)


def seed(val: Optional[int] = None):
    """Seed the generator.

    Parameters
    ----------
    val : int
        Seed.
    """
    _RANDOM_GENERATOR.seed(val)


def shuffle(x: np.ndarray):
    """Modify a sequence in-place by shuffling its contents.

    Parameters
    ----------
    x : np.ndarray
        The array or list to be shuffled.
    """
    _RANDOM_GENERATOR.shuffle(x)


__all__ = ["choice", "random", "seed", "shuffle"]
