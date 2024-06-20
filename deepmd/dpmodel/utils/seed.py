# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Union,
    overload,
)


@overload
def child_seed(seed: None, idx: int) -> None: ...


@overload
def child_seed(seed: Union[int, List[int]], idx: int) -> List[int]: ...


def child_seed(seed: Optional[Union[int, List[int]]], idx: int) -> Optional[List[int]]:
    """Generate a child seed from a parent seed.

    Parameters
    ----------
    seed
        The parent seed.
    idx
        The index of the child seed.

    Returns
    -------
    Optional[List[int]]
        The child seed.
    """
    # See https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds
    if seed is None:
        return None
    elif isinstance(seed, int):
        return [idx, seed]
    elif isinstance(seed, list):
        return [idx, *seed]
    else:
        raise TypeError(f"seed must be int or list, not {type(seed)}")
