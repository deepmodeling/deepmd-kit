# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    cast,
)


def resolve_huber_deltas(huber_delta: float | list[float]) -> tuple[float, float, float]:
    """Resolve Huber delta config into energy, force, and virial deltas.

    Parameters
    ----------
    huber_delta : float | list[float]
        Shared delta or three deltas ordered as [energy, force, virial].

    Returns
    -------
    tuple[float, float, float]
        Deltas for energy, force, and virial.

    Raises
    ------
    ValueError
        If `huber_delta` is a list with a length other than 3.
    """
    if isinstance(huber_delta, list):
        if len(huber_delta) != 3:
            raise ValueError(
                "huber_delta must be a float or a list of three values for energy, force and virial."
            )
        return cast(
            "tuple[float, float, float]",
            (huber_delta[0], huber_delta[1], huber_delta[2]),
        )
    return huber_delta, huber_delta, huber_delta
