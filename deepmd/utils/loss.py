# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    cast,
)

import array_api_compat


def symmetrize_virial(value: Any) -> Any:
    """Return the symmetric virial, preserving its shape, dtype and device.

    The input has nine Cartesian components per frame. The projection stays
    differentiable and does not modify the model prediction or its labels.
    """
    xp = array_api_compat.array_namespace(value)
    tensor = xp.reshape(value, (-1, 3, 3))
    return xp.reshape(0.5 * (tensor + xp.matrix_transpose(tensor)), value.shape)


def resolve_huber_deltas(
    huber_delta: float | list[float],
) -> tuple[float, float, float]:
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
