# SPDX-License-Identifier: LGPL-3.0-or-later
"""Degree-wise real Cartesian geometry for DPA4C."""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
)

import array_api_compat
import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )

MAX_ANGULAR_DEGREE = 4
SUPPORTED_CHANNELS = (8, 16, 32, 64, 128)
SUPPORTED_LMAX = (2, 3, 4)


def derive_degree_channels(channels: object, lmax: object) -> list[int]:
    """Derive the fixed degree-wise channel profile.

    The non-scalar widths follow the scalar width sublinearly: each
    non-scalar degree costs ``2 * l + 1`` moment accumulators per channel and
    a Gram quadratic in its width, so a linear profile would let the angular
    blocks dominate both the reduction payload and the output width. Degree
    one takes the geometric mean of the scalar width and the floor, degree two
    takes half of that, and degrees three and above keep a single channel.

    Parameters
    ----------
    channels
        Scalar degree-zero channel width.
    lmax
        Maximum angular degree.

    Returns
    -------
    list[int]
        Channel widths for degrees zero through ``lmax``.

    Raises
    ------
    TypeError
        If either argument is not an integer.
    ValueError
        If ``channels`` or ``lmax`` is outside the compiled profile set.
    """
    if not isinstance(channels, int) or isinstance(channels, bool):
        raise TypeError(f"`channels` must be an integer, got {type(channels).__name__}")
    if channels not in SUPPORTED_CHANNELS:
        raise ValueError(
            f"`channels` must be one of {SUPPORTED_CHANNELS}, got {channels}"
        )
    if not isinstance(lmax, int) or isinstance(lmax, bool):
        raise TypeError(f"`lmax` must be an integer, got {type(lmax).__name__}")
    if lmax not in SUPPORTED_LMAX:
        raise ValueError(f"`lmax` must be one of {SUPPORTED_LMAX}, got {lmax}")

    # `channels` is a power of two, so the geometric mean is an exact shift.
    exponent = channels.bit_length() - 1
    degree_one = max(4, 1 << ((exponent + 1) // 2))
    degree_two = max(4, degree_one >> 1)
    return [channels, degree_one, degree_two] + [1] * (lmax - 2)


def build_angular_basis(direction: Array, lmax: int) -> Array:
    r"""Build normalized real Cartesian harmonics through degree four.

    For vectors :math:`u` and :math:`v`, each degree block obeys

    .. math::

       B_\ell(u)\cdot B_\ell(v)
       =(\lVert u\rVert\lVert v\rVert)^\ell
        P_\ell\left(
        \frac{u\cdot v}{\lVert u\rVert\lVert v\rVert}\right).

    Parameters
    ----------
    direction
        Regularized edge directions with shape ``(E, 3)``.
    lmax
        Maximum angular degree. Supported values are zero through four.

    Returns
    -------
    Array
        Concatenated degree blocks with shape ``(E, (lmax + 1) ** 2)``.

    Raises
    ------
    ValueError
        If ``lmax`` is outside the supported range.
    """
    if lmax < 0 or lmax > MAX_ANGULAR_DEGREE:
        raise ValueError(
            f"`lmax` must be between 0 and {MAX_ANGULAR_DEGREE}, got {lmax}"
        )
    xp = array_api_compat.array_namespace(direction)
    x, y, z = direction[:, 0], direction[:, 1], direction[:, 2]
    squared_norm = x * x + y * y + z * z
    blocks = [xp.ones_like(x)[:, None]]
    if lmax >= 1:
        blocks.append(xp.stack([x, y, z], axis=-1))
    if lmax >= 2:
        sqrt_three = math.sqrt(3.0)
        blocks.append(
            xp.stack(
                [
                    sqrt_three * x * y,
                    sqrt_three * y * z,
                    0.5 * (3.0 * z * z - squared_norm),
                    sqrt_three * x * z,
                    0.5 * sqrt_three * (x * x - y * y),
                ],
                axis=-1,
            )
        )
    if lmax >= 3:
        blocks.append(
            xp.stack(
                [
                    math.sqrt(5.0 / 8.0) * y * (3.0 * x * x - y * y),
                    math.sqrt(15.0) * x * y * z,
                    math.sqrt(3.0 / 8.0) * y * (5.0 * z * z - squared_norm),
                    0.5 * z * (5.0 * z * z - 3.0 * squared_norm),
                    math.sqrt(3.0 / 8.0) * x * (5.0 * z * z - squared_norm),
                    0.5 * math.sqrt(15.0) * z * (x * x - y * y),
                    math.sqrt(5.0 / 8.0) * x * (x * x - 3.0 * y * y),
                ],
                axis=-1,
            )
        )
    if lmax >= 4:
        z_squared = z * z
        x2_minus_y2 = x * x - y * y
        blocks.append(
            xp.stack(
                [
                    0.5 * math.sqrt(35.0) * x * y * x2_minus_y2,
                    0.25 * math.sqrt(70.0) * y * z * (3.0 * x * x - y * y),
                    0.5 * math.sqrt(5.0) * x * y * (7.0 * z_squared - squared_norm),
                    0.25
                    * math.sqrt(10.0)
                    * y
                    * z
                    * (7.0 * z_squared - 3.0 * squared_norm),
                    0.125
                    * (
                        35.0 * z_squared * z_squared
                        - 30.0 * z_squared * squared_norm
                        + 3.0 * squared_norm * squared_norm
                    ),
                    0.25
                    * math.sqrt(10.0)
                    * x
                    * z
                    * (7.0 * z_squared - 3.0 * squared_norm),
                    0.25
                    * math.sqrt(5.0)
                    * x2_minus_y2
                    * (7.0 * z_squared - squared_norm),
                    0.25 * math.sqrt(70.0) * x * z * (x * x - 3.0 * y * y),
                    0.125 * math.sqrt(35.0) * (x**4 - 6.0 * x * x * y * y + y**4),
                ],
                axis=-1,
            )
        )
    return xp.concat(blocks, axis=-1)


def packed_l2_to_stf(packed: Array) -> Array:
    r"""Convert normalized degree-two coefficients to STF matrices.

    The conversion preserves

    .. math::

       p\cdot q = \operatorname{STF}(p):\operatorname{STF}(q).

    Parameters
    ----------
    packed
        Degree-two coefficients with shape ``(..., 5)``.

    Returns
    -------
    Array
        Symmetric-traceless matrices with shape ``(..., 3, 3)``.
    """
    xp = array_api_compat.array_namespace(packed)
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    inv_sqrt_six = 1.0 / math.sqrt(6.0)
    q0, q1, q2, q3, q4 = (packed[..., index] for index in range(5))
    qxy = q0 * inv_sqrt_two
    qyz = q1 * inv_sqrt_two
    qxz = q3 * inv_sqrt_two
    qxx = -q2 * inv_sqrt_six + q4 * inv_sqrt_two
    qyy = -q2 * inv_sqrt_six - q4 * inv_sqrt_two
    qzz = 2.0 * q2 * inv_sqrt_six
    return xp.stack(
        [
            xp.stack([qxx, qxy, qxz], axis=-1),
            xp.stack([qxy, qyy, qyz], axis=-1),
            xp.stack([qxz, qyz, qzz], axis=-1),
        ],
        axis=-2,
    )


def degree_offsets(degree_channels: Sequence[int]) -> tuple[int, ...]:
    """Return offsets for the flat degree-wise moment representation.

    Parameters
    ----------
    degree_channels
        Channel widths for degrees zero through ``lmax``.

    Returns
    -------
    tuple[int, ...]
        Cumulative offsets with length ``len(degree_channels) + 1``.
    """
    offsets = [0]
    for degree, width in enumerate(degree_channels):
        offsets.append(offsets[-1] + (2 * degree + 1) * int(width))
    return tuple(offsets)


def build_moment_indices(
    degree_channels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build channel and harmonic indices for one flat edge payload.

    Every degree reads the leading channels of the shared radial map, so the
    channel index of a degree is simply ``range(degree_channels[degree])``.

    Parameters
    ----------
    degree_channels
        Number of channels for degrees zero through ``lmax``.

    Returns
    -------
    channel_index
        Edge-amplitude channel indices with shape ``(S,)``.
    harmonic_index
        Packed harmonic indices with shape ``(S,)``, where
        ``S = sum((2 * l + 1) * degree_channels[l])``.
    """
    channel_index = []
    harmonic_index = []
    for degree, width in enumerate(degree_channels):
        for component in range(2 * degree + 1):
            channel_index.extend(range(int(width)))
            harmonic_index.extend([degree * degree + component] * int(width))
    return (
        np.asarray(channel_index, dtype=np.int64),
        np.asarray(harmonic_index, dtype=np.int64),
    )
