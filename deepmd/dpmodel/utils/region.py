# SPDX-License-Identifier: LGPL-3.0-or-later
import array_api_compat

from deepmd.dpmodel.common import (
    ArrayLike,
)


def phys2inter(
    coord: ArrayLike,
    cell: ArrayLike,
) -> ArrayLike:
    """Convert physical coordinates to internal(direct) coordinates.

    Parameters
    ----------
    coord : ArrayLike
        physical coordinates of shape [*, na, 3].
    cell : ArrayLike
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    inter_coord: ArrayLike
        the internal coordinates

    """
    xp = array_api_compat.array_namespace(coord, cell)
    rec_cell = xp.linalg.inv(cell)
    return xp.matmul(coord, rec_cell)


def inter2phys(
    coord: ArrayLike,
    cell: ArrayLike,
) -> ArrayLike:
    """Convert internal(direct) coordinates to physical coordinates.

    Parameters
    ----------
    coord : ArrayLike
        internal coordinates of shape [*, na, 3].
    cell : ArrayLike
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    phys_coord: ArrayLike
        the physical coordinates

    """
    xp = array_api_compat.array_namespace(coord, cell)
    return xp.matmul(coord, cell)


def normalize_coord(
    coord: ArrayLike,
    cell: ArrayLike,
) -> ArrayLike:
    """Apply PBC according to the atomic coordinates.

    Parameters
    ----------
    coord : ArrayLike
        original coordinates of shape [*, na, 3].
    cell : ArrayLike
        simulation cell shape [*, 3, 3].

    Returns
    -------
    wrapped_coord: ArrayLike
        wrapped coordinates of shape [*, na, 3].

    """
    xp = array_api_compat.array_namespace(coord, cell)
    icoord = phys2inter(coord, cell)
    icoord = xp.remainder(icoord, xp.asarray(1.0))
    return inter2phys(icoord, cell)


def to_face_distance(
    cell: ArrayLike,
) -> ArrayLike:
    """Compute the to-face-distance of the simulation cell.

    Parameters
    ----------
    cell : ArrayLike
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    dist: ArrayLike
        the to face distances of shape [*, 3]

    """
    xp = array_api_compat.array_namespace(cell)
    cshape = cell.shape
    dist = b_to_face_distance(xp.reshape(cell, (-1, 3, 3)))
    return xp.reshape(dist, tuple(list(cshape[:-2]) + [3]))  # noqa:RUF005


def b_to_face_distance(cell: ArrayLike) -> ArrayLike:
    xp = array_api_compat.array_namespace(cell)
    volume = xp.linalg.det(cell)
    c_yz = xp.linalg.cross(cell[:, 1, ...], cell[:, 2, ...], axis=-1)
    _h2yz = volume / xp.linalg.vector_norm(c_yz, axis=-1)
    c_zx = xp.linalg.cross(cell[:, 2, ...], cell[:, 0, ...], axis=-1)
    _h2zx = volume / xp.linalg.vector_norm(c_zx, axis=-1)
    c_xy = xp.linalg.cross(cell[:, 0, ...], cell[:, 1, ...], axis=-1)
    _h2xy = volume / xp.linalg.vector_norm(c_xy, axis=-1)
    return xp.stack([_h2yz, _h2zx, _h2xy], axis=1)
