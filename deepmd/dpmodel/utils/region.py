# SPDX-License-Identifier: LGPL-3.0-or-later
import array_api_compat
import numpy as np


def phys2inter(
    coord: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Convert physical coordinates to internal(direct) coordinates.

    Parameters
    ----------
    coord : np.ndarray
        physical coordinates of shape [*, na, 3].
    cell : np.ndarray
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    inter_coord: np.ndarray
        the internal coordinates

    """
    xp = array_api_compat.array_namespace(coord, cell)
    rec_cell = xp.linalg.inv(cell)
    return xp.matmul(coord, rec_cell)


def inter2phys(
    coord: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Convert internal(direct) coordinates to physical coordinates.

    Parameters
    ----------
    coord : np.ndarray
        internal coordinates of shape [*, na, 3].
    cell : np.ndarray
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    phys_coord: np.ndarray
        the physical coordinates

    """
    xp = array_api_compat.array_namespace(coord, cell)
    return xp.matmul(coord, cell)


def normalize_coord(
    coord: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Apply PBC according to the atomic coordinates.

    Parameters
    ----------
    coord : np.ndarray
        original coordinates of shape [*, na, 3].
    cell : np.ndarray
        simulation cell shape [*, 3, 3].

    Returns
    -------
    wrapped_coord: np.ndarray
        wrapped coordinates of shape [*, na, 3].

    """
    xp = array_api_compat.array_namespace(coord, cell)
    icoord = phys2inter(coord, cell)
    icoord = xp.remainder(icoord, xp.asarray(1.0))
    return inter2phys(icoord, cell)


def to_face_distance(
    cell: np.ndarray,
) -> np.ndarray:
    """Compute the to-face-distance of the simulation cell.

    Parameters
    ----------
    cell : np.ndarray
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    dist: np.ndarray
        the to face distances of shape [*, 3]

    """
    xp = array_api_compat.array_namespace(cell)
    cshape = cell.shape
    dist = b_to_face_distance(xp.reshape(cell, [-1, 3, 3]))
    return xp.reshape(dist, list(cshape[:-2]) + [3])  # noqa:RUF005


def b_to_face_distance(cell):
    xp = array_api_compat.array_namespace(cell)
    volume = xp.linalg.det(cell)
    c_yz = xp.linalg.cross(cell[:, 1, ...], cell[:, 2, ...], axis=-1)
    _h2yz = volume / xp.linalg.vector_norm(c_yz, axis=-1)
    c_zx = xp.linalg.cross(cell[:, 2, ...], cell[:, 0, ...], axis=-1)
    _h2zx = volume / xp.linalg.vector_norm(c_zx, axis=-1)
    c_xy = xp.linalg.cross(cell[:, 0, ...], cell[:, 1, ...], axis=-1)
    _h2xy = volume / xp.linalg.vector_norm(c_xy, axis=-1)
    return xp.stack([_h2yz, _h2zx, _h2xy], axis=1)
