# SPDX-License-Identifier: LGPL-3.0-or-later
import paddle


def phys2inter(
    coord: paddle.Tensor,
    cell: paddle.Tensor,
) -> paddle.Tensor:
    """Convert physical coordinates to internal(direct) coordinates.

    Parameters
    ----------
    coord : paddle.Tensor
        physical coordinates of shape [*, na, 3].
    cell : paddle.Tensor
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    inter_coord: paddle.Tensor
        the internal coordinates

    """
    if paddle.in_dynamic_mode():
        try:
            rec_cell = paddle.linalg.inv(cell)
        except Exception as e:
            rec_cell = paddle.full_like(cell, float("nan"))
            rec_cell.stop_gradient = cell.stop_gradient
    else:
        rec_cell = paddle.linalg.inv(cell)
    return paddle.matmul(coord, rec_cell)


def inter2phys(
    coord: paddle.Tensor,
    cell: paddle.Tensor,
) -> paddle.Tensor:
    """Convert internal(direct) coordinates to physical coordinates.

    Parameters
    ----------
    coord : paddle.Tensor
        internal coordinates of shape [*, na, 3].
    cell : paddle.Tensor
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    phys_coord: paddle.Tensor
        the physical coordinates

    """
    return paddle.matmul(coord, cell)


def to_face_distance(
    cell: paddle.Tensor,
) -> paddle.Tensor:
    """Compute the to-face-distance of the simulation cell.

    Parameters
    ----------
    cell : paddle.Tensor
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    dist: paddle.Tensor
        the to face distances of shape [*, 3]

    """
    cshape = cell.shape
    dist = b_to_face_distance(cell.reshape([-1, 3, 3]))
    return dist.reshape(list(cshape[:-2]) + [3])  # noqa:RUF005


def b_to_face_distance(cell):
    volume = paddle.linalg.det(cell)
    c_yz = paddle.cross(cell[:, 1], cell[:, 2], axis=-1)
    _h2yz = volume / paddle.linalg.norm(c_yz, axis=-1)
    c_zx = paddle.cross(cell[:, 2], cell[:, 0], axis=-1)
    _h2zx = volume / paddle.linalg.norm(c_zx, axis=-1)
    c_xy = paddle.cross(cell[:, 0], cell[:, 1], axis=-1)
    _h2xy = volume / paddle.linalg.norm(c_xy, axis=-1)
    return paddle.stack([_h2yz, _h2zx, _h2xy], axis=1)


# b_to_face_distance = paddle.vmap(
#   _to_face_distance, in_dims=(0), out_dims=(0))


def normalize_coord(
    coord: paddle.Tensor,
    cell: paddle.Tensor,
) -> paddle.Tensor:
    """Apply PBC according to the atomic coordinates.

    Parameters
    ----------
    coord : paddle.Tensor
        original coordinates of shape [*, na, 3].

    Returns
    -------
    wrapped_coord: paddle.Tensor
        wrapped coordinates of shape [*, na, 3].

    """
    icoord = phys2inter(coord, cell)
    icoord = paddle.remainder(icoord, paddle.full([], 1.0, dtype=icoord.dtype))
    return inter2phys(icoord, cell)
