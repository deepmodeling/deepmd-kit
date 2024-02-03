# SPDX-License-Identifier: LGPL-3.0-or-later
import torch


def phys2inter(
    coord: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Convert physical coordinates to internal(direct) coordinates.

    Parameters
    ----------
    coord : torch.Tensor
        physical coordinates of shape [*, na, 3].
    cell : torch.Tensor
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    inter_coord: torch.Tensor
        the internal coordinates

    """
    rec_cell = torch.linalg.inv(cell)
    return torch.matmul(coord, rec_cell)


def inter2phys(
    coord: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Convert internal(direct) coordinates to physical coordinates.

    Parameters
    ----------
    coord : torch.Tensor
        internal coordinates of shape [*, na, 3].
    cell : torch.Tensor
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    phys_coord: torch.Tensor
        the physical coordinates

    """
    return torch.matmul(coord, cell)


def to_face_distance(
    cell: torch.Tensor,
) -> torch.Tensor:
    """Compute the to-face-distance of the simulation cell.

    Parameters
    ----------
    cell : torch.Tensor
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    dist: torch.Tensor
        the to face distances of shape [*, 3]

    """
    cshape = cell.shape
    dist = b_to_face_distance(cell.view([-1, 3, 3]))
    return dist.view(list(cshape[:-2]) + [3])  # noqa:RUF005


def _to_face_distance(cell):
    volume = torch.linalg.det(cell)
    c_yz = torch.cross(cell[1], cell[2])
    _h2yz = volume / torch.linalg.norm(c_yz)
    c_zx = torch.cross(cell[2], cell[0])
    _h2zx = volume / torch.linalg.norm(c_zx)
    c_xy = torch.cross(cell[0], cell[1])
    _h2xy = volume / torch.linalg.norm(c_xy)
    return torch.stack([_h2yz, _h2zx, _h2xy])


def b_to_face_distance(cell):
    volume = torch.linalg.det(cell)
    c_yz = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    _h2yz = volume / torch.linalg.norm(c_yz, dim=-1)
    c_zx = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    _h2zx = volume / torch.linalg.norm(c_zx, dim=-1)
    c_xy = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    _h2xy = volume / torch.linalg.norm(c_xy, dim=-1)
    return torch.stack([_h2yz, _h2zx, _h2xy], dim=1)


# b_to_face_distance = torch.vmap(
#   _to_face_distance, in_dims=(0), out_dims=(0))


def normalize_coord(
    coord: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Apply PBC according to the atomic coordinates.

    Parameters
    ----------
    coord : torch.Tensor
        orignal coordinates of shape [*, na, 3].

    Returns
    -------
    wrapped_coord: torch.Tensor
        wrapped coordinates of shape [*, na, 3].

    """
    icoord = phys2inter(coord, cell)
    icoord = torch.remainder(icoord, 1.0)
    return inter2phys(icoord, cell)
