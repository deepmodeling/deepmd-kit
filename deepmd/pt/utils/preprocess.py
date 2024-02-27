# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Union,
)

import torch

from deepmd.pt.utils import (
    env,
)

log = logging.getLogger(__name__)


class Region3D:
    def __init__(self, boxt):
        """Construct a simulation box."""
        boxt = boxt.reshape([3, 3])
        self.boxt = boxt  # convert physical coordinates to internal ones
        self.rec_boxt = torch.linalg.inv(
            self.boxt
        )  # convert internal coordinates to physical ones

        self.volume = torch.linalg.det(self.boxt)  # compute the volume

        # boxt = boxt.permute(1, 0)
        c_yz = torch.cross(boxt[1], boxt[2])
        self._h2yz = self.volume / torch.linalg.norm(c_yz)
        c_zx = torch.cross(boxt[2], boxt[0])
        self._h2zx = self.volume / torch.linalg.norm(c_zx)
        c_xy = torch.cross(boxt[0], boxt[1])
        self._h2xy = self.volume / torch.linalg.norm(c_xy)

    def phys2inter(self, coord):
        """Convert physical coordinates to internal ones."""
        return coord @ self.rec_boxt

    def inter2phys(self, coord):
        """Convert internal coordinates to physical ones."""
        return coord @ self.boxt

    def get_face_distance(self):
        """Return face distinces to each surface of YZ, ZX, XY."""
        return torch.stack([self._h2yz, self._h2zx, self._h2xy])


def normalize_coord(coord, region: Region3D, nloc: int):
    """Move outer atoms into region by mirror.

    Args:
    - coord: shape is [nloc*3]
    """
    tmp_coord = coord.clone()
    inter_cood = torch.remainder(region.phys2inter(tmp_coord), 1.0)
    tmp_coord = region.inter2phys(inter_cood)
    return tmp_coord


def compute_serial_cid(cell_offset, ncell):
    """Tell the sequential cell ID in its 3D space.

    Args:
    - cell_offset: shape is [3]
    - ncell: shape is [3]
    """
    cell_offset[:, 0] *= ncell[1] * ncell[2]
    cell_offset[:, 1] *= ncell[2]
    return cell_offset.sum(-1)


def compute_pbc_shift(cell_offset, ncell):
    """Tell shift count to move the atom into region."""
    shift = torch.zeros_like(cell_offset)
    shift = shift + (cell_offset < 0) * -(
        torch.div(cell_offset, ncell, rounding_mode="floor")
    )
    shift = shift + (cell_offset >= ncell) * -(
        torch.div((cell_offset - ncell), ncell, rounding_mode="floor") + 1
    )
    assert torch.all(cell_offset + shift * ncell >= 0)
    assert torch.all(cell_offset + shift * ncell < ncell)
    return shift


def build_inside_clist(coord, region: Region3D, ncell):
    """Build cell list on atoms inside region.

    Args:
    - coord: shape is [nloc*3]
    - ncell: shape is [3]
    """
    loc_ncell = int(torch.prod(ncell))  # num of local cells
    nloc = coord.numel() // 3  # num of local atoms
    inter_cell_size = 1.0 / ncell

    inter_cood = region.phys2inter(coord.view(-1, 3))
    cell_offset = torch.floor(inter_cood / inter_cell_size).to(torch.long)
    # numerical error brought by conversion from phys to inter back and force
    # may lead to negative value
    cell_offset[cell_offset < 0] = 0
    delta = cell_offset - ncell
    a2c = compute_serial_cid(cell_offset, ncell)  # cell id of atoms
    arange = torch.arange(0, loc_ncell, 1)
    cellid = a2c == arange.unsqueeze(-1)  # one hot cellid
    c2a = cellid.nonzero()
    lst = []
    cnt = 0
    bincount = torch.bincount(a2c, minlength=loc_ncell)
    for i in range(loc_ncell):
        n = bincount[i]
        lst.append(c2a[cnt : cnt + n, 1])
        cnt += n
    return a2c, lst


def append_neighbors(coord, region: Region3D, atype, rcut: float):
    """Make ghost atoms who are valid neighbors.

    Args:
    - coord: shape is [nloc*3]
    - atype: shape is [nloc]
    """
    to_face = region.get_face_distance()

    # compute num and size of local cells
    ncell = torch.floor(to_face / rcut).to(torch.long)
    ncell[ncell == 0] = 1
    cell_size = to_face / ncell
    ngcell = (
        torch.floor(rcut / cell_size).to(torch.long) + 1
    )  # num of cells out of local, which contain ghost atoms

    # add ghost atoms
    a2c, c2a = build_inside_clist(coord, region, ncell)
    xi = torch.arange(-ngcell[0], ncell[0] + ngcell[0], 1)
    yi = torch.arange(-ngcell[1], ncell[1] + ngcell[1], 1)
    zi = torch.arange(-ngcell[2], ncell[2] + ngcell[2], 1)
    xyz = xi.view(-1, 1, 1, 1) * torch.tensor([1, 0, 0], dtype=torch.long)
    xyz = xyz + yi.view(1, -1, 1, 1) * torch.tensor([0, 1, 0], dtype=torch.long)
    xyz = xyz + zi.view(1, 1, -1, 1) * torch.tensor([0, 0, 1], dtype=torch.long)
    xyz = xyz.view(-1, 3)
    mask_a = (xyz >= 0).all(dim=-1)
    mask_b = (xyz < ncell).all(dim=-1)
    mask = ~torch.logical_and(mask_a, mask_b)
    xyz = xyz[mask]  # cell coord
    shift = compute_pbc_shift(xyz, ncell)
    coord_shift = region.inter2phys(shift.to(env.GLOBAL_PT_FLOAT_PRECISION))
    mirrored = shift * ncell + xyz
    cid = compute_serial_cid(mirrored, ncell)

    n_atoms = coord.shape[0]
    aid = [c2a[ci] + i * n_atoms for i, ci in enumerate(cid)]
    aid = torch.cat(aid)
    tmp = torch.div(aid, n_atoms, rounding_mode="trunc")
    aid = aid % n_atoms
    tmp_coord = coord[aid] - coord_shift[tmp]
    tmp_atype = atype[aid]

    # merge local and ghost atoms
    merged_coord = torch.cat([coord, tmp_coord])
    merged_coord_shift = torch.cat([torch.zeros_like(coord), coord_shift[tmp]])
    merged_atype = torch.cat([atype, tmp_atype])
    merged_mapping = torch.cat([torch.arange(atype.numel()), aid])
    return merged_coord_shift, merged_atype, merged_mapping


def build_neighbor_list(
    nloc: int, coord, atype, rcut: float, sec, mapping, type_split=True, min_check=False
):
    """For each atom inside region, build its neighbor list.

    Args:
    - coord: shape is [nall*3]
    - atype: shape is [nall]
    """
    nall = coord.numel() // 3
    coord = coord.float()
    nlist = [[] for _ in range(nloc)]
    coord_l = coord.view(-1, 1, 3)[:nloc]
    coord_r = coord.view(1, -1, 3)
    distance = coord_l - coord_r
    distance = torch.linalg.norm(distance, dim=-1)
    DISTANCE_INF = distance.max().detach() + rcut
    distance[:nloc, :nloc] += torch.eye(nloc, dtype=torch.bool) * DISTANCE_INF
    if min_check:
        if distance.min().abs() < 1e-6:
            RuntimeError("Atom dist too close!")
    if not type_split:
        sec = sec[-1:]
    lst = []
    nlist = torch.zeros((nloc, sec[-1].item())).long() - 1
    nlist_loc = torch.zeros((nloc, sec[-1].item())).long() - 1
    nlist_type = torch.zeros((nloc, sec[-1].item())).long() - 1
    for i, nnei in enumerate(sec):
        if i > 0:
            nnei = nnei - sec[i - 1]
        if not type_split:
            tmp = distance
        else:
            mask = atype.unsqueeze(0) == i
            tmp = distance + (~mask) * DISTANCE_INF
        if tmp.shape[1] >= nnei:
            _sorted, indices = torch.topk(tmp, nnei, dim=1, largest=False)
        else:
            # when nnei > nall
            indices = torch.zeros((nloc, nnei)).long() - 1
            _sorted = torch.ones((nloc, nnei)).long() * DISTANCE_INF
            _sorted_nnei, indices_nnei = torch.topk(
                tmp, tmp.shape[1], dim=1, largest=False
            )
            _sorted[:, : tmp.shape[1]] = _sorted_nnei
            indices[:, : tmp.shape[1]] = indices_nnei
        mask = (_sorted < rcut).to(torch.long)
        indices_loc = mapping[indices]
        indices = indices * mask + -1 * (1 - mask)  # -1 for padding
        indices_loc = indices_loc * mask + -1 * (1 - mask)  # -1 for padding
        if i == 0:
            start = 0
        else:
            start = sec[i - 1]
        end = min(sec[i], start + indices.shape[1])
        nlist[:, start:end] = indices[:, :nnei]
        nlist_loc[:, start:end] = indices_loc[:, :nnei]
        nlist_type[:, start:end] = atype[indices[:, :nnei]] * mask + -1 * (1 - mask)
    return nlist, nlist_loc, nlist_type


def compute_smooth_weight(distance, rmin: float, rmax: float):
    """Compute smooth weight for descriptor elements."""
    min_mask = distance <= rmin
    max_mask = distance >= rmax
    mid_mask = torch.logical_not(torch.logical_or(min_mask, max_mask))
    uu = (distance - rmin) / (rmax - rmin)
    vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1
    return vv * mid_mask + min_mask


def make_env_mat(
    coord,
    atype,
    region,
    rcut: Union[float, list],
    sec,
    pbc=True,
    type_split=True,
    min_check=False,
):
    """Based on atom coordinates, return environment matrix.

    Returns
    -------
    nlist: nlist, [nloc, nnei]
    merged_coord_shift: shift on nall atoms, [nall, 3]
    merged_mapping: mapping from nall index to nloc index, [nall]
    """
    # move outer atoms into cell
    hybrid = isinstance(rcut, list)
    _rcut = rcut
    if hybrid:
        _rcut = max(rcut)
    if pbc:
        merged_coord_shift, merged_atype, merged_mapping = append_neighbors(
            coord, region, atype, _rcut
        )
        merged_coord = coord[merged_mapping] - merged_coord_shift
        if merged_coord.shape[0] <= coord.shape[0]:
            log.warning("No ghost atom is added for system ")
    else:
        merged_coord_shift = torch.zeros_like(coord)
        merged_atype = atype.clone()
        merged_mapping = torch.arange(atype.numel())
        merged_coord = coord.clone()

    # build nlist
    if not hybrid:
        nlist, nlist_loc, nlist_type = build_neighbor_list(
            coord.shape[0],
            merged_coord,
            merged_atype,
            rcut,
            sec,
            merged_mapping,
            type_split=type_split,
            min_check=min_check,
        )
    else:
        nlist, nlist_loc, nlist_type = [], [], []
        for ii, single_rcut in enumerate(rcut):
            nlist_tmp, nlist_loc_tmp, nlist_type_tmp = build_neighbor_list(
                coord.shape[0],
                merged_coord,
                merged_atype,
                single_rcut,
                sec[ii],
                merged_mapping,
                type_split=type_split,
                min_check=min_check,
            )
            nlist.append(nlist_tmp)
            nlist_loc.append(nlist_loc_tmp)
            nlist_type.append(nlist_type_tmp)
    return nlist, nlist_loc, nlist_type, merged_coord_shift, merged_mapping
