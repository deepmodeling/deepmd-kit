# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import paddle

from deepmd.pd.utils import (
    decomp,
    env,
)
from deepmd.pd.utils.region import (
    normalize_coord,
    to_face_distance,
)


def extend_input_and_build_neighbor_list(
    coord,
    atype,
    rcut: float,
    sel: list[int],
    mixed_types: bool = False,
    box: Optional[paddle.Tensor] = None,
):
    nframes, nloc = atype.shape[:2]
    if box is not None:
        box_gpu = box
        coord_normalized = normalize_coord(
            coord.reshape([nframes, nloc, 3]),
            box_gpu.reshape([nframes, 3, 3]),
        )
    else:
        box_gpu = None
        coord_normalized = coord.clone()
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, box_gpu, rcut, box
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=(not mixed_types),
    )
    extended_coord = extended_coord.reshape([nframes, -1, 3])
    return extended_coord, extended_atype, mapping, nlist


def build_neighbor_list(
    coord: paddle.Tensor,
    atype: paddle.Tensor,
    nloc: int,
    rcut: float,
    sel: Union[int, list[int]],
    distinguish_types: bool = True,
) -> paddle.Tensor:
    """Build neighbor list for a single frame. keeps nsel neighbors.

    Parameters
    ----------
    coord : paddle.Tensor
        exptended coordinates of shape [batch_size, nall x 3]
    atype : paddle.Tensor
        extended atomic types of shape [batch_size, nall]
        if type < 0 the atom is treat as virtual atoms.
    nloc : int
        number of local atoms.
    rcut : float
        cut-off radius
    sel : int or list[int]
        maximal number of neighbors (of each type).
        if distinguish_types==True, nsel should be list and
        the length of nsel should be equal to number of
        types.
    distinguish_types : bool
        distinguish different types.

    Returns
    -------
    neighbor_list : paddle.Tensor
        Neighbor list of shape [batch_size, nloc, nsel], the neighbors
        are stored in an ascending order. If the number of
        neighbors is less than nsel, the positions are masked
        with -1. The neighbor list of an atom looks like
        |------ nsel ------|
        xx xx xx xx -1 -1 -1
        if distinguish_types==True and we have two types
        |---- nsel[0] -----| |---- nsel[1] -----|
        xx xx xx xx -1 -1 -1 xx xx xx -1 -1 -1 -1
        For virtual atoms all neighboring positions are filled with -1.

    """
    batch_size = coord.shape[0]
    coord = coord.reshape([batch_size, -1])
    nall = coord.shape[1] // 3
    # fill virtual atoms with large coords so they are not neighbors of any
    # real atom.

    # NOTE: control flow with double backward is not supported well yet by paddle.jit
    if not paddle.in_dynamic_mode() or decomp.numel(coord) > 0:
        xmax = paddle.max(coord) + 2.0 * rcut
    else:
        xmax = paddle.zeros([], dtype=coord.dtype).to(device=coord.place) + 2.0 * rcut
    # nf x nall
    is_vir = atype < 0
    coord1 = paddle.where(
        is_vir[:, :, None], xmax, coord.reshape([batch_size, nall, 3])
    ).reshape([batch_size, nall * 3])
    if isinstance(sel, int):
        sel = [sel]
    # nloc x 3
    coord0 = coord1[:, : nloc * 3]
    # nloc x nall x 3
    diff = coord1.reshape([batch_size, -1, 3]).unsqueeze(1) - coord0.reshape(
        [batch_size, -1, 3]
    ).unsqueeze(2)
    if paddle.in_dynamic_mode():
        assert list(diff.shape) == [batch_size, nloc, nall, 3]
    # nloc x nall
    rr = paddle.linalg.norm(diff, axis=-1)
    # if central atom has two zero distances, sorting sometimes can not exclude itself
    rr = rr - paddle.eye(nloc, nall, dtype=rr.dtype).to(device=rr.place).unsqueeze(0)
    rr, nlist = paddle.sort(rr, axis=-1), paddle.argsort(rr, axis=-1)
    # nloc x (nall-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]

    return _trim_mask_distinguish_nlist(
        is_vir, atype, rr, nlist, rcut, sel, distinguish_types
    )


def _trim_mask_distinguish_nlist(
    is_vir_cntl: paddle.Tensor,
    atype_neig: paddle.Tensor,
    rr: paddle.Tensor,
    nlist: paddle.Tensor,
    rcut: float,
    sel: list[int],
    distinguish_types: bool,
) -> paddle.Tensor:
    """Trim the size of nlist, mask if any central atom is virtual, distinguish types if necessary."""
    nsel = sum(sel)
    # nloc x nsel
    batch_size, nloc, nnei = rr.shape
    if paddle.in_dynamic_mode():
        assert batch_size == is_vir_cntl.shape[0]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = paddle.concat(
            [
                rr,
                paddle.ones([batch_size, nloc, nsel - nnei]).to(
                    device=rr.place, dtype=rr.dtype
                )
                + rcut,
            ],
            axis=-1,
        )
        nlist = paddle.concat(
            [
                nlist,
                paddle.ones([batch_size, nloc, nsel - nnei], dtype=nlist.dtype).to(
                    device=rr.place
                ),
            ],
            axis=-1,
        )
        if paddle.in_dynamic_mode():
            assert list(nlist.shape) == [batch_size, nloc, nsel]
    nlist = paddle.where(
        paddle.logical_or((rr > rcut), is_vir_cntl[:, :nloc, None]), -1, nlist
    )
    if distinguish_types:
        return nlist_distinguish_types(nlist, atype_neig, sel)
    else:
        return nlist


def build_directional_neighbor_list(
    coord_cntl: paddle.Tensor,
    atype_cntl: paddle.Tensor,
    coord_neig: paddle.Tensor,
    atype_neig: paddle.Tensor,
    rcut: float,
    sel: Union[int, list[int]],
    distinguish_types: bool = True,
) -> paddle.Tensor:
    """Build directional neighbor list.

    With each central atom, all the neighbor atoms in the cut-off radius will
    be recorded in the neighbor list. The maximum neighbors is nsel. If the real
    number of neighbors is larger than nsel, the neighbors will be sorted with the
    distance and the first nsel neighbors are kept.

    Important: the central and neighboring atoms are assume to be different atoms.

    Parameters
    ----------
    coord_central : paddle.Tensor
        coordinates of central atoms. assumed to be local atoms.
        shape [batch_size, nloc_central x 3]
    atype_central : paddle.Tensor
        atomic types of central atoms. shape [batch_size, nloc_central]
        if type < 0 the atom is treated as virtual atoms.
    coord_neighbor : paddle.Tensor
        extended coordinates of neighbors atoms. shape [batch_size, nall_neighbor x 3]
    atype_central : paddle.Tensor
        extended atomic types of neighbors atoms. shape [batch_size, nall_neighbor]
        if type < 0 the atom is treated as virtual atoms.
    rcut : float
        cut-off radius
    sel : int or list[int]
        maximal number of neighbors (of each type).
        if distinguish_types==True, nsel should be list and
        the length of nsel should be equal to number of
        types.
    distinguish_types : bool
        distinguish different types.

    Returns
    -------
    neighbor_list : paddle.Tensor
        Neighbor list of shape [batch_size, nloc_central, nsel], the neighbors
        are stored in an ascending order. If the number of neighbors is less than nsel,
        the positions are masked with -1. The neighbor list of an atom looks like
        |------ nsel ------|
        xx xx xx xx -1 -1 -1
        if distinguish_types==True and we have two types
        |---- nsel[0] -----| |---- nsel[1] -----|
        xx xx xx xx -1 -1 -1 xx xx xx -1 -1 -1 -1
        For virtual atoms all neighboring positions are filled with -1.
    """
    batch_size = coord_cntl.shape[0]
    coord_cntl = coord_cntl.reshape([batch_size, -1])
    nloc_cntl = coord_cntl.shape[1] // 3
    coord_neig = coord_neig.reshape([batch_size, -1])
    nall_neig = coord_neig.shape[1] // 3
    # fill virtual atoms with large coords so they are not neighbors of any
    # real atom.
    # NOTE: control flow with double backward is not supported well yet by paddle.jit
    if not paddle.in_dynamic_mode() or decomp.numel(coord_neig) > 0:
        xmax = paddle.max(coord_cntl) + 2.0 * rcut
    else:
        xmax = (
            paddle.zeros([1], dtype=coord_neig.dtype, device=coord_neig.place)
            + 2.0 * rcut
        )
    # nf x nloc
    is_vir_cntl = atype_cntl < 0
    # nf x nall
    is_vir_neig = atype_neig < 0
    # nf x nloc x 3
    coord_cntl = coord_cntl.reshape([batch_size, nloc_cntl, 3])
    # nf x nall x 3
    coord_neig = paddle.where(
        is_vir_neig[:, :, None], xmax, coord_neig.reshape([batch_size, nall_neig, 3])
    ).reshape([batch_size, nall_neig, 3])
    # nsel
    if isinstance(sel, int):
        sel = [sel]
    # nloc x nall x 3
    diff = coord_neig[:, None, :, :] - coord_cntl[:, :, None, :]
    if paddle.in_dynamic_mode():
        assert list(diff.shape) == [batch_size, nloc_cntl, nall_neig, 3]
    # nloc x nall
    rr = paddle.linalg.norm(diff, axis=-1)
    rr, nlist = paddle.sort(rr, axis=-1), paddle.argsort(rr, axis=-1)

    # We assume that the central and neighbor atoms are diffferent,
    # thus we do not need to exclude self-neighbors.
    # # if central atom has two zero distances, sorting sometimes can not exclude itself
    # rr -= paddle.eye(nloc_cntl, nall_neig, dtype=rr.dtype, device=rr.place).unsqueeze(0)
    # rr, nlist = paddle.sort(rr, axis=-1)
    # # nloc x (nall-1)
    # rr = rr[:, :, 1:]
    # nlist = nlist[:, :, 1:]

    return _trim_mask_distinguish_nlist(
        is_vir_cntl, atype_neig, rr, nlist, rcut, sel, distinguish_types
    )


def nlist_distinguish_types(
    nlist: paddle.Tensor,
    atype: paddle.Tensor,
    sel: list[int],
):
    """Given a nlist that does not distinguish atom types, return a nlist that
    distinguish atom types.

    """
    nf, nloc, nnei = nlist.shape
    ret_nlist = []
    # nloc x nall
    tmp_atype = paddle.tile(atype.unsqueeze(1), [1, nloc, 1])
    mask = nlist == -1
    # nloc x s(nsel)
    tnlist = paddle.take_along_axis(
        tmp_atype,
        axis=2,
        indices=nlist.masked_fill(mask, 0),
    )
    tnlist = tnlist.masked_fill(mask, -1)
    snsel = tnlist.shape[2]
    for ii, ss in enumerate(sel):
        # nloc x s(nsel)
        # to int because bool cannot be sort on GPU
        pick_mask = (tnlist == ii).to(paddle.int64)
        # nloc x s(nsel), stable sort, nearer neighbors first
        pick_mask, imap = (
            paddle.sort(pick_mask, axis=-1, descending=True, stable=True),
            paddle.argsort(pick_mask, axis=-1, descending=True, stable=True),
        )
        # nloc x s(nsel)
        inlist = paddle.take_along_axis(nlist, axis=2, indices=imap)
        inlist = inlist.masked_fill(~(pick_mask.to(paddle.bool)), -1)
        # nloc x nsel[ii]
        ret_nlist.append(paddle.split(inlist, [ss, snsel - ss], axis=-1)[0])
    return paddle.concat(ret_nlist, axis=-1)


# build_neighbor_list = paddle.vmap(
#   build_neighbor_list_lower,
#   in_dims=(0,0,None,None,None),
#   out_dims=(0),
# )


def get_multiple_nlist_key(
    rcut: float,
    nsel: int,
) -> str:
    return str(rcut) + "_" + str(nsel)


def build_multiple_neighbor_list(
    coord: paddle.Tensor,
    nlist: paddle.Tensor,
    rcuts: list[float],
    nsels: list[int],
) -> dict[str, paddle.Tensor]:
    """Input one neighbor list, and produce multiple neighbor lists with
    different cutoff radius and numbers of selection out of it.  The
    required rcuts and nsels should be smaller or equal to the input nlist.

    Parameters
    ----------
    coord : paddle.Tensor
        exptended coordinates of shape [batch_size, nall x 3]
    nlist : paddle.Tensor
        Neighbor list of shape [batch_size, nloc, nsel], the neighbors
        should be stored in an ascending order.
    rcuts : list[float]
        list of cut-off radius in ascending order.
    nsels : list[int]
        maximal number of neighbors in ascending order.

    Returns
    -------
    nlist_dict : dict[str, paddle.Tensor]
        A dict of nlists, key given by get_multiple_nlist_key(rc, nsel)
        value being the corresponding nlist.

    """
    if paddle.in_dynamic_mode():
        assert len(rcuts) == len(nsels)
    if len(rcuts) == 0:
        return {}
    nb, nloc, nsel = nlist.shape
    if nsel < nsels[-1]:
        pad = -paddle.ones(
            [nb, nloc, nsels[-1] - nsel],
            dtype=nlist.dtype,
        ).to(device=nlist.place)
        # nb x nloc x nsel
        nlist = paddle.concat([nlist, pad], axis=-1)
        if paddle.is_tensor(nsel):
            nsel = paddle.to_tensor(nsels[-1], dtype=nsel.dtype)
        else:
            nsel = nsels[-1]

    # nb x nall x 3
    coord1 = coord.reshape([nb, -1, 3])
    nall = coord1.shape[1]
    # nb x nloc x 3
    coord0 = coord1[:, :nloc, :]
    nlist_mask = nlist == -1
    # nb x (nloc x nsel) x 3
    index = (
        nlist.masked_fill(nlist_mask, 0)
        .reshape([nb, nloc * nsel])
        .unsqueeze(-1)
        .expand([-1, -1, 3])
    )
    # nb x nloc x nsel x 3
    coord2 = paddle.take_along_axis(coord1, axis=1, indices=index).reshape(
        [nb, nloc, nsel, 3]
    )
    # nb x nloc x nsel x 3
    diff = coord2 - coord0[:, :, None, :]
    # nb x nloc x nsel
    rr = paddle.linalg.norm(diff, axis=-1)
    rr.masked_fill(nlist_mask, float("inf"))
    nlist0 = nlist
    ret = {}
    for rc, ns in zip(rcuts[::-1], nsels[::-1]):
        nlist0 = nlist0[:, :, :ns].masked_fill(rr[:, :, :ns] > rc, -1)
        ret[get_multiple_nlist_key(rc, ns)] = nlist0
    return ret


def extend_coord_with_ghosts(
    coord: paddle.Tensor,
    atype: paddle.Tensor,
    cell: Optional[paddle.Tensor],
    rcut: float,
    cell_cpu: Optional[paddle.Tensor] = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Extend the coordinates of the atoms by appending peridoc images.
    The number of images is large enough to ensure all the neighbors
    within rcut are appended.

    Parameters
    ----------
    coord : paddle.Tensor
        original coordinates of shape [-1, nloc*3].
    atype : paddle.Tensor
        atom type of shape [-1, nloc].
    cell : paddle.Tensor
        simulation cell tensor of shape [-1, 9].
    rcut : float
        the cutoff radius
    cell_cpu : paddle.Tensor
        cell on cpu for performance

    Returns
    -------
    extended_coord: paddle.Tensor
        extended coordinates of shape [-1, nall*3].
    extended_atype: paddle.Tensor
        extended atom type of shape [-1, nall].
    index_mapping: paddle.Tensor
        maping extended index to the local index

    """
    device = coord.place
    nf, nloc = atype.shape[:2]
    # int64 for index
    aidx = paddle.tile(paddle.arange(nloc).to(device=device).unsqueeze(0), [nf, 1])  # pylint: disable=no-explicit-dtype
    if cell is None:
        nall = nloc
        extend_coord = coord.clone()
        extend_atype = atype.clone()
        extend_aidx = aidx.clone()
    else:
        coord = coord.reshape([nf, nloc, 3])
        cell = cell.reshape([nf, 3, 3])
        cell_cpu = cell_cpu.reshape([nf, 3, 3]) if cell_cpu is not None else cell
        # nf x 3
        to_face = to_face_distance(cell_cpu)
        # nf x 3
        # *2: ghost copies on + and - directions
        # +1: central cell
        nbuff = paddle.ceil(rcut / to_face)
        INT64_MIN = -9223372036854775808
        nbuff = paddle.where(
            paddle.isinf(nbuff),
            paddle.full_like(nbuff, INT64_MIN, dtype=paddle.int64),
            nbuff.astype(paddle.int64),
        )
        # 3
        nbuff = paddle.amax(nbuff, axis=0)
        nbuff_cpu = nbuff.cpu()
        xi = (
            paddle.arange(-nbuff_cpu[0], nbuff_cpu[0] + 1, 1).to(
                dtype=env.GLOBAL_PD_FLOAT_PRECISION
            )
            # .cpu()
        )  # pylint: disable=no-explicit-dtype
        yi = (
            paddle.arange(-nbuff_cpu[1], nbuff_cpu[1] + 1, 1).to(
                dtype=env.GLOBAL_PD_FLOAT_PRECISION
            )
            # .cpu()
        )  # pylint: disable=no-explicit-dtype
        zi = (
            paddle.arange(-nbuff_cpu[2], nbuff_cpu[2] + 1, 1).to(
                dtype=env.GLOBAL_PD_FLOAT_PRECISION
            )
            # .cpu()
        )  # pylint: disable=no-explicit-dtype
        eye_3 = (
            paddle.eye(3, dtype=env.GLOBAL_PD_FLOAT_PRECISION).to(
                dtype=env.GLOBAL_PD_FLOAT_PRECISION
            )
            # .cpu()
        )
        xyz = xi.reshape([-1, 1, 1, 1]) * eye_3[0]
        xyz = xyz + yi.reshape([1, -1, 1, 1]) * eye_3[1]
        xyz = xyz + zi.reshape([1, 1, -1, 1]) * eye_3[2]
        xyz = xyz.reshape([-1, 3])
        # xyz = xyz.to(device=device)
        # ns x 3
        shift_idx = xyz[paddle.argsort(paddle.norm(xyz, axis=1))]
        ns, _ = shift_idx.shape
        nall = ns * nloc
        # nf x ns x 3
        shift_vec = paddle.einsum("sd,fdk->fsk", shift_idx, cell)
        # nf x ns x nloc x 3
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        # nf x ns x nloc
        extend_atype = paddle.tile(atype.unsqueeze(-2), [1, ns, 1])
        # nf x ns x nloc
        extend_aidx = paddle.tile(aidx.unsqueeze(-2), [1, ns, 1])
    return (
        extend_coord.reshape([nf, nall * 3]).to(device),
        extend_atype.reshape([nf, nall]).to(device),
        extend_aidx.reshape([nf, nall]).to(device),
    )
