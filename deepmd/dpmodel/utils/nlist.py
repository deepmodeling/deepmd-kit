# SPDX-License-Identifier: LGPL-3.0-or-later

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
    xp_take_along_axis,
    xp_take_first_n,
)

from .region import (
    normalize_coord,
    to_face_distance,
)


def extend_input_and_build_neighbor_list(
    coord: Array,
    atype: Array,
    rcut: float,
    sel: list[int],
    mixed_types: bool = False,
    box: Array | None = None,
) -> tuple[Array, Array]:
    xp = array_api_compat.array_namespace(coord, atype)
    nframes, nloc = atype.shape[:2]
    if box is not None:
        coord_normalized = normalize_coord(
            xp.reshape(coord, (nframes, nloc, 3)),
            xp.reshape(box, (nframes, 3, 3)),
        )
    else:
        coord_normalized = coord
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, box, rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=(not mixed_types),
    )
    extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
    return extended_coord, extended_atype, mapping, nlist


## translated from torch implementation by chatgpt
def build_neighbor_list(
    coord: Array,
    atype: Array,
    nloc: int,
    rcut: float,
    sel: int | list[int],
    distinguish_types: bool = True,
) -> Array:
    """Build neighbor list for a single frame. keeps nsel neighbors.

    Parameters
    ----------
    coord : Array
        exptended coordinates of shape [batch_size, nall x 3]
    atype : Array
        extended atomic types of shape [batch_size, nall]
        type < 0 the atom is treat as virtual atoms.
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
    neighbor_list : Array
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
    xp = array_api_compat.array_namespace(coord, atype)
    batch_size = coord.shape[0]
    coord = xp.reshape(coord, (batch_size, -1))
    nall = coord.shape[1] // 3
    # fill virtual atoms with large coords so they are not neighbors of any
    # real atom.
    if array_api_compat.size(coord) > 0:
        xmax = xp.max(coord) + 2.0 * rcut
    else:
        xmax = 2.0 * rcut
    # nf x nall
    is_vir = atype < 0
    coord1 = xp.where(
        is_vir[:, :, None], xmax, xp.reshape(coord, (batch_size, nall, 3))
    )
    coord1 = xp.reshape(coord1, (batch_size, nall * 3))
    if isinstance(sel, int):
        sel = [sel]
    nsel = sum(sel)
    coord0 = coord1[:, : nloc * 3]
    diff = (
        xp.reshape(coord1, (batch_size, -1, 3))[:, None, :, :]
        - xp.reshape(coord0, (batch_size, -1, 3))[:, :, None, :]
    )
    assert list(diff.shape) == [batch_size, nloc, nall, 3]
    rr = xp.linalg.vector_norm(diff, axis=-1)
    # if central atom has two zero distances, sorting sometimes can not exclude itself
    rr -= xp.eye(nloc, nall, dtype=diff.dtype, device=array_api_compat.device(diff))[
        xp.newaxis, :, :
    ]
    nlist = xp.argsort(rr, axis=-1)
    rr = xp.sort(rr, axis=-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]
    nnei = rr.shape[2]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = xp.concatenate(
            [
                rr,
                xp.ones(
                    [batch_size, nloc, nsel - nnei],
                    dtype=rr.dtype,
                    device=array_api_compat.device(rr),
                )
                + rcut,
            ],
            axis=-1,
        )
        nlist = xp.concatenate(
            [
                nlist,
                xp.ones(
                    [batch_size, nloc, nsel - nnei],
                    dtype=nlist.dtype,
                    device=array_api_compat.device(nlist),
                ),
            ],
            axis=-1,
        )
    assert list(nlist.shape) == [batch_size, nloc, nsel]
    nlist = xp.where(
        xp.logical_or((rr > rcut), is_vir[:, :nloc, None]),
        xp.full_like(nlist, -1),
        nlist,
    )

    if distinguish_types:
        return nlist_distinguish_types(nlist, atype, sel)
    else:
        return nlist


def nlist_distinguish_types(
    nlist: Array,
    atype: Array,
    sel: list[int],
) -> Array:
    """Given a nlist that does not distinguish atom types, return a nlist that
    distinguish atom types.

    """
    xp = array_api_compat.array_namespace(nlist, atype)
    nf, nloc, _ = nlist.shape
    ret_nlist = []
    tmp_atype = xp.tile(atype[:, None, :], (1, nloc, 1))
    mask = nlist == -1
    tnlist_0 = xp.where(mask, xp.zeros_like(nlist), nlist)
    tnlist = xp_take_along_axis(tmp_atype, tnlist_0, axis=2)
    tnlist = xp.where(mask, xp.full_like(tnlist, -1), tnlist)
    for ii, ss in enumerate(sel):
        pick_mask = xp.astype(tnlist == ii, xp.int32)
        sorted_indices = xp.argsort(-pick_mask, stable=True, axis=-1)
        pick_mask_sorted = -xp.sort(-pick_mask, axis=-1)
        inlist = xp_take_along_axis(nlist, sorted_indices, axis=2)
        inlist = xp.where(
            ~xp.astype(pick_mask_sorted, xp.bool), xp.full_like(inlist, -1), inlist
        )
        ret_nlist.append(inlist[..., :ss])
    ret = xp.concat(ret_nlist, axis=-1)
    return ret


def get_multiple_nlist_key(rcut: float, nsel: int) -> str:
    return str(rcut) + "_" + str(nsel)


## translated from torch implementation by chatgpt
def build_multiple_neighbor_list(
    coord: Array,
    nlist: Array,
    rcuts: list[float],
    nsels: list[int],
) -> dict[str, Array]:
    """Input one neighbor list, and produce multiple neighbor lists with
    different cutoff radius and numbers of selection out of it.  The
    required rcuts and nsels should be smaller or equal to the input nlist.

    Parameters
    ----------
    coord : Array
        exptended coordinates of shape [batch_size, nall x 3]
    nlist : Array
        Neighbor list of shape [batch_size, nloc, nsel], the neighbors
        should be stored in an ascending order.
    rcuts : list[float]
        list of cut-off radius in ascending order.
    nsels : list[int]
        maximal number of neighbors in ascending order.

    Returns
    -------
    nlist_dict : dict[str, Array]
        A dict of nlists, key given by get_multiple_nlist_key(rc, nsel)
        value being the corresponding nlist.

    """
    xp = array_api_compat.array_namespace(coord, nlist)
    assert len(rcuts) == len(nsels)
    if len(rcuts) == 0:
        return {}
    nb, nloc, nsel = nlist.shape
    if nsel < nsels[-1]:
        pad = -1 * xp.ones(
            (nb, nloc, nsels[-1] - nsel),
            dtype=nlist.dtype,
            device=array_api_compat.device(nlist),
        )
        nlist = xp.concat([nlist, pad], axis=-1)
        nsel = nsels[-1]
    coord1 = xp.reshape(coord, (nb, -1, 3))
    coord0 = xp_take_first_n(coord1, 1, nloc)
    nlist_mask = nlist == -1
    tnlist_0 = xp.where(nlist_mask, xp.zeros_like(nlist), nlist)
    index = xp.tile(xp.reshape(tnlist_0, (nb, nloc * nsel, 1)), (1, 1, 3))
    coord2 = xp_take_along_axis(coord1, index, axis=1)
    coord2 = xp.reshape(coord2, (nb, nloc, nsel, 3))
    diff = coord2 - coord0[:, :, None, :]
    rr = xp.linalg.vector_norm(diff, axis=-1)
    rr = xp.where(nlist_mask, xp.full_like(rr, float("inf")), rr)
    nlist0 = nlist
    ret = {}
    for rc, ns in zip(rcuts[::-1], nsels[::-1], strict=True):
        tnlist_1 = nlist0[:, :, :ns]
        tnlist_1 = xp.where(rr[:, :, :ns] > rc, xp.full_like(tnlist_1, -1), tnlist_1)
        ret[get_multiple_nlist_key(rc, ns)] = tnlist_1
    return ret


## translated from torch implementation by chatgpt
def extend_coord_with_ghosts(
    coord: Array,
    atype: Array,
    cell: Array | None,
    rcut: float,
) -> tuple[Array, Array]:
    """Extend the coordinates of the atoms by appending peridoc images.
    The number of images is large enough to ensure all the neighbors
    within rcut are appended.

    Parameters
    ----------
    coord : Array
        original coordinates of shape [-1, nloc*3].
    atype : Array
        atom type of shape [-1, nloc].
    cell : Array
        simulation cell tensor of shape [-1, 9].
    rcut : float
        the cutoff radius

    Returns
    -------
    extended_coord: Array
        extended coordinates of shape [-1, nall*3].
    extended_atype: Array
        extended atom type of shape [-1, nall].
    index_mapping: Array
        mapping extended index to the local index

    """
    xp = array_api_compat.array_namespace(coord, atype)
    nf, nloc = atype.shape
    # int64 for index
    aidx = xp.tile(
        xp.arange(nloc, dtype=xp.int64, device=array_api_compat.device(atype))[
            xp.newaxis, :
        ],
        (nf, 1),
    )
    if cell is None:
        nall = nloc
        extend_coord = coord
        extend_atype = atype
        extend_aidx = aidx
    else:
        coord = xp.reshape(coord, (nf, nloc, 3))
        cell = xp.reshape(cell, (nf, 3, 3))
        to_face = to_face_distance(cell)
        nbuff = xp.astype(xp.ceil(rcut / to_face), xp.int64)
        nbuff = xp.max(nbuff, axis=0)
        xi = xp.arange(
            -int(nbuff[0]),
            int(nbuff[0]) + 1,
            1,
            dtype=xp.int64,
            device=array_api_compat.device(coord),
        )
        yi = xp.arange(
            -int(nbuff[1]),
            int(nbuff[1]) + 1,
            1,
            dtype=xp.int64,
            device=array_api_compat.device(coord),
        )
        zi = xp.arange(
            -int(nbuff[2]),
            int(nbuff[2]) + 1,
            1,
            dtype=xp.int64,
            device=array_api_compat.device(coord),
        )
        xyz = xp.linalg.outer(
            xi, xp.asarray([1, 0, 0], device=array_api_compat.device(xi))
        )[:, xp.newaxis, xp.newaxis, :]
        xyz = (
            xyz
            + xp.linalg.outer(
                yi, xp.asarray([0, 1, 0], device=array_api_compat.device(yi))
            )[xp.newaxis, :, xp.newaxis, :]
        )
        xyz = (
            xyz
            + xp.linalg.outer(
                zi, xp.asarray([0, 0, 1], device=array_api_compat.device(zi))
            )[xp.newaxis, xp.newaxis, :, :]
        )
        xyz = xp.reshape(xyz, (-1, 3))
        xyz = xp.astype(xyz, coord.dtype)
        shift_idx = xp.take(xyz, xp.argsort(xp.linalg.vector_norm(xyz, axis=1)), axis=0)
        ns, _ = shift_idx.shape
        nall = ns * nloc
        if array_api_compat.is_jax_namespace(xp):
            # Avoid JAX internal errors in tensordot.
            shift_vec = xp.sum(
                shift_idx[xp.newaxis, :, :, xp.newaxis] * cell[:, xp.newaxis, :, :],
                axis=2,
            )
        else:
            # shift_vec = xp.einsum("sd,fdk->fsk", shift_idx, cell)
            shift_vec = xp.tensordot(shift_idx, cell, axes=([1], [1]))
            shift_vec = xp.permute_dims(shift_vec, (1, 0, 2))
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        extend_atype = xp.tile(atype[:, :, xp.newaxis], (1, ns, 1))
        extend_aidx = xp.tile(aidx[:, :, xp.newaxis], (1, ns, 1))

    return (
        xp.reshape(extend_coord, (nf, nall * 3)),
        xp.reshape(extend_atype, (nf, nall)),
        xp.reshape(extend_aidx, (nf, nall)),
    )


def is_vesin_available() -> bool:
    """Whether the optional ``vesin`` O(N) neighbor-list backend is importable."""
    try:
        import vesin  # noqa: F401
    except ImportError:
        return False
    return True


def build_neighbor_list_vesin(
    coords: Array,
    cells: Array | None,
    atom_types: Array,
    rcut: float,
    sel: list[int],
    distinguish_types: bool,
) -> tuple[Array, Array, Array, Array]:
    """Build the extended system and neighbor list with the O(N) ``vesin`` cell list.

    This is a host-side, drop-in replacement for the native all-pairs O(N^2)
    :func:`extend_input_and_build_neighbor_list` on the Python inference path.
    The neighbor *search* is non-differentiable -- it only produces integer
    index arrays and the gathered ghost coordinates -- so an external cell-list
    library may be used without affecting the autograd graph of the model.

    Parameters
    ----------
    coords : np.ndarray
        local atom coordinates, shape (nframes, nloc, 3).
    cells : np.ndarray or None
        simulation cell, shape (nframes, 9) or (nframes, 3, 3). ``None`` for
        non-periodic systems.
    atom_types : np.ndarray
        atom types, shape (nframes, nloc).
    rcut : float
        cutoff radius.
    sel : list[int]
        maximal number of selected neighbors (summed over types).
    distinguish_types : bool
        whether to reorder the neighbor list per atom type (``not mixed_types``).

    Returns
    -------
    extended_coord : np.ndarray, shape (nframes, nall, 3)
    extended_atype : np.ndarray, shape (nframes, nall)
    nlist : np.ndarray, shape (nframes, nloc, sum(sel))
    mapping : np.ndarray, shape (nframes, nall)
    """
    import numpy as np

    coords = np.asarray(coords, dtype=np.float64).reshape(coords.shape[0], -1, 3)
    nframes = coords.shape[0]
    atom_types = np.asarray(atom_types).reshape(nframes, -1)
    if cells is not None:
        cells = np.asarray(cells, dtype=np.float64).reshape(nframes, 3, 3)

    frame_results = [
        _build_neighbor_list_vesin_single(
            coords[ff],
            cells[ff] if cells is not None else None,
            atom_types[ff],
            rcut,
            sel,
            distinguish_types,
        )
        for ff in range(nframes)
    ]
    # pad to a common nall across frames
    max_nall = max(ec.shape[0] for ec, _, _, _ in frame_results)
    ext_coords, ext_atypes, nlists, mappings = [], [], [], []
    for ec, ea, nl, mp in frame_results:
        pad = max_nall - ec.shape[0]
        if pad > 0:
            ec = np.concatenate([ec, np.zeros((pad, 3), dtype=ec.dtype)], axis=0)
            ea = np.concatenate([ea, np.full(pad, -1, dtype=ea.dtype)], axis=0)
            mp = np.concatenate([mp, np.zeros(pad, dtype=mp.dtype)], axis=0)
        ext_coords.append(ec)
        ext_atypes.append(ea)
        nlists.append(nl)
        mappings.append(mp)
    return (
        np.stack(ext_coords, axis=0),
        np.stack(ext_atypes, axis=0),
        np.stack(nlists, axis=0),
        np.stack(mappings, axis=0),
    )


def _build_neighbor_list_vesin_single(
    positions: Array,
    cell: Array | None,
    atype: Array,
    rcut: float,
    sel: list[int],
    distinguish_types: bool,
) -> tuple[Array, Array, Array, Array]:
    """Single-frame variant of :func:`build_neighbor_list_vesin`."""
    import numpy as np
    import vesin

    nsel = sum(sel)
    nloc = positions.shape[0]
    periodic = cell is not None
    box = cell if periodic else np.zeros((3, 3), dtype=np.float64)

    nl = vesin.NeighborList(cutoff=rcut, full_list=True)
    ii, jj, ss = nl.compute(
        points=positions, box=box, periodic=periodic, quantities="ijS"
    )
    ii = ii.astype(np.int64)
    jj = jj.astype(np.int64)
    ss = ss.astype(np.float64)

    # ghost atoms: neighbors reached through a non-zero periodic shift
    out_mask = np.any(ss != 0, axis=1)
    out_idx = jj[out_mask]
    out_coords = positions[out_idx] + ss[out_mask].dot(box)
    nghost = out_idx.size

    extended_coord = np.concatenate((positions, out_coords), axis=0)
    extended_atype = np.concatenate((atype, atype[out_idx]))
    mapping = np.concatenate((np.arange(nloc, dtype=np.int64), out_idx))

    # remap neighbor column indices: ghosts -> [nloc, nloc + nghost)
    neigh_idx = jj.copy()
    neigh_idx[out_mask] = np.arange(nloc, nloc + nghost, dtype=np.int64)

    # group pairs by center atom (vesin does not guarantee CSR ordering)
    counts = np.bincount(ii, minlength=nloc)
    max_nn = int(counts.max()) if counts.size > 0 else 0
    order = np.argsort(ii, kind="stable")
    rows = ii[order]
    cols = np.arange(ii.size, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    dense_idx = np.full((nloc, max_nn), -1, dtype=np.int64)
    if ii.size > 0:
        dense_idx[rows, cols] = neigh_idx[order]

    # sort candidates by distance, keep the nsel nearest within rcut, pad with -1
    valid = dense_idx >= 0
    lookup = np.where(valid, dense_idx, 0)
    dists = np.linalg.norm(extended_coord[lookup] - positions[:, None, :], axis=-1)
    valid &= dists <= rcut
    dists = np.where(valid, dists, np.inf)
    sort_order = np.argsort(dists, axis=-1)
    sorted_idx = np.take_along_axis(dense_idx, sort_order, axis=-1)
    sorted_valid = np.take_along_axis(valid, sort_order, axis=-1)

    nlist = np.full((nloc, nsel), -1, dtype=np.int64)
    keep = min(nsel, max_nn)
    if keep > 0:
        nlist[:, :keep] = np.where(sorted_valid[:, :keep], sorted_idx[:, :keep], -1)

    if distinguish_types:
        nlist = nlist_distinguish_types(nlist[None], extended_atype[None], sel)[0]

    return extended_coord, extended_atype, nlist, mapping
