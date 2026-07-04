# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import (
    TYPE_CHECKING,
    Any,
)

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

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask


def _is_ndtensorflow_namespace(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "deepmd._vendors.ndtensorflow"


def _arange_nbuff(nbuff: Array, index: int, xp: Any, device: Any) -> Array:
    bound = nbuff[index]
    if not _is_ndtensorflow_namespace(xp):
        bound = int(bound)
    return xp.arange(-bound, bound + 1, 1, dtype=xp.int64, device=device)


def _size(x: Array, xp: Any) -> Any:
    if _is_ndtensorflow_namespace(xp):
        return x.size
    return array_api_compat.size(x)


def _is_static_shape(shape: Any) -> bool:
    return all(isinstance(dim, int) for dim in shape)


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


def apply_pair_exclusion_nlist(
    nlist: Array,
    atype_ext: Array,
    pair_excl: "PairExcludeMask | None",
) -> Array:
    """Apply model-level pair-type exclusion to a dense neighbor list.

    Replaces excluded neighbor entries with ``-1`` so that downstream
    descriptors see them as empty slots.  Identity (returns ``nlist``
    unchanged) when *pair_excl* is ``None`` or its exclude-types list is
    empty.

    This is the nlist-representation counterpart of
    :func:`deepmd.dpmodel.utils.neighbor_graph.apply_pair_exclusion`.

    See Also
    --------
    C++ twin ``applyPairExclusionNlist`` in ``source/api_cc/include/commonPT.h``
        The inference-path mirror. Same argument order (nlist, atype_ext, ...),
        same variable names (``type_ij``, ``keep``): it computes ``type_ij``
        from the center/neighbor types via the flat ``(ntypes+1)^2`` table and
        replaces excluded entries with ``-1``.

    Parameters
    ----------
    nlist : Array
        Dense neighbor list of shape ``(nf, nloc, nnei)``.  Entries equal
        to ``-1`` indicate empty / padding slots.
    atype_ext : Array
        Extended atom types of shape ``(nf, nall)``.
    pair_excl : PairExcludeMask or None
        Exclusion mask object, or ``None`` / empty to skip.

    Returns
    -------
    Array
        Neighbor list of the same shape with excluded entries set to ``-1``.
        Erasing ``-1`` entries a second time is a no-op (idempotent).
    """
    if pair_excl is None or len(pair_excl.exclude_types) == 0:
        return nlist
    xp = array_api_compat.array_namespace(nlist, atype_ext)
    pair_mask = pair_excl.build_type_exclude_mask(nlist, atype_ext)
    return xp.where(pair_mask == 1, nlist, xp.full_like(nlist, -1))


## translated from torch implementation by chatgpt
def build_neighbor_list(
    coord: Array,
    atype: Array,
    nloc: int,
    rcut: float,
    sel: int | list[int],
    distinguish_types: bool = True,
    pair_excl: "PairExcludeMask | None" = None,
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
    pair_excl : PairExcludeMask or None, optional
        When provided, excluded type pairs are erased from the returned
        neighbor list (entries set to ``-1``) immediately after the
        geometric search.  This is a convenience shortcut for calling
        :func:`apply_pair_exclusion_nlist` separately.  ``None`` (default)
        leaves the list unchanged.

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
    if _size(coord, xp) > 0:
        xmax = xp.max(coord) + 2.0 * rcut
    else:
        if _is_ndtensorflow_namespace(xp):
            xmax = xp.asarray(
                2.0 * rcut,
                dtype=coord.dtype,
                device=array_api_compat.device(coord),
            )
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
    if _is_static_shape(diff.shape):
        assert list(diff.shape) == [batch_size, nloc, nall, 3]
    rr = xp.linalg.vector_norm(diff, axis=-1)
    # if central atom has two zero distances, sorting sometimes can not exclude itself
    rr -= xp.eye(nloc, nall, dtype=diff.dtype, device=array_api_compat.device(diff))[
        xp.newaxis, :, :
    ]
    nlist = xp.astype(xp.argsort(rr, axis=-1), xp.int64)
    rr = xp.sort(rr, axis=-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]
    nnei = rr.shape[2]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = xp.concat(
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
        nlist = xp.concat(
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
    if _is_static_shape(nlist.shape):
        assert list(nlist.shape) == [batch_size, nloc, nsel]
    nlist = xp.where(
        xp.logical_or((rr > rcut), is_vir[:, :nloc, None]),
        xp.full_like(nlist, -1),
        nlist,
    )

    if distinguish_types:
        nlist = nlist_distinguish_types(nlist, atype, sel)
    return apply_pair_exclusion_nlist(nlist, atype, pair_excl)


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


def format_nlist(
    extended_coord: Array,
    nlist: Array,
    nnei: int,
    rcut: float,
    extra_nlist_sort: bool = False,
) -> Array:
    """Format a neighbor list to a fixed neighbor count.

    If the input neighbor axis is shorter than ``nnei``, pad it with ``-1``.
    If the input neighbor axis is longer than ``nnei``, sort neighbors by
    distance, mask neighbors outside ``rcut`` with ``-1``, and truncate to
    ``nnei`` entries. Otherwise, preserve the input order and mask neighbors
    outside ``rcut`` with ``-1``. When ``extra_nlist_sort`` is true, use the
    sort-and-truncate path even when the input neighbor axis is not longer than
    ``nnei``.

    Parameters
    ----------
    extended_coord : Array
        Extended coordinates of shape ``[nf, nall, 3]`` or
        ``[nf, nall * 3]``.
    nlist : Array
        Neighbor list of shape ``[nf, nloc, n_nnei]``. Invalid neighbor
        entries are marked with ``-1``.
    nnei : int
        Target number of selected neighbors.
    rcut : float
        Cutoff radius. Neighbors farther than ``rcut`` are marked with ``-1``.
    extra_nlist_sort : bool, optional
        Whether to force distance sorting and truncation even when the input
        neighbor axis is not larger than ``nnei``. This is needed by models
        whose lower-level forward path requires a sorted neighbor list.

    Returns
    -------
    Array
        Formatted neighbor list of shape ``[nf, nloc, nnei]``. Missing or
        out-of-cutoff neighbors are marked with ``-1``.
    """
    xp = array_api_compat.array_namespace(extended_coord, nlist)
    n_nf, n_nloc, n_nnei = nlist.shape
    extended_coord = extended_coord.reshape([n_nf, -1, 3])
    ret = nlist

    if n_nnei < nnei:
        ret = xp.concat(
            [
                nlist,
                -1
                * xp.ones(
                    [n_nf, n_nloc, nnei - n_nnei],
                    dtype=nlist.dtype,
                    device=array_api_compat.device(nlist),
                ),
            ],
            axis=-1,
        )

    # Order matters for torch.export: Python evaluates `or` left-to-right
    # with short-circuit.  When `extra_nlist_sort=True` (Python bool) is
    # on the left, the right-hand `n_nnei > nnei` is not evaluated, so no
    # symbolic guard is registered on the dynamic `n_nnei` dimension.
    # Swapping the operands would force the SymInt comparison to run and
    # emit an `_assert_scalar` node in the exported graph.
    if extra_nlist_sort or n_nnei > nnei:
        n_nf, n_nloc, n_nnei = nlist.shape
        m_real_nei = nlist >= 0
        ret = xp.where(m_real_nei, nlist, 0)
        coord0 = xp_take_first_n(extended_coord, 1, n_nloc)
        index = xp.tile(ret.reshape(n_nf, n_nloc * n_nnei, 1), (1, 1, 3))
        coord1 = xp_take_along_axis(extended_coord, index, axis=1)
        coord1 = coord1.reshape(n_nf, n_nloc, n_nnei, 3)
        rr = xp.linalg.norm(coord0[:, :, None, :] - coord1, axis=-1)
        rr = xp.where(m_real_nei, rr, float("inf"))
        rr, ret_mapping = xp.sort(rr, axis=-1), xp.argsort(rr, axis=-1)
        ret = xp_take_along_axis(ret, ret_mapping, axis=2)
        ret = xp.where(rr > rcut, -1, ret)
        ret = ret[..., :nnei]
    else:
        # not extra_nlist_sort and n_nnei <= nnei: no reordering is
        # needed (these descriptors reduce over neighbors order-
        # independently), but we must still drop neighbors beyond rcut.
        # The C++/LAMMPS neighbor list is built with rcut+skin and is
        # NOT rcut-filtered before forward_lower; without this, out-of-
        # rcut neighbors leak into the descriptor whenever the per-atom
        # neighbor count <= nnei (this branch), making the result
        # order-dependent (see discussion #5438).
        if n_nnei == nnei:
            ret = nlist
        # else (n_nnei < nnei): `ret` is already padded to nnei above.
        n_nf, n_nloc, n_pad = ret.shape
        m_real_nei = ret >= 0
        coord0 = xp_take_first_n(extended_coord, 1, n_nloc)
        index = xp.tile(
            xp.where(m_real_nei, ret, 0).reshape(n_nf, n_nloc * n_pad, 1),
            (1, 1, 3),
        )
        coord1 = xp_take_along_axis(extended_coord, index, axis=1)
        coord1 = coord1.reshape(n_nf, n_nloc, n_pad, 3)
        rr = xp.linalg.norm(coord0[:, :, None, :] - coord1, axis=-1)
        ret = xp.where(m_real_nei & (rr > rcut), -1, ret)
    if isinstance(ret.shape[-1], int):
        assert ret.shape[-1] == nnei
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
        device = array_api_compat.device(coord)
        xi = _arange_nbuff(nbuff, 0, xp, device)
        yi = _arange_nbuff(nbuff, 1, xp, device)
        zi = _arange_nbuff(nbuff, 2, xp, device)
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
