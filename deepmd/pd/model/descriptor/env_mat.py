# SPDX-License-Identifier: LGPL-3.0-or-later

import paddle

from deepmd.pd.utils.preprocess import (
    compute_smooth_weight,
)


def _make_env_mat(
    nlist,
    coord,
    rcut: float,
    ruct_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
):
    """Make smooth environment matrix."""
    bsz, natoms, nnei = nlist.shape
    coord = coord.reshape([bsz, -1, 3])
    nall = coord.shape[1]
    mask = nlist >= 0
    # nlist = nlist * mask  ## this impl will contribute nans in Hessian calculation.
    nlist = paddle.where(mask, nlist, nall - 1)
    coord_l = coord[:, :natoms].reshape([bsz, -1, 1, 3])
    index = nlist.reshape([bsz, -1]).unsqueeze(-1).expand([-1, -1, 3])
    coord_r = paddle.take_along_axis(coord, axis=1, indices=index)
    coord_r = coord_r.reshape([bsz, natoms, nnei, 3])
    diff = coord_r - coord_l
    length = paddle.linalg.norm(diff, axis=-1, keepdim=True)
    # for index 0 nloc atom
    length = length + (~mask.unsqueeze(-1)).astype(length.dtype)
    t0 = 1 / (length + protection)
    t1 = diff / (length + protection) ** 2
    weight = compute_smooth_weight(length, ruct_smth, rcut)
    weight = weight * mask.unsqueeze(-1).astype(weight.dtype)
    if radial_only:
        env_mat = t0 * weight
    else:
        env_mat = paddle.concat([t0.astype(t1.dtype), t1], axis=-1) * weight
    return env_mat, diff * mask.unsqueeze(-1).astype(diff.dtype), weight


def prod_env_mat(
    extended_coord,
    nlist,
    atype,
    mean,
    stddev,
    rcut: float,
    rcut_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
):
    """Generate smooth environment matrix from atom coordinates and other context.

    Args:
    - extended_coord: Copied atom coordinates with shape [nframes, nall*3].
    - atype: Atom types with shape [nframes, nloc].
    - mean: Average value of descriptor per element type with shape [len(sec), nnei, 4 or 1].
    - stddev: Standard deviation of descriptor per element type with shape [len(sec), nnei, 4 or 1].
    - rcut: Cut-off radius.
    - rcut_smth: Smooth hyper-parameter for pair force & energy.
    - radial_only: Whether to return a full description or a radial-only descriptor.
    - protection: Protection parameter to prevent division by zero errors during calculations.

    Returns
    -------
    - env_mat: Shape is [nframes, natoms[1]*nnei*4].
    """
    _env_mat_se_a, diff, switch = _make_env_mat(
        nlist,
        extended_coord,
        rcut,
        rcut_smth,
        radial_only,
        protection=protection,
    )  # shape [n_atom, dim, 4 or 1]
    t_avg = mean[atype]  # [n_atom, dim, 4 or 1]
    t_std = stddev[atype]  # [n_atom, dim, 4 or 1]
    env_mat_se_a = (_env_mat_se_a - t_avg) / t_std
    return env_mat_se_a, diff, switch
