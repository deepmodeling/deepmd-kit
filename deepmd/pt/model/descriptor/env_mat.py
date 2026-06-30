# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.utils.preprocess import (
    compute_exp_sw,
    compute_smooth_weight,
)


def _make_env_mat(
    nlist: torch.Tensor,
    coord: torch.Tensor,
    rcut: float,
    ruct_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
    use_exp_switch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make smooth environment matrix."""
    bsz, natoms, nnei = nlist.shape
    coord = coord.view(bsz, -1, 3)
    nall = coord.shape[1]
    mask = nlist >= 0
    # nlist = nlist * mask  ## this impl will contribute nans in Hessian calculation.
    nlist = torch.where(mask, nlist, nall)
    coord_l = coord[:, :natoms].view(bsz, -1, 1, 3)
    index = nlist.view(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
    coord_pad = torch.concat([coord, coord[:, -1:, :] + rcut], dim=1)
    coord_r = torch.gather(coord_pad, 1, index)
    coord_r = coord_r.view(bsz, natoms, nnei, 3)
    diff = coord_r - coord_l
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    # for index 0 nloc atom
    length = length + ~mask.unsqueeze(-1)
    t0 = 1 / (length + protection)
    t1 = diff / (length + protection) ** 2
    weight = (
        compute_smooth_weight(length, ruct_smth, rcut)
        if not use_exp_switch
        else compute_exp_sw(length, ruct_smth, rcut)
    )
    weight = weight * mask.unsqueeze(-1)
    if radial_only:
        env_mat = t0 * weight
    else:
        env_mat = torch.cat([t0, t1], dim=-1) * weight
    return env_mat, diff * mask.unsqueeze(-1), weight


def prod_env_mat(
    extended_coord: torch.Tensor,
    nlist: torch.Tensor,
    atype: torch.Tensor,
    mean: torch.Tensor,
    stddev: torch.Tensor,
    rcut: float,
    rcut_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
    use_exp_switch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    - use_exp_switch: Whether to use the exponential switch function.

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
        use_exp_switch=use_exp_switch,
    )  # shape [n_atom, dim, 4 or 1]
    t_avg = mean[atype]  # [n_atom, dim, 4 or 1]
    t_std = stddev[atype]  # [n_atom, dim, 4 or 1]
    env_mat_se_a = (_env_mat_se_a - t_avg) / t_std
    return env_mat_se_a, diff, switch


def prod_env_mat_flat(
    extended_coord_flat: torch.Tensor,
    nlist_flat: torch.Tensor,
    atype_flat: torch.Tensor,
    mean: torch.Tensor,
    stddev: torch.Tensor,
    rcut: float,
    rcut_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
    use_exp_switch: bool = False,
    coord_flat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate smooth environment matrix in flat format.

    Parameters
    ----------
    extended_coord_flat
        Extended atom coordinates with shape ``[nall, 3]``.
    nlist_flat
        Neighbor list with shape ``[nloc, nnei]``. ``-1`` marks padding.
    atype_flat
        Central atom types with shape ``[nloc]``.
    mean, stddev
        Descriptor statistics with shape ``[ntypes, nnei, 4 or 1]``.
    rcut, rcut_smth
        Cutoff radius and smooth cutoff radius.
    radial_only
        Whether to return radial-only descriptors.
    protection
        Small positive value used in radial divisions.
    use_exp_switch
        Whether to use the exponential switch function.
    coord_flat
        Optional central atom coordinates with shape ``[nloc, 3]``.

    Returns
    -------
    env_mat
        Environment matrix with shape ``[nloc, nnei, 4 or 1]``.
    diff
        Difference vectors with shape ``[nloc, nnei, 3]``.
    switch
        Switch function values with shape ``[nloc, nnei, 1]``.
    """
    nloc, nnei = nlist_flat.shape
    nall = extended_coord_flat.shape[0]

    mask = nlist_flat >= 0
    nlist_safe = torch.where(mask, nlist_flat, nall)

    # coord_l: [nloc, 1, 3]
    if coord_flat is not None:
        coord_l = coord_flat.view(nloc, 1, 3)
    else:
        coord_l = extended_coord_flat[:nloc].view(nloc, 1, 3)

    # Gather neighbor coordinates
    index = nlist_safe.view(-1).unsqueeze(-1).expand(-1, 3)
    coord_pad = torch.cat(
        [extended_coord_flat, extended_coord_flat[-1:, :] + rcut], dim=0
    )
    coord_r = torch.gather(coord_pad, 0, index)
    coord_r = coord_r.view(nloc, nnei, 3)

    # Compute differences and distances
    diff = coord_r - coord_l
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    length = length + ~mask.unsqueeze(-1)

    t0 = 1 / (length + protection)
    t1 = diff / (length + protection) ** 2

    weight = (
        compute_smooth_weight(length, rcut_smth, rcut)
        if not use_exp_switch
        else compute_exp_sw(length, rcut_smth, rcut)
    )
    weight = weight * mask.unsqueeze(-1)

    if radial_only:
        env_mat = t0 * weight
    else:
        env_mat = torch.cat([t0, t1], dim=-1) * weight

    diff = diff * mask.unsqueeze(-1)

    # Normalize by mean and stddev
    t_avg = mean[atype_flat]  # [nloc, nnei, 4]
    t_std = stddev[atype_flat]  # [nloc, nnei, 4]
    env_mat = (env_mat - t_avg) / t_std

    return env_mat, diff, weight
