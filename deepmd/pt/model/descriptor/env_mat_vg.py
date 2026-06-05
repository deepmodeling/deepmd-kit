# SPDX-License-Identifier: LGPL-3.0-or-later
"""Environment matrix for variational-Gaussian (VG) smooth descriptors."""

import math

import torch

from deepmd.pt.utils.preprocess import (
    compute_smooth_weight,
)

VG_ENV_DIM: int = 5
_SQRT2: float = math.sqrt(2.0)


def vg_gaussian_radial_phi(
    length: torch.Tensor,
    sigma_ij: torch.Tensor,
    protection: float = 0.0,
) -> torch.Tensor:
    """Gaussian-averaged 1/r kernel: (1/r) * erf(r / (sqrt(2)*sigma))."""
    r = length + protection
    sigma = sigma_ij + 1e-12
    return (1.0 / r) * torch.erf(r / (_SQRT2 * sigma))


def vg_smooth_radial(
    length: torch.Tensor,
    sigma_ij: torch.Tensor,
    rcut_smth: float,
    rcut: float,
    protection: float = 0.0,
) -> torch.Tensor:
    """Radial kernel s(r, sigma) with the same smooth cutoff as DP-SE."""
    phi = vg_gaussian_radial_phi(length, sigma_ij, protection=protection)
    weight = compute_smooth_weight(length, rcut_smth, rcut)
    return phi * weight


def _gather_neighbor_sigma(
    aparam: torch.Tensor,
    nlist: torch.Tensor,
    nloc: int,
    nall: int,
) -> torch.Tensor:
    """Map per-atom aparam to neighbor-list sigma values."""
    nf, _, nnei = nlist.shape
    sigma_loc = aparam[:, :nloc, 0].to(dtype=nlist.dtype)
    sigma_ext = torch.zeros(
        (nf, nall),
        dtype=sigma_loc.dtype,
        device=sigma_loc.device,
    )
    sigma_ext[:, :nloc] = sigma_loc
    index = nlist.reshape(nf, -1)
    sigma_nei = torch.gather(sigma_ext, 1, index)
    return sigma_nei.view(nf, nloc, nnei)


def _make_env_mat_vg(
    nlist: torch.Tensor,
    coord: torch.Tensor,
    aparam: torch.Tensor,
    rcut: float,
    rcut_smth: float,
    protection: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the 5-column VG local environment matrix."""
    bsz, natoms, nnei = nlist.shape
    coord = coord.view(bsz, -1, 3)
    nall = coord.shape[1]
    mask = nlist >= 0
    nlist_safe = torch.where(mask, nlist, nall)

    coord_l = coord[:, :natoms].view(bsz, -1, 1, 3)
    index = nlist_safe.view(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
    coord_pad = torch.concat([coord, coord[:, -1:, :] + rcut], dim=1)
    coord_r = torch.gather(coord_pad, 1, index).view(bsz, natoms, nnei, 3)
    diff = coord_r - coord_l
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    length = length + (~mask).unsqueeze(-1)

    sigma_center = aparam[:, :natoms, 0].unsqueeze(-1)
    sigma_neighbor = _gather_neighbor_sigma(aparam, nlist, natoms, nall).unsqueeze(-1)
    sigma_ij = torch.sqrt(sigma_center * sigma_center + sigma_neighbor * sigma_neighbor)

    s_val = vg_smooth_radial(
        length,
        sigma_ij,
        rcut_smth,
        rcut,
        protection=protection,
    )
    s_val = s_val * mask.unsqueeze(-1)
    x_hat = diff / (length + protection)
    sigma_col = s_val * sigma_ij / (length + protection)

    env_mat = torch.cat(
        [
            s_val,
            s_val * x_hat,
            sigma_col,
        ],
        dim=-1,
    )
    return env_mat, diff * mask.unsqueeze(-1), s_val


def prod_env_mat_vg(
    extended_coord: torch.Tensor,
    nlist: torch.Tensor,
    atype: torch.Tensor,
    aparam: torch.Tensor,
    mean: torch.Tensor,
    stddev: torch.Tensor,
    rcut: float,
    rcut_smth: float,
    protection: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalized VG environment matrix for descriptor embedding."""
    env_mat, diff, switch = _make_env_mat_vg(
        nlist,
        extended_coord,
        aparam,
        rcut,
        rcut_smth,
        protection=protection,
    )
    t_avg = mean[atype]
    t_std = stddev[atype]
    env_mat = (env_mat - t_avg) / t_std
    return env_mat, diff, switch


def tabulate_fusion_se_a_vg(
    table: torch.Tensor,
    table_info: torch.Tensor,
    em_x: torch.Tensor,
    em: torch.Tensor,
    last_layer_size: int,
) -> torch.Tensor:
    """Tabulate the VG 5-column env mat via two 4-column fusion calls."""
    gr4 = torch.ops.deepmd.tabulate_fusion_se_a(
        table,
        table_info,
        em_x,
        em[..., :4].contiguous(),
        last_layer_size,
    )[0]
    em5 = torch.zeros_like(em[..., :4])
    em5[..., 0:1] = em[..., 4:5]
    gr5 = torch.ops.deepmd.tabulate_fusion_se_a(
        table,
        table_info,
        em_x,
        em5.contiguous(),
        last_layer_size,
    )[0]
    return torch.cat([gr4, gr5[:, 0:1, :]], dim=1)
