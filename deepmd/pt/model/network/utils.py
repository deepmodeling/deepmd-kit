# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from typing import (
    Optional,
)

import torch

from deepmd.pt.utils import (
    env,
)


@torch.jit.export
def aggregate(
    data: torch.Tensor,
    owners: torch.Tensor,
    average: bool = True,
    num_owner: Optional[int] = None,
) -> torch.Tensor:
    """
    Aggregate rows in data by specifying the owners.

    Parameters
    ----------
    data : data tensor to aggregate [n_row, feature_dim]
    owners : specify the owner of each row [n_row, 1]
    average : if True, average the rows, if False, sum the rows.
        Default = True
    num_owner : the number of owners, this is needed if the
        max idx of owner is not presented in owners tensor
        Default = None

    Returns
    -------
    output: [num_owner, feature_dim]
    """
    bin_count = torch.bincount(owners)
    bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

    if (num_owner is not None) and (bin_count.shape[0] != num_owner):
        difference = num_owner - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])

    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, owners, data)
    if average:
        output = (output.T / bin_count).T
    return output


@torch.jit.export
def get_graph_index(
    nlist: torch.Tensor,
    nlist_mask: torch.Tensor,
    a_nlist_mask: torch.Tensor,
    d_nlist_mask: torch.Tensor,
    nall: int,
    calculate_dihedral: bool = False,
    use_loc_mapping: bool = True,
):
    """
    Get the index mapping for edge graph and angle graph, ready in `aggregate` or `index_select`.

    Parameters
    ----------
    nlist : nf x nloc x nnei
        Neighbor list. (padded neis are set to 0)
    nlist_mask : nf x nloc x nnei
        Masks of the neighbor list. real nei 1 otherwise 0
    a_nlist_mask : nf x nloc x a_nnei
        Masks of the neighbor list for angle. real nei 1 otherwise 0
    nall
        The number of extended atoms.

    Returns
    -------
    edge_index : n_edge x 2
        n2e_index : n_edge
            Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
        n_ext2e_index : n_edge
            Broadcast indices from extended node(j) to edge(ij).
    angle_index : n_angle x 3
        n2a_index : n_angle
            Broadcast indices from extended node(j) to angle(ijk).
        eij2a_index : n_angle
            Broadcast indices from edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
        eik2a_index : n_angle
            Broadcast indices from edge(ik) to angle(ijk).
    dihedral_index : n_dihedral x 2
        aijk2d_index : n_dihedral
            Broadcast indices from angle(ijk) to dihedral(ijkl), or reduction indices from dihedral(ijkl) to angle(ijk).
        aijl2d_index : n_dihedral
            Broadcast indices from angle(ijl) to dihedral(ijkl).
    """
    nf, nloc, nnei = nlist.shape
    _, _, a_nnei = a_nlist_mask.shape
    # nf x nloc x nnei x nnei
    # nlist_mask_3d = nlist_mask[:, :, :, None] & nlist_mask[:, :, None, :]
    a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
    n_edge = nlist_mask.sum().item()

    # following: get n2e_index, n_ext2e_index, n2a_index, eij2a_index, eik2a_index

    # 1. atom graph
    # node(i) to edge(ij) index_select; edge(ij) to node aggregate
    nlist_loc_index = torch.arange(0, nf * nloc, dtype=nlist.dtype, device=nlist.device)
    # nf x nloc x nnei
    n2e_index = nlist_loc_index.reshape(nf, nloc, 1).expand(-1, -1, nnei)
    # n_edge
    n2e_index = n2e_index[nlist_mask]  # graph node index, atom_graph[:, 0]

    # node_ext(j) to edge(ij) index_select
    frame_shift = torch.arange(0, nf, dtype=nlist.dtype, device=nlist.device) * (
        nall if not use_loc_mapping else nloc
    )
    shifted_nlist = nlist + frame_shift[:, None, None]
    # n_edge
    n_ext2e_index = shifted_nlist[nlist_mask]  # graph neighbor index, atom_graph[:, 1]

    # 2. edge graph
    # node(i) to angle(ijk) index_select
    n2a_index = nlist_loc_index.reshape(nf, nloc, 1, 1).expand(-1, -1, a_nnei, a_nnei)
    # n_angle
    n2a_index = n2a_index[a_nlist_mask_3d]

    # edge(ij) to angle(ijk) index_select; angle(ijk) to edge(ij) aggregate
    edge_id = torch.arange(0, n_edge, dtype=nlist.dtype, device=nlist.device)
    # nf x nloc x nnei
    edge_index = torch.zeros([nf, nloc, nnei], dtype=nlist.dtype, device=nlist.device)
    edge_index[nlist_mask] = edge_id
    # only cut a_nnei neighbors, to avoid nnei x nnei
    edge_index = edge_index[:, :, :a_nnei]
    edge_index_ij = edge_index.unsqueeze(-1).expand(-1, -1, -1, a_nnei)
    # n_angle
    eij2a_index = edge_index_ij[a_nlist_mask_3d]

    # edge(ik) to angle(ijk) index_select
    edge_index_ik = edge_index.unsqueeze(-2).expand(-1, -1, a_nnei, -1)
    # n_angle
    eik2a_index = edge_index_ik[a_nlist_mask_3d]

    if calculate_dihedral:
        # 3. angle graph
        n_angle = a_nlist_mask_3d.sum().item()
        _, _, d_nnei = d_nlist_mask.shape

        # nf x nloc x d_nnei x d_nnei x d_nnei
        # should expel same j k l
        d_nlist_mask_4d = (
            d_nlist_mask[:, :, :, None, None]
            & d_nlist_mask[:, :, None, :, None]
            & d_nlist_mask[:, :, None, None, :]
        )
        # d_nnei x d_nnei
        d_eye = torch.eye(d_nnei, dtype=d_nlist_mask.dtype, device=d_nlist_mask.device)
        d_eye = d_eye[:, :, None] | d_eye[:, None, :] | d_eye[None, :, :]
        d_nlist_mask_4d = d_nlist_mask_4d & ~d_eye[None, None, ...]

        # angle(ijk) to dihedral(ijkl) index_select; dihedral(ijkl) to angle(ijk) aggregate
        angle_id = torch.arange(0, n_angle, dtype=nlist.dtype, device=nlist.device)
        # nf x nloc x a_nnei x a_nnei
        angle_index = torch.zeros(
            [nf, nloc, a_nnei, a_nnei], dtype=nlist.dtype, device=nlist.device
        )
        angle_index[a_nlist_mask_3d] = angle_id

        # only cut d_nnei neighbors, to avoid a_nnei x a_nnei x a_nnei
        angle_index = angle_index[:, :, :d_nnei, :d_nnei]
        angle_index_ijk = angle_index.unsqueeze(-1).expand(-1, -1, -1, -1, d_nnei)
        # n_dihedral
        aijk2d_index = angle_index_ijk[d_nlist_mask_4d]

        # angle(ijl) to dihedral(ijkl) index_select;
        angle_index_ijl = angle_index.unsqueeze(-2).expand(-1, -1, -1, d_nnei, -1)
        # n_dihedral
        aijl2d_index = angle_index_ijl[d_nlist_mask_4d]

        dihedral_index = torch.cat(
            [aijk2d_index.unsqueeze(-1), aijl2d_index.unsqueeze(-1)], dim=-1
        )
    else:
        dihedral_index = None
        d_nlist_mask_4d = None

    return (
        torch.cat([n2e_index.unsqueeze(-1), n_ext2e_index.unsqueeze(-1)], dim=-1),
        torch.cat(
            [
                n2a_index.unsqueeze(-1),
                eij2a_index.unsqueeze(-1),
                eik2a_index.unsqueeze(-1),
            ],
            dim=-1,
        ),
        dihedral_index,
        a_nlist_mask_3d,
        d_nlist_mask_4d,
    )


class BesselBasis(torch.nn.Module):
    """f : (*, 1) -> (*, bessel_basis_num)."""

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ):
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = torch.FloatTensor(
            [n * math.pi / cutoff_length for n in range(1, bessel_basis_num + 1)]
        )
        if trainable_coeff:
            self.coeffs = torch.nn.Parameter(self.coeffs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.prefactor * torch.sin(self.coeffs * r) / (r + 1e-8)


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(
            start, stop, num_gaussians, device=env.DEVICE, dtype=torch.float32
        )
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialMLP(torch.nn.Module):
    """Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels."""

    def __init__(self, channels_list) -> None:
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(torch.nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break

            modules.append(torch.nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class PolynomialEnvelope(torch.nn.Module):
    """Polynomial envelope function that ensures a smooth cutoff."""

    def __init__(self, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
