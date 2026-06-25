# SPDX-License-Identifier: LGPL-3.0-or-later
"""Autograd assembly: graph energy -> force/virial/atom_virial via grad(E, edge_vec).

torch-only. The pure-array scatter (edge_force_virial) is shared with dpmodel;
this module supplies the single backward pass that produces g_e = dE/d(edge_vec).
"""

import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    edge_force_virial,
)


def edge_energy_deriv(
    energy: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    n_node: torch.Tensor,
    do_atomic_virial: bool = False,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return (force, atom_virial_or_None, virial) from a graph energy.

    g_e = dE/d(edge_vec) via one torch.autograd.grad, then the shared
    edge_force_virial scatter. ``virial`` (per-frame) is always computed;
    ``atom_virial`` is materialized only when do_atomic_virial=True.
    """
    (g_e,) = torch.autograd.grad(
        energy.sum() if energy.dim() else energy,
        edge_vec,
        create_graph=create_graph,
        retain_graph=True,
    )
    force, atom_virial, virial = edge_force_virial(
        g_e, edge_vec, edge_index, edge_mask, n_node
    )
    return force, (atom_virial if do_atomic_virial else None), virial
