# SPDX-License-Identifier: LGPL-3.0-or-later
"""Autograd assembly: graph energy -> force/virial/atom_virial via grad(E, edge_vec).

torch-only. The pure-array scatter (edge_force_virial) is shared with dpmodel;
this module supplies the single backward pass that produces g_e = dE/d(edge_vec).
"""

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    get_deriv_name,
    get_reduce_name,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    edge_force_virial,
)
from deepmd.pt.utils import (
    env,
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


def fit_output_to_model_output_graph(
    fit_ret: dict[str, torch.Tensor],
    fit_output_def: FittingOutputDef,
    edge_vec: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    n_node: torch.Tensor,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Graph analogue of the dense pt_expt ``fit_output_to_model_output``.

    OUTPUT-AGNOSTIC: reduces EVERY reducible fitting output (cast to energy
    precision, summed/averaged over the atom axis) and, for every reducible +
    ``r_differentiable`` output, assembles per-component force / virial /
    (optional) atom-virial from :func:`edge_energy_deriv` (one ``grad`` w.r.t.
    ``edge_vec`` per scalar component, then the shared full-to-``src`` scatter).

    Mirrors the dense :func:`deepmd.pt_expt.model.transform_output.take_deriv`
    output shapes -- ``<var>_derv_r`` is ``(nf, nloc, *shape, 3)``,
    ``<var>_derv_c`` is ``(nf, nloc, *shape, 9)``, ``<var>_derv_c_redu`` is
    ``(nf, *shape, 9)`` -- except the graph is ghost-free so the dense ``nall``
    atom axis collapses to ``nloc`` LOCAL atoms.

    Parameters
    ----------
    fit_ret
        Raw rectangular fitting output, ``(nf, nloc, *shape)`` per key.
    fit_output_def
        The fitting output definition.
    edge_vec
        (E, 3) edge vectors; MUST be the autograd leaf of ``fit_ret``.
    edge_index
        (2, E) ``[src, dst]`` edge endpoints (flat local indices).
    edge_mask
        (E,) valid-edge mask.
    n_node
        (nf,) per-frame local atom counts.
    do_atomic_virial
        Whether to also assemble the per-atom virial ``<var>_derv_c``.
    create_graph
        Whether the backward retains a graph (training).
    mask
        (nf, nloc) realness mask; used only for intensive-output reduction.
    """
    redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
    nf = int(n_node.shape[0])
    # N == sum(n_node) == nf * nloc here (rectangular carry-all graph).
    nloc = int(fit_ret[next(iter(fit_ret))].shape[1])
    model_ret: dict[str, torch.Tensor] = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if not vdef.reducible:
            continue
        kk_redu = get_reduce_name(kk)
        if vdef.intensive:
            if mask is not None:
                model_ret[kk_redu] = torch.sum(
                    vv.to(redu_prec), dim=atom_axis
                ) / torch.sum(mask, dim=-1, keepdim=True)
            else:
                model_ret[kk_redu] = torch.mean(vv.to(redu_prec), dim=atom_axis)
        else:
            model_ret[kk_redu] = torch.sum(vv.to(redu_prec), dim=atom_axis)
        if not vdef.r_differentiable:
            continue
        kk_derv_r, kk_derv_c = get_deriv_name(kk)
        size = 1
        for ii in shap:
            size *= ii
        # split the reduced output into ``size`` per-frame scalar components.
        svv = model_ret[kk_redu].reshape(nf, size)
        ff_list: list[torch.Tensor] = []
        av_list: list[torch.Tensor] = []
        vir_list: list[torch.Tensor] = []
        for c in range(size):
            force, atom_vir, vir = edge_energy_deriv(
                svv[:, c],
                edge_vec,
                edge_index,
                edge_mask,
                n_node,
                do_atomic_virial=(vdef.c_differentiable and do_atomic_virial),
                create_graph=create_graph,
            )
            # force (N, 3) -> (nf, nloc, 1, 3)
            ff_list.append(force.reshape(nf, nloc, 1, 3))
            if vdef.c_differentiable:
                # virial (nf, 3, 3) -> (nf, 1, 9)
                vir_list.append(vir.reshape(nf, 1, 9))
                if do_atomic_virial:
                    assert atom_vir is not None
                    # atom_virial (N, 3, 3) -> (nf, nloc, 1, 9)
                    av_list.append(atom_vir.reshape(nf, nloc, 1, 9))
        # (nf, nloc, size, 3) -> (nf, nloc, *shape, 3)
        model_ret[kk_derv_r] = torch.cat(ff_list, dim=-2).reshape([nf, nloc, *shap, 3])
        if vdef.c_differentiable:
            # (nf, size, 9) -> (nf, *shape, 9)
            model_ret[kk_derv_c + "_redu"] = torch.cat(vir_list, dim=-2).reshape(
                [nf, *shap, 9]
            )
            if do_atomic_virial:
                # (nf, nloc, size, 9) -> (nf, nloc, *shape, 9)
                model_ret[kk_derv_c] = torch.cat(av_list, dim=-2).reshape(
                    [nf, nloc, *shap, 9]
                )
    return model_ret
