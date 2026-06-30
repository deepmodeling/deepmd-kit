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
    NeighborGraph,
    edge_force_virial,
    frame_id_from_n_node,
    segment_sum,
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
    node_capacity: int | None = None,
    *,
    do_atomic_virial: bool = False,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return (force, atom_virial_or_None, virial) from a graph energy.

    g_e = dE/d(edge_vec) via one torch.autograd.grad, then the shared
    edge_force_virial scatter.

    Parameters
    ----------
    energy
        the reduced per-frame energy to differentiate. ``(nf,)`` (or scalar).
    edge_vec
        (E, 3) per-edge displacement; the autograd leaf of ``energy``.
    edge_index
        (2, E) ``[src, dst]`` edge endpoints.
    edge_mask
        (E,) valid-edge mask.
    n_node
        (nf,) per-frame node counts.
    node_capacity
        Static node-axis size ``N``.  ``None`` (eager default) falls back to
        ``int(n_node.sum())``.  Pass a static value (e.g. ``atype.shape[0]``)
        to keep this function trace-safe under ``make_fx``/``torch.export``.
    do_atomic_virial
        whether to materialize the per-atom virial (else ``None`` is returned).
    create_graph
        whether the backward retains a graph (training, for second-order grad).

    Returns
    -------
    force
        (N, 3) per-node force.
    atom_virial
        (N, 3, 3) per-node virial when ``do_atomic_virial`` else ``None``.
    virial
        (nf, 3, 3) per-frame virial (always computed).
    """
    (g_e,) = torch.autograd.grad(
        energy.sum() if energy.dim() else energy,
        edge_vec,
        create_graph=create_graph,
        retain_graph=True,
    )
    force, atom_virial, virial = edge_force_virial(
        g_e, edge_vec, edge_index, edge_mask, n_node, node_capacity=node_capacity
    )
    return force, (atom_virial if do_atomic_virial else None), virial


def fit_output_to_model_output_graph(
    fit_ret: dict[str, torch.Tensor],
    fit_output_def: FittingOutputDef,
    graph: NeighborGraph,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Graph analogue of the dense pt_expt ``fit_output_to_model_output``.

    OUTPUT-AGNOSTIC: reduces EVERY reducible fitting output (cast to energy
    precision, summed/averaged per frame via ``segment_sum`` over ``frame_id``)
    and, for every reducible + ``r_differentiable`` output, assembles
    per-component force / virial / (optional) atom-virial from
    :func:`edge_energy_deriv` (one ``grad`` w.r.t. ``edge_vec`` per scalar
    component, then the shared full-to-``src`` scatter).

    All per-atom outputs stay FLAT with leading dimension ``N = sum(n_node)``:
    ``<var>`` is ``(N, *shape)``, ``<var>_derv_r`` is ``(N, *shape, 3)``,
    ``<var>_derv_c`` is ``(N, *shape, 9)``.  Per-frame reductions have leading
    dimension ``nf``: ``<var>_redu`` is ``(nf, *shape)``,
    ``<var>_derv_c_redu`` is ``(nf, *shape, 9)``.

    Parameters
    ----------
    fit_ret
        Raw flat fitting output, ``(N, *shape)`` per key (``N = sum(n_node)``).
    fit_output_def
        The fitting output definition.
    graph
        the :class:`~deepmd.dpmodel.utils.neighbor_graph.NeighborGraph`. Its
        ``edge_vec`` MUST be the autograd leaf for ``fit_ret`` (the force backward
        differentiates the reduced energy w.r.t. it); ``edge_index``/``edge_mask``
        define the scatter, ``n_node`` the node->frame map.
    do_atomic_virial
        Whether to also assemble the per-atom virial ``<var>_derv_c``.
    create_graph
        Whether the backward retains a graph (training).
    mask
        (N,) flat realness mask; used only for intensive-output reduction.

    Returns
    -------
    model_ret
        ``fit_ret`` plus, for each reducible key, the per-frame reduction
        ``<var>_redu`` ``(nf, *shape)`` and -- for ``r_differentiable`` keys --
        the FLAT per-atom force ``<var>_derv_r`` ``(N, *shape, 3)``, the
        per-frame virial ``<var>_derv_c_redu`` ``(nf, *shape, 9)``, and (when
        ``do_atomic_virial``) the per-atom virial ``<var>_derv_c``
        ``(N, *shape, 9)``.
    """
    edge_vec = graph.edge_vec
    edge_index = graph.edge_index
    edge_mask = graph.edge_mask
    n_node = graph.n_node
    redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
    # Keep ``nf`` as a (possibly symbolic) shape value: under symbolic make_fx /
    # torch.export ``n_node`` dim-0 is the dynamic frame axis, and ``int()`` on a
    # SymInt SPECIALIZES it -- baking the trace-time frame count into every
    # per-frame reduction (energy_redu / virial) and breaking multi-frame infer.
    nf = n_node.shape[0]
    # Derive N from the fitting output's leading shape rather than int(n_node.sum()).
    # shape attributes are always static Python ints (or SymInts in symbolic-mode
    # tracing) and are trace-safe; reading a tensor VALUE via int() is not.
    N = next(iter(fit_ret.values())).shape[0]
    frame_id = frame_id_from_n_node(
        n_node, n_total=N
    )  # (N,) int64 frame index per atom
    model_ret: dict[str, torch.Tensor] = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        shap = vdef.shape
        if not vdef.reducible:
            continue
        kk_redu = get_reduce_name(kk)
        # segment_sum reduces axis 0 (the flat atom axis) per frame
        vv_e = vv.to(redu_prec)  # (N, *shape)
        redu = segment_sum(vv_e, frame_id, nf)  # (nf, *shape)
        if vdef.intensive:
            if mask is not None:
                # real-atom count per frame: segment_sum of the mask
                cnt = segment_sum(mask.to(redu_prec), frame_id, nf)  # (nf,)
                # broadcast cnt to (nf, 1, ..., 1) to match redu shape
                cnt = cnt.reshape(nf, *([1] * (redu.ndim - 1)))
            else:
                cnt = n_node.to(redu_prec).reshape(nf, *([1] * (redu.ndim - 1)))
            redu = redu / cnt
        model_ret[kk_redu] = redu
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
                node_capacity=N,
                do_atomic_virial=(vdef.c_differentiable and do_atomic_virial),
                create_graph=create_graph,
            )
            # force (N, 3) -> (N, 1, 3)  [flat; caller unravels at I/O boundary]
            ff_list.append(force.reshape(N, 1, 3))
            if vdef.c_differentiable:
                # virial (nf, 3, 3) -> (nf, 1, 9)
                vir_list.append(vir.reshape(nf, 1, 9))
                if do_atomic_virial:
                    assert atom_vir is not None
                    # atom_virial (N, 3, 3) -> (N, 1, 9)  [flat]
                    av_list.append(atom_vir.reshape(N, 1, 9))
        # (N, size, 3) -> (N, *shape, 3)
        model_ret[kk_derv_r] = torch.cat(ff_list, dim=-2).reshape([N, *shap, 3])
        if vdef.c_differentiable:
            # (nf, size, 9) -> (nf, *shape, 9)
            model_ret[kk_derv_c + "_redu"] = torch.cat(vir_list, dim=-2).reshape(
                [nf, *shap, 9]
            )
            if do_atomic_virial:
                # (N, size, 9) -> (N, *shape, 9)
                model_ret[kk_derv_c] = torch.cat(av_list, dim=-2).reshape([N, *shap, 9])
    return model_ret
