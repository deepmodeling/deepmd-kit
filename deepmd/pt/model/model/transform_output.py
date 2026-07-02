# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_reduce_name,
)
from deepmd.kernels.utils import (
    triton_infer_level,
)
from deepmd.pt.utils import (
    env,
)


def atomic_virial_corr(
    extended_coord: torch.Tensor,
    atom_energy: torch.Tensor,
) -> torch.Tensor:
    nall = extended_coord.shape[1]
    nloc = atom_energy.shape[1]
    coord, _ = torch.split(extended_coord, [nloc, nall - nloc], dim=1)
    # no derivative with respect to the loc coord.
    coord = coord.detach()
    ce = coord * atom_energy
    sumce0, sumce1, sumce2 = torch.split(torch.sum(ce, dim=1), [1, 1, 1], dim=-1)
    faked_grad = torch.ones_like(sumce0)
    lst = torch.jit.annotate(list[torch.Tensor | None], [faked_grad])
    extended_virial_corr0 = torch.autograd.grad(
        [sumce0],
        [extended_coord],
        grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr0 is not None
    extended_virial_corr1 = torch.autograd.grad(
        [sumce1],
        [extended_coord],
        grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr1 is not None
    extended_virial_corr2 = torch.autograd.grad(
        [sumce2],
        [extended_coord],
        grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr2 is not None
    extended_virial_corr = torch.concat(
        [
            extended_virial_corr0.unsqueeze(-1),
            extended_virial_corr1.unsqueeze(-1),
            extended_virial_corr2.unsqueeze(-1),
        ],
        dim=-1,
    )
    return extended_virial_corr


def task_deriv_one(
    atom_energy: torch.Tensor,
    energy: torch.Tensor,
    extended_coord: torch.Tensor,
    do_virial: bool = True,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    faked_grad = torch.ones_like(energy)
    lst = torch.jit.annotate(list[torch.Tensor | None], [faked_grad])
    extended_force = torch.autograd.grad(
        [energy],
        [extended_coord],
        grad_outputs=lst,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    assert extended_force is not None
    extended_force = -extended_force
    if do_virial:
        extended_virial = torch.einsum(
            "...ik,...ij->...ikj", extended_force, extended_coord
        )
        # the correction sums to zero, which does not contribute to global virial
        if do_atomic_virial:
            extended_virial_corr = atomic_virial_corr(extended_coord, atom_energy)
            extended_virial = extended_virial + extended_virial_corr
        # to [...,3,3] -> [...,9]
        extended_virial = extended_virial.view(list(extended_virial.shape[:-2]) + [9])  # noqa:RUF005
    else:
        extended_virial = None
    return extended_force, extended_virial


def get_leading_dims(
    vv: torch.Tensor,
    vdef: OutputVariableDef,
) -> list[int]:
    """Get the dimensions of nf x nloc."""
    vshape = vv.shape
    return list(vshape[: (len(vshape) - len(vdef.shape))])


def take_deriv(
    vv: torch.Tensor,
    svv: torch.Tensor,
    vdef: OutputVariableDef,
    coord_ext: torch.Tensor,
    do_virial: bool = False,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    size = 1
    for ii in vdef.shape:
        size *= ii
    vv1 = vv.view(list(get_leading_dims(vv, vdef)) + [size])  # noqa: RUF005
    svv1 = svv.view(list(get_leading_dims(svv, vdef)) + [size])  # noqa: RUF005
    split_vv1 = torch.split(vv1, [1] * size, dim=-1)
    split_svv1 = torch.split(svv1, [1] * size, dim=-1)
    split_ff, split_avir = [], []
    for vvi, svvi in zip(split_vv1, split_svv1):
        # nf x nloc x 3, nf x nloc x 9
        ffi, aviri = task_deriv_one(
            vvi,
            svvi,
            coord_ext,
            do_virial=do_virial,
            do_atomic_virial=do_atomic_virial,
            create_graph=create_graph,
        )
        # nf x nloc x 1 x 3, nf x nloc x 1 x 9
        ffi = ffi.unsqueeze(-2)
        split_ff.append(ffi)
        if do_virial:
            assert aviri is not None
            aviri = aviri.unsqueeze(-2)
            split_avir.append(aviri)
    # nf x nall x v_dim x 3, nf x nall x v_dim x 9
    out_lead_shape = list(coord_ext.shape[:-1]) + vdef.shape
    ff = torch.concat(split_ff, dim=-2).view(out_lead_shape + [3])  # noqa: RUF005
    if do_virial:
        avir = torch.concat(split_avir, dim=-2).view(out_lead_shape + [9])  # noqa: RUF005
    else:
        avir = None
    return ff, avir


def fit_output_to_model_output(
    fit_ret: dict[str, torch.Tensor],
    fit_output_def: FittingOutputDef,
    coord_ext: torch.Tensor,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
    mask: torch.Tensor | None = None,
    extended_coord_corr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Transform the output of the fitting network to
    the model output.

    """
    redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
    model_ret = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if vdef.reducible:
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
            if vdef.r_differentiable:
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                dr, dc = take_deriv(
                    vv,
                    model_ret[kk_redu],
                    vdef,
                    coord_ext,
                    do_virial=vdef.c_differentiable,
                    do_atomic_virial=do_atomic_virial,
                    create_graph=create_graph,
                )
                model_ret[kk_derv_r] = dr
                if vdef.c_differentiable:
                    assert dc is not None
                    if extended_coord_corr is not None:
                        dc_corr = (
                            dr.squeeze(-2).unsqueeze(-1)
                            @ extended_coord_corr.unsqueeze(-2).to(dr.dtype)
                        ).view(list(dc.shape[:-2]) + [1, 9])  # noqa: RUF005
                        dc = dc + dc_corr
                    model_ret[kk_derv_c] = dc
                    model_ret[kk_derv_c + "_redu"] = torch.sum(
                        model_ret[kk_derv_c].to(redu_prec), dim=1
                    )
    return model_ret


def edge_energy_deriv(
    energy_redu: torch.Tensor,
    edge_vec: torch.Tensor,
    src_ext: torch.Tensor,
    dst_ext: torch.Tensor,
    edge_mask: torch.Tensor,
    nf: int,
    nall: int,
    create_graph: bool,
    extended_coord_corr: torch.Tensor | None = None,
    spin_leaf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Assemble extended force, virial and atomic virial from edge gradients.

    The energy depends on coordinates only through the per-edge displacement
    vectors ``edge_vec``.  A single ``autograd.grad`` produces the per-edge
    gradient ``g_e = dE / d(edge_vec_e)``; force, global virial and per-atom
    virial are assembled from it with explicit scatter and outer-product ops.

    With edge ``e`` running from receiver ``dst(e)`` to sender ``src(e)`` and
    ``edge_vec_e = r_{src(e)} - r_{dst(e)}``, the chain rule
    ``d(edge_vec_e)/dr_k = (delta_{k,src} - delta_{k,dst}) I`` gives the
    conservative force and the pairwise virial::

        F_k = sum_{dst(e)=k} g_e - sum_{src(e)=k} g_e
        W   = - sum_e g_e (x) edge_vec_e

    ``src_ext`` and ``dst_ext`` index the flattened extended space
    ``[0, nf * nall)``, so the scatter produces per-ghost extended tensors
    consumed by ``communicate_extended_output`` and the lower interface.

    ``edge_vec`` carries the coordinate precision (``GLOBAL_PT_FLOAT_PRECISION``),
    so ``g`` and the assembled force / virial share that dtype -- the dtype the
    ``communicate_extended_output`` scatter buffers and the reduced energy
    expect.  The reduced global virial is summed in
    ``GLOBAL_PT_ENER_FLOAT_PRECISION``.

    Parameters
    ----------
    energy_redu
        Reduced per-frame energy with shape ``(nf, 1)``.
    edge_vec
        Per-edge displacement leaf with shape ``(E, 3)`` carrying ``requires_grad``.
    src_ext, dst_ext
        Sender / receiver indices into the flattened extended space, each with
        shape ``(E,)``.
    edge_mask
        Boolean validity mask with shape ``(E,)``.
    nf, nall
        Frame count and extended-atom count.
    create_graph
        Keep the first-derivative graph alive so the force-loss second backward
        can reach the parameters.
    extended_coord_corr
        Optional spin virtual-displacement correction with shape
        ``(nf, nall, 3)``; adds ``force (x) coord_corr`` per extended atom.
    spin_leaf
        Optional per-atom spin leaf with shape ``(nf, nloc, 3)`` for the native
        spin scheme. When provided, the energy is also differentiated with
        respect to it in the same backward, so the magnetic force shares the
        first-derivative graph used by the force-loss second backward.

    Returns
    -------
    energy_derv_r
        Extended force with shape ``(nf, nall, 1, 3)``.
    energy_derv_c
        Extended per-atom virial with shape ``(nf, nall, 1, 9)``, split
        symmetrically between the two endpoints of each edge.
    energy_derv_c_redu
        Reduced global virial with shape ``(nf, 1, 9)``.
    energy_derv_r_mag
        Magnetic force ``-dE/dspin`` with shape ``(nf, nloc, 1, 3)`` when
        ``spin_leaf`` is provided, otherwise ``None``.
    """
    grad_inputs = [edge_vec] if spin_leaf is None else [edge_vec, spin_leaf]
    grads = torch.autograd.grad(
        [energy_redu],
        grad_inputs,
        grad_outputs=[torch.ones_like(energy_redu)],
        create_graph=create_graph,
        retain_graph=True,
    )
    g = grads[0]
    # Padded edges carry no energy contribution, so their gradient is zero;
    # mask defensively before the scatter.
    g = torch.where(edge_mask.unsqueeze(-1), g, torch.zeros_like(g))

    n_ext = nf * nall
    if triton_infer_level() >= 1 and not create_graph and g.is_cuda:
        # Inference: assemble force and per-atom virial with two CSR segment
        # reductions instead of four ``index_add`` scatters (which serialize
        # on the colliding edges of each atom) and a materialized ``(E, 9)``
        # outer product. The extended indices carry no ordering guarantee, so
        # the topology is sorted here; these integer ops trace as ordinary
        # aten nodes under ``make_fx``.
        from deepmd.kernels.triton.sezm.force_assembly import (
            edge_force_assembly,
        )

        dst_order = torch.argsort(dst_ext)
        src_order = torch.argsort(src_ext)
        boundaries = torch.arange(n_ext + 1, device=g.device, dtype=dst_ext.dtype)
        dst_row_ptr = torch.searchsorted(dst_ext.index_select(0, dst_order), boundaries)
        src_row_ptr = torch.searchsorted(src_ext.index_select(0, src_order), boundaries)
        force_flat, av_flat = edge_force_assembly(
            g.contiguous(),
            edge_vec.detach().contiguous(),
            dst_order,
            dst_row_ptr,
            src_order,
            src_row_ptr,
        )
        extended_force = force_flat.view(nf, nall, 3)
        extended_virial = av_flat.view(nf, nall, 9)
    else:
        # Force: F_k = sum_{dst=k} g_e - sum_{src=k} g_e.
        force_flat = torch.zeros(n_ext, 3, dtype=g.dtype, device=g.device)
        force_flat = force_flat.index_add(0, dst_ext, g)
        force_flat = force_flat.index_add(0, src_ext, -g)
        extended_force = force_flat.view(nf, nall, 3)

        # Per-edge virial outer product w_e[k, j] = -g_e^k * edge_vec_e^j,
        # flattened to 9 with (force component k, coordinate component j)
        # ordering.
        w_edge = -torch.einsum("ek,ej->ekj", g, edge_vec).reshape(-1, 9)
        # Atomic virial: split each per-edge tensor symmetrically between
        # endpoints.
        half_w = 0.5 * w_edge
        av_flat = torch.zeros(n_ext, 9, dtype=g.dtype, device=g.device)
        av_flat = av_flat.index_add(0, dst_ext, half_w)
        av_flat = av_flat.index_add(0, src_ext, half_w)
        extended_virial = av_flat.view(nf, nall, 9)

    if extended_coord_corr is not None:
        # Spin: the virtual-atom displacement adds force (x) coord_corr per atom.
        corr = (
            extended_force.unsqueeze(-1)
            @ extended_coord_corr.unsqueeze(-2).to(extended_force.dtype)
        ).reshape(nf, nall, 9)
        extended_virial = extended_virial + corr

    energy_derv_r = extended_force.unsqueeze(-2)
    energy_derv_c = extended_virial.unsqueeze(-2)
    energy_derv_c_redu = energy_derv_c.to(env.GLOBAL_PT_ENER_FLOAT_PRECISION).sum(dim=1)

    # Magnetic force is the negative spin gradient, matching the dataset
    # ``force_mag = -dE/dspin`` convention (the virtual-atom scheme reaches the
    # same quantity through ``F_virtual * virtual_scale``).
    energy_derv_r_mag: torch.Tensor | None = None
    if spin_leaf is not None:
        energy_derv_r_mag = (-grads[1]).unsqueeze(-2)
    return energy_derv_r, energy_derv_c, energy_derv_c_redu, energy_derv_r_mag


def communicate_extended_output(
    model_ret: dict[str, torch.Tensor],
    model_output_def: ModelOutputDef,
    mapping: torch.Tensor,  # nf x nloc
    do_atomic_virial: bool = False,
) -> dict[str, torch.Tensor]:
    """Transform the output of the model network defined on
    local and ghost (extended) atoms to local atoms.

    """
    redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
    new_ret = {}
    for kk in model_output_def.keys_outp():
        vv = model_ret[kk]
        vdef = model_output_def[kk]
        new_ret[kk] = vv
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            new_ret[kk_redu] = model_ret[kk_redu]
            # nf x nloc
            vldims = get_leading_dims(vv, vdef)
            # nf x nall
            mldims = list(mapping.shape)
            kk_derv_r, kk_derv_c = get_deriv_name(kk)
            if vdef.r_differentiable:
                # vdim x 3
                derv_r_ext_dims = list(vdef.shape) + [3]  # noqa:RUF005
                mapping = mapping.view(mldims + [1] * len(derv_r_ext_dims)).expand(
                    [-1] * len(mldims) + derv_r_ext_dims
                )
                force = torch.zeros(
                    vldims + derv_r_ext_dims, dtype=vv.dtype, device=vv.device
                )
                # nf x nloc x nvar x 3
                new_ret[kk_derv_r] = torch.scatter_reduce(
                    force,
                    1,
                    index=mapping,
                    src=model_ret[kk_derv_r],
                    reduce="sum",
                )
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                derv_c_ext_dims = list(vdef.shape) + [9]  # noqa:RUF005
                # nf x nloc x nvar x 3 -> nf x nloc x nvar x 9
                mapping = torch.tile(
                    mapping,
                    [1] * (len(mldims) + len(vdef.shape)) + [3],
                )
                virial = torch.zeros(
                    vldims + derv_c_ext_dims, dtype=vv.dtype, device=vv.device
                )
                # nf x nloc x nvar x 9
                new_ret[kk_derv_c] = torch.scatter_reduce(
                    virial,
                    1,
                    index=mapping,
                    src=model_ret[kk_derv_c],
                    reduce="sum",
                )
                new_ret[kk_derv_c + "_redu"] = torch.sum(
                    new_ret[kk_derv_c].to(redu_prec), dim=1
                )
                if not do_atomic_virial:
                    # pop atomic virial, because it is not correctly calculated.
                    new_ret.pop(kk_derv_c)
    return new_ret
