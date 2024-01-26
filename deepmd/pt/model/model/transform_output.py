# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

import torch
from deepmd.model_format import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_reduce_name,
)


def atomic_virial_corr(
    extended_coord: torch.Tensor,
    atom_energy: torch.Tensor,
):
    nall = extended_coord.shape[1]
    nloc = atom_energy.shape[1]
    coord, _ = torch.split(extended_coord, [nloc, nall - nloc], dim=1)
    # no derivative with respect to the loc coord.
    coord = coord.detach()
    ce = coord * atom_energy
    sumce0, sumce1, sumce2 = torch.split(torch.sum(ce, dim=1), [1, 1, 1], dim=-1)
    faked_grad = torch.ones_like(sumce0)
    lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
    extended_virial_corr0 = torch.autograd.grad(
        [sumce0], [extended_coord], grad_outputs=lst, create_graph=True
    )[0]
    assert extended_virial_corr0 is not None
    extended_virial_corr1 = torch.autograd.grad(
        [sumce1], [extended_coord], grad_outputs=lst, create_graph=True
    )[0]
    assert extended_virial_corr1 is not None
    extended_virial_corr2 = torch.autograd.grad(
        [sumce2], [extended_coord], grad_outputs=lst, create_graph=True
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
    do_atomic_virial: bool = False,
):
    faked_grad = torch.ones_like(energy)
    lst = torch.jit.annotate(List[Optional[torch.Tensor]], [faked_grad])
    extended_force = torch.autograd.grad(
        [energy], [extended_coord], grad_outputs=lst, create_graph=True
    )[0]
    assert extended_force is not None
    extended_force = -extended_force
    extended_virial = extended_force.unsqueeze(-1) @ extended_coord.unsqueeze(-2)
    # the correction sums to zero, which does not contribute to global virial
    if do_atomic_virial:
        extended_virial_corr = atomic_virial_corr(extended_coord, atom_energy)
        extended_virial = extended_virial + extended_virial_corr
    return extended_force, extended_virial


def get_leading_dims(
    vv: torch.Tensor,
    vdef: OutputVariableDef,
):
    """Get the dimensions of nf x nloc."""
    vshape = vv.shape
    return list(vshape[: (len(vshape) - len(vdef.shape))])


def get_atom_axis(
    vdef: torch.Tensor,
):
    """Get the axis of atoms."""
    atom_axis = -(len(vdef.shape) + 1)
    return atom_axis


def take_deriv(
    vv: torch.Tensor,
    svv: torch.Tensor,
    vdef: OutputVariableDef,
    coord_ext: torch.Tensor,
    do_atomic_virial: bool = False,
):
    size = 1
    for ii in vdef.shape:
        size *= ii
    vv1 = vv.view(list(get_leading_dims(vv, vdef)) + [size])  # noqa: RUF005
    svv1 = svv.view(list(get_leading_dims(svv, vdef)) + [size])  # noqa: RUF005
    split_vv1 = torch.split(vv1, [1] * size, dim=-1)
    split_svv1 = torch.split(svv1, [1] * size, dim=-1)
    split_ff, split_avir = [], []
    for vvi, svvi in zip(split_vv1, split_svv1):
        # nf x nloc x 3, nf x nloc x 3 x 3
        ffi, aviri = task_deriv_one(
            vvi, svvi, coord_ext, do_atomic_virial=do_atomic_virial
        )
        # nf x nloc x 1 x 3, nf x nloc x 1 x 3 x 3
        ffi = ffi.unsqueeze(-2)
        aviri = aviri.unsqueeze(-3)
        split_ff.append(ffi)
        split_avir.append(aviri)
    # nf x nloc x v_dim x 3, nf x nloc x v_dim x 3 x 3
    ff = torch.concat(split_ff, dim=-2)
    avir = torch.concat(split_avir, dim=-3)
    return ff, avir


def fit_output_to_model_output(
    fit_ret: Dict[str, torch.Tensor],
    fit_output_def: FittingOutputDef,
    coord_ext: torch.Tensor,
    do_atomic_virial: bool = False,
) -> Dict[str, torch.Tensor]:
    """Transform the output of the fitting network to
    the model output.

    """
    model_ret = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if vdef.reduciable:
            kk_redu = get_reduce_name(kk)
            model_ret[kk_redu] = torch.sum(vv, dim=atom_axis)
            if vdef.differentiable:
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                dr, dc = take_deriv(
                    vv,
                    model_ret[kk_redu],
                    vdef,
                    coord_ext,
                    do_atomic_virial=do_atomic_virial,
                )
                model_ret[kk_derv_r] = dr
                model_ret[kk_derv_c] = dc
    return model_ret


def communicate_extended_output(
    model_ret: Dict[str, torch.Tensor],
    model_output_def: ModelOutputDef,
    mapping: torch.Tensor,  # nf x nloc
    do_atomic_virial: bool = False,
) -> Dict[str, torch.Tensor]:
    """Transform the output of the model network defined on
    local and ghost (extended) atoms to local atoms.

    """
    new_ret = {}
    for kk in model_output_def.keys_outp():
        vv = model_ret[kk]
        vdef = model_output_def[kk]
        new_ret[kk] = vv
        if vdef.reduciable:
            kk_redu = get_reduce_name(kk)
            new_ret[kk_redu] = model_ret[kk_redu]
            if vdef.differentiable:
                # nf x nloc
                vldims = get_leading_dims(vv, vdef)
                # nf x nall
                mldims = list(mapping.shape)
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                # vdim x 3
                derv_r_ext_dims = list(vdef.shape) + [3]  # noqa:RUF005
                mapping = mapping.view(mldims + [1] * len(derv_r_ext_dims)).expand(
                    [-1] * len(mldims) + derv_r_ext_dims
                )
                force = torch.zeros(
                    vldims + derv_r_ext_dims, dtype=vv.dtype, device=vv.device
                )
                # nf x nloc x 1 x 3
                new_ret[kk_derv_r] = torch.scatter_reduce(
                    force,
                    1,
                    index=mapping,
                    src=model_ret[kk_derv_r],
                    reduce="sum",
                )
                mapping = mapping.unsqueeze(-1).expand(
                    [-1] * (len(mldims) + len(derv_r_ext_dims)) + [3]
                )
                virial = torch.zeros(
                    vldims + derv_r_ext_dims + [3], dtype=vv.dtype, device=vv.device
                )
                # nf x nloc x 1 x 3
                new_ret[kk_derv_c] = torch.scatter_reduce(
                    virial,
                    1,
                    index=mapping,
                    src=model_ret[kk_derv_c],
                    reduce="sum",
                )
                new_ret[kk_derv_c + "_redu"] = torch.sum(new_ret[kk_derv_c], dim=1)
                if not do_atomic_virial:
                    # pop atomic virial, because it is not correctly calculated.
                    new_ret.pop(kk_derv_c)
    return new_ret
