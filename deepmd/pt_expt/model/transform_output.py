# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_reduce_name,
)
from deepmd.pt_expt.utils import (
    env,
)


def atomic_virial_corr(
    extended_coord: torch.Tensor,
    atom_energy: torch.Tensor,
) -> torch.Tensor:
    nall = extended_coord.shape[1]
    nf = extended_coord.shape[0]
    nloc = atom_energy.shape[1]
    coord, _ = torch.split(extended_coord, [nloc, nall - nloc], dim=1)
    # no derivative with respect to the loc coord.
    coord = coord.detach()
    ce = coord * atom_energy
    sumce = torch.sum(ce, dim=1)  # [nf, 3]

    # Use vmap to batch the 3 backward passes (one per spatial component)
    basis = torch.eye(3, dtype=sumce.dtype, device=sumce.device)  # [3, 3]
    basis = basis.unsqueeze(1).expand(3, nf, 3)  # [3, nf, 3]

    def grad_fn(grad_output: torch.Tensor) -> torch.Tensor:
        result = torch.autograd.grad(
            [sumce],
            [extended_coord],
            grad_outputs=[grad_output],
            create_graph=False,
            retain_graph=True,
        )[0]
        assert result is not None
        return result

    # [3, nf, nall, 3] — batched over the 3 spatial components
    extended_virial_corr = torch.vmap(grad_fn)(basis)
    # [3, nf, nall, 3] -> [nf, nall, 3, 3]
    return extended_virial_corr.permute(1, 2, 3, 0)


def task_deriv_one(
    atom_energy: torch.Tensor,
    energy: torch.Tensor,
    extended_coord: torch.Tensor,
    do_virial: bool = True,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    faked_grad = torch.ones_like(energy)
    lst: list[torch.Tensor | None] = [faked_grad]
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
                    model_ret[kk_derv_c] = dc
                    model_ret[kk_derv_c + "_redu"] = torch.sum(
                        model_ret[kk_derv_c].to(redu_prec), dim=1
                    )
    return model_ret
