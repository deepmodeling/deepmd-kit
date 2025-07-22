# SPDX-License-Identifier: LGPL-3.0-or-later

import paddle

from deepmd.dpmodel import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_reduce_name,
)
from deepmd.pd.utils import (
    decomp,
    env,
)


def atomic_virial_corr(
    extended_coord: paddle.Tensor,
    atom_energy: paddle.Tensor,
):
    nall = extended_coord.shape[1]
    nloc = atom_energy.shape[1]
    coord, _ = paddle.split(extended_coord, [nloc, nall - nloc], axis=1)
    # no derivative with respect to the loc coord.
    coord = coord.detach()
    ce = coord * atom_energy
    sumce0, sumce1, sumce2 = paddle.split(paddle.sum(ce, axis=1), [1, 1, 1], axis=-1)
    # faked_grad = paddle.ones_like(sumce0)
    extended_virial_corr0 = paddle.autograd.grad(
        [sumce0],
        [extended_coord],
        # grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr0 is not None
    extended_virial_corr1 = paddle.autograd.grad(
        [sumce1],
        [extended_coord],
        # grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr1 is not None
    extended_virial_corr2 = paddle.autograd.grad(
        [sumce2],
        [extended_coord],
        # grad_outputs=lst,
        create_graph=False,
        retain_graph=True,
    )[0]
    assert extended_virial_corr2 is not None
    extended_virial_corr = paddle.concat(
        [
            extended_virial_corr0.unsqueeze(-1),
            extended_virial_corr1.unsqueeze(-1),
            extended_virial_corr2.unsqueeze(-1),
        ],
        axis=-1,
    )
    return extended_virial_corr


def task_deriv_one(
    atom_energy: paddle.Tensor,
    energy: paddle.Tensor,
    extended_coord: paddle.Tensor,
    do_virial: bool = True,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
):
    # faked_grad = paddle.ones_like(energy)
    # lst = paddle.jit.annotate(List[Optional[paddle.Tensor]], [faked_grad])
    extended_force = paddle.autograd.grad(
        [energy],
        [extended_coord],
        # grad_outputs=lst,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    assert extended_force is not None
    extended_force = -extended_force
    if do_virial:
        extended_virial = extended_force.unsqueeze(-1) @ extended_coord.unsqueeze(-2)
        # the correction sums to zero, which does not contribute to global virial
        if do_atomic_virial:
            extended_virial_corr = atomic_virial_corr(extended_coord, atom_energy)
            extended_virial = extended_virial + extended_virial_corr
        # to [...,3,3] -> [...,9]
        extended_virial = extended_virial.reshape(
            [*list(extended_virial.shape[:-2]), 9]
        )
    else:
        extended_virial = None
    return extended_force, extended_virial


def get_leading_dims(
    vv: paddle.Tensor,
    vdef: OutputVariableDef,
):
    """Get the dimensions of nf x nloc."""
    vshape = vv.shape
    return list(vshape[: (len(vshape) - len(vdef.shape))])


def take_deriv(
    vv: paddle.Tensor,
    svv: paddle.Tensor,
    vdef: OutputVariableDef,
    coord_ext: paddle.Tensor,
    do_virial: bool = False,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
):
    size = 1
    for ii in vdef.shape:
        size *= ii
    vv1 = vv.reshape(list(get_leading_dims(vv, vdef)) + [size])  # noqa: RUF005
    svv1 = svv.reshape(list(get_leading_dims(svv, vdef)) + [size])  # noqa: RUF005
    split_vv1 = paddle.split(vv1, [1] * size, axis=-1)
    split_svv1 = paddle.split(svv1, [1] * size, axis=-1)
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
    ff = paddle.concat(split_ff, axis=-2).reshape(out_lead_shape + [3])  # noqa: RUF005
    if do_virial:
        avir = paddle.concat(split_avir, axis=-2).reshape(out_lead_shape + [9])  # noqa: RUF005
    else:
        avir = None
    return ff, avir


def fit_output_to_model_output(
    fit_ret: dict[str, paddle.Tensor],
    fit_output_def: FittingOutputDef,
    coord_ext: paddle.Tensor,
    do_atomic_virial: bool = False,
    create_graph: bool = True,
) -> dict[str, paddle.Tensor]:
    """Transform the output of the fitting network to
    the model output.

    """
    redu_prec = env.GLOBAL_PD_ENER_FLOAT_PRECISION
    model_ret = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            if vdef.intensive:
                model_ret[kk_redu] = paddle.mean(vv.astype(redu_prec), axis=atom_axis)
            else:
                model_ret[kk_redu] = paddle.sum(vv.astype(redu_prec), axis=atom_axis)
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
                    model_ret[kk_derv_c + "_redu"] = paddle.sum(
                        model_ret[kk_derv_c].astype(redu_prec), axis=1
                    )
    return model_ret


def communicate_extended_output(
    model_ret: dict[str, paddle.Tensor],
    model_output_def: ModelOutputDef,
    mapping: paddle.Tensor,  # nf x nloc
    do_atomic_virial: bool = False,
) -> dict[str, paddle.Tensor]:
    """Transform the output of the model network defined on
    local and ghost (extended) atoms to local atoms.

    """
    redu_prec = env.GLOBAL_PD_ENER_FLOAT_PRECISION
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
                mapping = mapping.reshape(mldims + [1] * len(derv_r_ext_dims)).expand(
                    [-1] * len(mldims) + derv_r_ext_dims
                )
                force = paddle.zeros(vldims + derv_r_ext_dims, dtype=vv.dtype).to(
                    device=vv.place
                )
                # nf x nloc x nvar x 3
                new_ret[kk_derv_r] = decomp.scatter_reduce(
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
                mapping = paddle.tile(
                    mapping,
                    [1] * (len(mldims) + len(vdef.shape)) + [3],
                )
                virial = paddle.zeros(vldims + derv_c_ext_dims, dtype=vv.dtype).to(
                    device=vv.place
                )
                # nf x nloc x nvar x 9
                new_ret[kk_derv_c] = decomp.scatter_reduce(
                    virial,
                    1,
                    index=mapping,
                    src=model_ret[kk_derv_c],
                    reduce="sum",
                )
                new_ret[kk_derv_c + "_redu"] = paddle.sum(
                    new_ret[kk_derv_c].to(redu_prec), axis=1
                )
                if not do_atomic_virial:
                    # pop atomic virial, because it is not correctly calculated.
                    new_ret.pop(kk_derv_c)
    return new_ret
