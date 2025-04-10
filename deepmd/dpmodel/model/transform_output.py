# SPDX-License-Identifier: LGPL-3.0-or-later

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    xp_scatter_sum,
)
from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_hessian_name,
    get_reduce_name,
)


def fit_output_to_model_output(
    fit_ret: dict[str, np.ndarray],
    fit_output_def: FittingOutputDef,
    coord_ext: np.ndarray,
    do_atomic_virial: bool = False,
) -> dict[str, np.ndarray]:
    """Transform the output of the fitting network to
    the model output.

    """
    xp = array_api_compat.get_namespace(coord_ext)
    model_ret = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            # cast to energy prec before reduction
            model_ret[kk_redu] = xp.sum(
                vv.astype(GLOBAL_ENER_FLOAT_PRECISION), axis=atom_axis
            )
            if vdef.r_differentiable:
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                # name-holders
                model_ret[kk_derv_r] = None
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                model_ret[kk_derv_c] = None
    return model_ret


def get_leading_dims(
    vv: np.ndarray,
    vdef: OutputVariableDef,
):
    """Get the dimensions of nf x nloc.

    Parameters
    ----------
    vv : np.ndarray
        The input array from which to compute the leading dimensions.
    vdef : OutputVariableDef
        The output variable definition containing the shape to exclude from `vv`.

    Returns
    -------
    list
        A list of leading dimensions of `vv`, excluding the last `len(vdef.shape)` dimensions.
    """
    vshape = vv.shape
    return list(vshape[: (len(vshape) - len(vdef.shape))])


def communicate_extended_output(
    model_ret: dict[str, np.ndarray],
    model_output_def: ModelOutputDef,
    mapping: np.ndarray,  # nf x nloc
    do_atomic_virial: bool = False,
) -> dict[str, np.ndarray]:
    """Transform the output of the model network defined on
    local and ghost (extended) atoms to local atoms.

    """
    xp = array_api_compat.get_namespace(mapping)
    mapping_ = mapping
    new_ret = {}
    for kk in model_output_def.keys_outp():
        vv = model_ret[kk]
        vdef = model_output_def[kk]
        new_ret[kk] = vv
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            new_ret[kk_redu] = model_ret[kk_redu]
            kk_derv_r, kk_derv_c = get_deriv_name(kk)
            mldims = list(mapping.shape)
            vldims = get_leading_dims(vv, vdef)
            if vdef.r_differentiable:
                if model_ret[kk_derv_r] is not None:
                    derv_r_ext_dims = list(vdef.shape) + [3]  # noqa:RUF005
                    mapping = xp.reshape(mapping, (mldims + [1] * len(derv_r_ext_dims)))
                    mapping = xp.tile(mapping, [1] * len(mldims) + derv_r_ext_dims)
                    force = xp.zeros(vldims + derv_r_ext_dims, dtype=vv.dtype)
                    force = xp_scatter_sum(
                        force,
                        1,
                        mapping,
                        model_ret[kk_derv_r],
                    )
                    new_ret[kk_derv_r] = force
                else:
                    # name holders
                    new_ret[kk_derv_r] = None
                if vdef.r_hessian:
                    kk_hess = get_hessian_name(kk)
                    if model_ret[kk_hess] is not None:
                        # [nf, *def, nall, 3, nall, 3]
                        hess_ = model_ret[kk_hess]
                        def_ndim = len(vdef.shape)
                        # [nf, nall1, nall2, *def, 3(1), 3(2)]
                        hess_1 = xp.permute_dims(
                            hess_,
                            (
                                0,
                                def_ndim + 1,
                                def_ndim + 3,
                                *range(1, def_ndim + 1),
                                def_ndim + 2,
                                def_ndim + 4,
                            ),
                        )
                        nall = hess_1.shape[1]
                        # (1) -> [nf, nloc1, nall2, *def, 3(1), 3(2)]
                        hessian1 = xp.zeros(
                            [*vldims, nall, *vdef.shape, 3, 3], dtype=vv.dtype
                        )
                        mapping_hess = xp.reshape(
                            mapping_, (mldims + [1] * (len(vdef.shape) + 3))
                        )
                        mapping_hess = xp.tile(
                            mapping_hess,
                            [1] * len(mldims) + [nall, *vdef.shape, 3, 3],
                        )
                        hessian1 = xp_scatter_sum(
                            hessian1,
                            1,
                            mapping_hess,
                            hess_1,
                        )
                        # [nf, nall2, nloc1, *def, 3(1), 3(2)]
                        hessian1 = xp.permute_dims(
                            hessian1,
                            (0, 2, 1, *range(3, def_ndim + 5)),
                        )
                        nloc = hessian1.shape[2]
                        # (2) -> [nf, nloc2, nloc1, *def, 3(1), 3(2)]
                        hessian = xp.zeros(
                            [*vldims, nloc, *vdef.shape, 3, 3], dtype=vv.dtype
                        )
                        mapping_hess = xp.reshape(
                            mapping_, (mldims + [1] * (len(vdef.shape) + 3))
                        )
                        mapping_hess = xp.tile(
                            mapping_hess,
                            [1] * len(mldims) + [nloc, *vdef.shape, 3, 3],
                        )
                        hessian = xp_scatter_sum(
                            hessian,
                            1,
                            mapping_hess,
                            hessian1,
                        )
                        # -> [nf, *def, nloc1, 3(1), nloc2, 3(2)]
                        hessian = xp.permute_dims(
                            hessian,
                            (
                                0,
                                *range(3, def_ndim + 3),
                                2,
                                def_ndim + 3,
                                1,
                                def_ndim + 4,
                            ),
                        )
                        # -> [nf, *def nloc1 * 3, nloc2 * 3]
                        hessian = xp.reshape(
                            hessian,
                            (hessian.shape[0], *vdef.shape, nloc * 3, nloc * 3),
                        )

                        new_ret[kk_hess] = hessian
                    else:
                        new_ret[kk_hess] = None
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                if model_ret[kk_derv_c] is not None:
                    derv_c_ext_dims = list(vdef.shape) + [9]  # noqa:RUF005
                    mapping = xp.tile(
                        mapping, [1] * (len(mldims) + len(vdef.shape)) + [3]
                    )
                    virial = xp.zeros(
                        vldims + derv_c_ext_dims,
                        dtype=vv.dtype,
                    )
                    # jax only
                    if array_api_compat.is_jax_array(virial):
                        from deepmd.jax.common import (
                            scatter_sum,
                        )

                        virial = scatter_sum(
                            virial,
                            1,
                            mapping,
                            model_ret[kk_derv_c],
                        )
                    else:
                        raise NotImplementedError("Only JAX arrays are supported.")
                    new_ret[kk_derv_c] = virial
                    new_ret[kk_derv_c + "_redu"] = xp.sum(new_ret[kk_derv_c], axis=1)
                else:
                    new_ret[kk_derv_c] = None
                    new_ret[kk_derv_c + "_redu"] = None
                if not do_atomic_virial:
                    # pop atomic virial, because it is not correctly calculated.
                    new_ret.pop(kk_derv_c)
    return new_ret
