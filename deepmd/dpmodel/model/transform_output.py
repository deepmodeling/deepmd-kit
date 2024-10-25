# SPDX-License-Identifier: LGPL-3.0-or-later

import array_api_compat
import numpy as np

from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
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
            # cast to energy prec brefore reduction
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
                    force = xp.zeros(
                        vldims + derv_r_ext_dims, dtype=vv.dtype, device=vv.device
                    )
                    # jax only
                    if array_api_compat.is_jax_array(force):
                        from deepmd.jax.env import (
                            jnp,
                        )

                        f_idx = xp.arange(force.size, dtype=xp.int64).reshape(
                            force.shape
                        )
                        new_idx = jnp.take_along_axis(f_idx, mapping, axis=1).ravel()
                        f_shape = force.shape
                        force = force.ravel()
                        force = force.at[new_idx].add(model_ret[kk_derv_r].ravel())
                        force = force.reshape(f_shape)
                    else:
                        raise NotImplementedError("Only JAX arrays are supported.")
                    new_ret[kk_derv_r] = force
                else:
                    # name holders
                    new_ret[kk_derv_r] = None
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                if model_ret[kk_derv_c] is not None:
                    derv_c_ext_dims = list(vdef.shape) + [9]  # noqa:RUF005
                    mapping = xp.tile(
                        mapping, [1] * (len(mldims) + len(vdef.shape)) + [3]
                    )
                    virial = xp.zeros(
                        vldims + derv_c_ext_dims, dtype=vv.dtype, device=vv.device
                    )
                    # jax only
                    if array_api_compat.is_jax_array(virial):
                        from deepmd.jax.env import (
                            jnp,
                        )

                        v_idx = xp.arange(virial.size, dtype=xp.int64).reshape(
                            virial.shape
                        )
                        new_idx = jnp.take_along_axis(v_idx, mapping, axis=1).ravel()
                        v_shape = virial.shape
                        virial = virial.ravel()
                        virial = virial.at[new_idx].add(model_ret[kk_derv_c].ravel())
                        virial = virial.reshape(v_shape)
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
