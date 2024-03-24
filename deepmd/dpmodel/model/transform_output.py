# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
)

import numpy as np

from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    get_deriv_name,
    get_reduce_name,
)


def fit_output_to_model_output(
    fit_ret: Dict[str, np.ndarray],
    fit_output_def: FittingOutputDef,
    coord_ext: np.ndarray,
    do_atomic_virial: bool = False,
) -> Dict[str, np.ndarray]:
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
            # cast to energy prec brefore reduction
            model_ret[kk_redu] = np.sum(
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


def communicate_extended_output(
    model_ret: Dict[str, np.ndarray],
    model_output_def: ModelOutputDef,
    mapping: np.ndarray,  # nf x nloc
    do_atomic_virial: bool = False,
) -> Dict[str, np.ndarray]:
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
            if vdef.r_differentiable:
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                # name holders
                new_ret[kk_derv_r] = None
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                new_ret[kk_derv_c] = None
                new_ret[kk_derv_c + "_redu"] = None
                if not do_atomic_virial:
                    # pop atomic virial, because it is not correctly calculated.
                    new_ret.pop(kk_derv_c)
    return new_ret
