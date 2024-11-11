# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_reduce_name,
)


def get_leading_dims(
    vv: tnp.ndarray,
    vdef: OutputVariableDef,
) -> tnp.ndarray:
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
    vshape = tf.shape(vv)
    return vshape[: (len(vshape) - len(vdef.shape))]


def communicate_extended_output(
    model_ret: dict[str, tnp.ndarray],
    model_output_def: ModelOutputDef,
    mapping: tnp.ndarray,  # nf x nloc
    do_atomic_virial: bool = False,
) -> dict[str, tnp.ndarray]:
    """Transform the output of the model network defined on
    local and ghost (extended) atoms to local atoms.

    """
    new_ret = {}
    for kk in model_output_def.keys_outp():
        vv = model_ret[kk]
        vdef = model_output_def[kk]
        new_ret[kk] = vv
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            new_ret[kk_redu] = model_ret[kk_redu]
            kk_derv_r, kk_derv_c = get_deriv_name(kk)
            mldims = tf.shape(mapping)
            vldims = get_leading_dims(vv, vdef)
            if vdef.r_differentiable:
                if model_ret[kk_derv_r] is not None:
                    derv_r_ext_dims = list(vdef.shape) + [3]  # noqa:RUF005
                    indices = mapping.reshape(tf.shape(mapping)[0], -1, 1)
                    # concat frame idx
                    indices = tf.concat(
                        [
                            tf.repeat(
                                tf.range(tf.shape(indices)[0], dtype=indices.dtype),
                                tf.shape(mapping)[1],
                            ).reshape(tf.shape(indices)),
                            indices,
                        ],
                        axis=-1,
                    )
                    force = tf.scatter_nd(
                        indices,
                        model_ret[kk_derv_r],
                        tf.cast(tf.concat([vldims, derv_r_ext_dims], axis=0), tf.int64),
                    )
                    new_ret[kk_derv_r] = force.reshape(
                        tf.concat([tf.shape(force)[:2], list(vdef.shape), [3]], axis=0)
                    )
                else:
                    # name holders
                    new_ret[kk_derv_r] = None
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                if model_ret[kk_derv_c] is not None:
                    derv_c_ext_dims = list(vdef.shape) + [9]  # noqa:RUF005
                    indices = mapping.reshape(tf.shape(mapping)[0], -1, 1)
                    # concat frame idx
                    indices = tf.concat(
                        [
                            tf.repeat(
                                tf.range(tf.shape(indices)[0], dtype=indices.dtype),
                                tf.shape(mapping)[1],
                            ).reshape(tf.shape(indices)),
                            indices,
                        ],
                        axis=-1,
                    )
                    virial = tf.scatter_nd(
                        indices,
                        model_ret[kk_derv_c],
                        tf.cast(tf.concat([vldims, derv_c_ext_dims], axis=0), tf.int64),
                    )
                    new_ret[kk_derv_c] = virial.reshape(
                        tf.concat([tf.shape(virial)[:2], list(vdef.shape), [9]], axis=0)
                    )
                    new_ret[kk_derv_c + "_redu"] = tnp.sum(new_ret[kk_derv_c], axis=1)
                else:
                    new_ret[kk_derv_c] = None
                    new_ret[kk_derv_c + "_redu"] = None
                if not do_atomic_virial:
                    # pop atomic virial, because it is not correctly calculated.
                    new_ret.pop(kk_derv_c)
    return new_ret
