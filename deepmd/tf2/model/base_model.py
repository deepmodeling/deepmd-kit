# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)
from deepmd.dpmodel.output_def import (
    get_deriv_name,
    get_hessian_name,
    get_reduce_name,
)
from deepmd.tf2.common import (
    to_tf_tensor,
    wrap_tensor,
)
from deepmd.tf2.env import (
    jnp,
    tf,
)

BaseModel = make_base_model()


def forward_common_atomic(
    self: "BaseModel",
    extended_coord: jnp.ndarray,
    extended_atype: jnp.ndarray,
    nlist: jnp.ndarray,
    mapping: jnp.ndarray | None = None,
    fparam: jnp.ndarray | None = None,
    aparam: jnp.ndarray | None = None,
    do_atomic_virial: bool = False,
    extended_coord_corr: jnp.ndarray | None = None,
    comm_dict: dict | None = None,
    charge_spin: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    del comm_dict  # tf2 path has no MPI ghost exchange

    coord_tensor = to_tf_tensor(extended_coord)
    assert coord_tensor is not None
    coord_array = wrap_tensor(coord_tensor)
    atomic_ret = self.atomic_model.forward_common_atomic(
        coord_array,
        extended_atype,
        nlist,
        mapping=mapping,
        fparam=fparam,
        aparam=aparam,
        charge_spin=charge_spin,
    )
    atomic_output_def = self.atomic_output_def()
    model_predict = {}
    for kk, vv in atomic_ret.items():
        model_predict[kk] = vv
        vdef = atomic_output_def[kk]
        atom_axis = -(len(vdef.shape) + 1)
        if not vdef.reducible:
            continue

        kk_redu = get_reduce_name(kk)
        if vdef.intensive:
            mask = atomic_ret["mask"] if "mask" in atomic_ret else None
            if mask is not None:
                model_predict[kk_redu] = jnp.sum(vv, axis=atom_axis) / jnp.sum(
                    mask, axis=-1, keepdims=True
                )
            else:
                model_predict[kk_redu] = jnp.mean(vv, axis=atom_axis)
        else:
            model_predict[kk_redu] = jnp.sum(vv, axis=atom_axis)

        kk_derv_r, kk_derv_c = get_deriv_name(kk)
        if vdef.r_differentiable:
            with tf.GradientTape() as tape:
                tape.watch(coord_tensor)
                grad_atomic_ret = self.atomic_model.forward_common_atomic(
                    wrap_tensor(coord_tensor),
                    extended_atype,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                    charge_spin=charge_spin,
                )
                reduced_output = jnp.sum(grad_atomic_ret[kk], axis=atom_axis)
                reduced_output_tensor = to_tf_tensor(reduced_output)
                assert reduced_output_tensor is not None
            ff_tensor = -tape.batch_jacobian(reduced_output_tensor, coord_tensor)
            ff = wrap_tensor(ff_tensor)

            # extended_force: [nf, nall, *def, 3]
            def_ndim = len(vdef.shape)
            model_predict[kk_derv_r] = jnp.transpose(
                ff, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
            )
            if vdef.r_hessian:
                kk_hessian = get_hessian_name(kk)
                model_predict[kk_hessian] = None

        if vdef.c_differentiable:
            assert vdef.r_differentiable
            # avr: [nf, *def, nall, 3, 3]
            avr = jnp.einsum("f...ai,faj->f...aij", ff, extended_coord)
            if extended_coord_corr is not None:
                avr = avr + jnp.einsum("f...ai,faj->f...aij", ff, extended_coord_corr)
            # The JAX backend adds an extra per-atom correction for
            # do_atomic_virial=True.  This tf2 path keeps the conservative
            # virial term; the correction can be added with training-gradient
            # support later.
            avr = jnp.reshape(avr, [*ff.shape[:-1], 9])
            # extended_virial: [nf, nall, *def, 9]
            extended_virial = jnp.transpose(
                avr, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
            )
            model_predict[kk_derv_c] = extended_virial
            # [nf, *def, 9]
            model_predict[kk_derv_c + "_redu"] = jnp.sum(extended_virial, axis=1)
    return model_predict
