# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)
from deepmd.dpmodel.output_def import (
    get_deriv_name,
    get_hessian_name,
    get_reduce_name,
)
from deepmd.jax.env import (
    jax,
    jnp,
)

BaseModel = make_base_model()


def forward_common_atomic(
    self,
    extended_coord: jnp.ndarray,
    extended_atype: jnp.ndarray,
    nlist: jnp.ndarray,
    mapping: Optional[jnp.ndarray] = None,
    fparam: Optional[jnp.ndarray] = None,
    aparam: Optional[jnp.ndarray] = None,
    do_atomic_virial: bool = False,
):
    atomic_ret = self.atomic_model.forward_common_atomic(
        extended_coord,
        extended_atype,
        nlist,
        mapping=mapping,
        fparam=fparam,
        aparam=aparam,
    )
    atomic_output_def = self.atomic_output_def()
    model_predict = {}
    for kk, vv in atomic_ret.items():
        model_predict[kk] = vv
        vdef = atomic_output_def[kk]
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if vdef.reducible:
            kk_redu = get_reduce_name(kk)
            model_predict[kk_redu] = jnp.sum(vv, axis=atom_axis)
            kk_derv_r, kk_derv_c = get_deriv_name(kk)
            if vdef.r_differentiable:

                def eval_output(
                    cc_ext,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    *,
                    _kk=kk,
                    _atom_axis=atom_axis,
                ):
                    atomic_ret = self.atomic_model.forward_common_atomic(
                        cc_ext[None, ...],
                        extended_atype[None, ...],
                        nlist[None, ...],
                        mapping=mapping[None, ...] if mapping is not None else None,
                        fparam=fparam[None, ...] if fparam is not None else None,
                        aparam=aparam[None, ...] if aparam is not None else None,
                    )
                    return jnp.sum(atomic_ret[_kk][0], axis=_atom_axis)

                # extended_coord: [nf, nall, 3]
                # ff: [nf, *def, nall, 3]
                ff = -jax.vmap(jax.jacrev(eval_output, argnums=0))(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                )
                # extended_force: [nf, nall, *def, 3]
                def_ndim = len(vdef.shape)
                extended_force = jnp.transpose(
                    ff, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
                )

                model_predict[kk_derv_r] = extended_force
                if vdef.r_hessian:
                    # [nf, *def, nall, 3, nall, 3]
                    hessian = jax.vmap(jax.hessian(eval_output, argnums=0))(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                    )
                    kk_hessian = get_hessian_name(kk)
                    model_predict[kk_hessian] = hessian
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                # avr: [nf, *def, nall, 3, 3]
                avr = jnp.einsum("f...ai,faj->f...aij", ff, extended_coord)
                # the correction sums to zero, which does not contribute to global virial
                if do_atomic_virial:

                    def eval_ce(
                        cc_ext,
                        extended_atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                        *,
                        _kk=kk,
                        _atom_axis=atom_axis - 1,
                    ):
                        # atomic_ret[_kk]: [nf, nloc, *def]
                        atomic_ret = self.atomic_model.forward_common_atomic(
                            cc_ext[None, ...],
                            extended_atype[None, ...],
                            nlist[None, ...],
                            mapping=mapping[None, ...] if mapping is not None else None,
                            fparam=fparam[None, ...] if fparam is not None else None,
                            aparam=aparam[None, ...] if aparam is not None else None,
                        )
                        nloc = nlist.shape[0]
                        cc_loc = jax.lax.stop_gradient(cc_ext)[:nloc, ...]
                        cc_loc = jnp.reshape(cc_loc, [nloc, *[1] * def_ndim, 3])
                        # [*def, 3]
                        return jnp.sum(
                            atomic_ret[_kk][0, ..., None] * cc_loc, axis=_atom_axis
                        )

                    # extended_virial_corr: [nf, *def, 3, nall, 3]
                    extended_virial_corr = jax.vmap(jax.jacrev(eval_ce, argnums=0))(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                    )
                    # move the first 3 to the last
                    # [nf, *def, nall, 3, 3]
                    extended_virial_corr = jnp.transpose(
                        extended_virial_corr,
                        [
                            0,
                            *range(1, def_ndim + 1),
                            def_ndim + 2,
                            def_ndim + 3,
                            def_ndim + 1,
                        ],
                    )
                    avr += extended_virial_corr
                # to [...,3,3] -> [...,9]
                # avr: [nf, *def, nall, 9]
                avr = jnp.reshape(avr, [*ff.shape[:-1], 9])
                # extended_virial: [nf, nall, *def, 9]
                extended_virial = jnp.transpose(
                    avr, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
                )
                model_predict[kk_derv_c] = extended_virial
                # [nf, *def, 9]
                model_predict[kk_derv_c + "_redu"] = jnp.sum(extended_virial, axis=1)
    return model_predict
