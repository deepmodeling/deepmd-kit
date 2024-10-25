# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)
from deepmd.dpmodel.output_def import (
    get_deriv_name,
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
            if vdef.c_differentiable:
                size = 1
                for ii in vdef.shape:
                    size *= ii

                split_ff = []
                split_vv = []
                for ss in range(size):

                    def eval_output(
                        cc_ext, extended_atype, nlist, mapping, fparam, aparam
                    ):
                        atomic_ret = self.atomic_model.forward_common_atomic(
                            cc_ext[None, ...],
                            extended_atype[None, ...],
                            nlist[None, ...],
                            mapping=mapping[None, ...] if mapping is not None else None,
                            fparam=fparam[None, ...] if fparam is not None else None,
                            aparam=aparam[None, ...] if aparam is not None else None,
                        )
                        return jnp.sum(atomic_ret[kk][0], axis=atom_axis)[ss]

                    ffi = -jax.vmap(jax.grad(eval_output, argnums=0))(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                    )
                    aviri = ffi[..., None] @ extended_coord[..., None, :]
                    ffi = ffi[..., None, :]
                    split_ff.append(ffi)
                    aviri = aviri[..., None, :]
                    split_vv.append(aviri)
                out_lead_shape = list(extended_coord.shape[:-1]) + vdef.shape
                extended_force = jnp.concat(split_ff, axis=-2).reshape(
                    *out_lead_shape, 3
                )

                model_predict[kk_derv_r] = extended_force
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                extended_virial = jnp.concat(split_vv, axis=-2).reshape(
                    *out_lead_shape, 9
                )
                # the correction sums to zero, which does not contribute to global virial
                if do_atomic_virial:
                    raise NotImplementedError("Atomic virial is not implemented yet.")
                # to [...,3,3] -> [...,9]
                model_predict[kk_derv_c] = extended_virial
    return model_predict
