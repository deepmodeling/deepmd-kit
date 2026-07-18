# SPDX-License-Identifier: LGPL-3.0-or-later

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
    del comm_dict  # JAX path has no MPI ghost exchange
    atomic_ret = None
    atomic_output_def = self.atomic_output_def()
    model_predict = {}

    def get_atomic_ret() -> dict[str, jnp.ndarray]:
        # Some outputs, such as masks, are not differentiated. Compute the
        # primal atomic outputs lazily so force paths can obtain them from
        # jacrev(has_aux=True) instead of paying for a separate forward pass.
        nonlocal atomic_ret
        if atomic_ret is None:
            atomic_ret = self.atomic_model.forward_common_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
            )
        return atomic_ret

    for kk in atomic_output_def.keys():
        vdef = atomic_output_def[kk]
        vv = None
        ff = None
        if vdef.reducible and vdef.r_differentiable:

            def eval_output(
                cc_ext: jnp.ndarray,
                extended_atype: jnp.ndarray,
                nlist: jnp.ndarray,
                mapping: jnp.ndarray | None,
                fparam: jnp.ndarray | None,
                aparam: jnp.ndarray | None,
                charge_spin_: jnp.ndarray | None,
                *,
                _kk: str = kk,
                _atom_axis: int = -(len(vdef.shape) + 1),
            ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
                atomic_ret_ = self.atomic_model.forward_common_atomic(
                    cc_ext[None, ...],
                    extended_atype[None, ...],
                    nlist[None, ...],
                    mapping=mapping[None, ...] if mapping is not None else None,
                    fparam=fparam[None, ...] if fparam is not None else None,
                    aparam=aparam[None, ...] if aparam is not None else None,
                    charge_spin=charge_spin_[None, ...]
                    if charge_spin_ is not None
                    else None,
                )
                output = jnp.sum(atomic_ret_[_kk][0], axis=_atom_axis)
                # This function is vmapped over frames; strip the leading
                # singleton frame dimension so the aux tree has the same
                # batched shape as a direct forward_common_atomic call.
                return output, {kk_: vv_[0] for kk_, vv_ in atomic_ret_.items()}

            # Compute the coordinate Jacobian and the primal atomic outputs in
            # one transformed forward. Without has_aux, the code would need a
            # separate forward_common_atomic call before/after jacrev just to
            # populate atom_energy, masks, and reduced outputs.
            # extended_coord: [nf, nall, 3]
            # ff: [nf, *def, nall, 3]
            ff, aux_atomic_ret = jax.vmap(
                jax.jacrev(eval_output, argnums=0, has_aux=True)
            )(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
                charge_spin,
            )
            ff = -ff
            if atomic_ret is None:
                atomic_ret = aux_atomic_ret
            vv = atomic_ret[kk]
        else:
            vv = get_atomic_ret()[kk]
        model_predict[kk] = vv
        shap = vdef.shape
        atom_axis = -(len(shap) + 1)
        if vdef.reducible:
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
                if vdef.r_hessian:
                    # [nf, *def, nall, 3, nall, 3]
                    hessian, _ = jax.vmap(
                        jax.hessian(eval_output, argnums=0, has_aux=True)
                    )(
                        extended_coord,
                        extended_atype,
                        nlist,
                        mapping,
                        fparam,
                        aparam,
                        charge_spin,
                    )
                    kk_hessian = get_hessian_name(kk)
                    model_predict[kk_hessian] = hessian
                # extended_force: [nf, nall, *def, 3]
                def_ndim = len(vdef.shape)
                extended_force = jnp.transpose(
                    ff, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
                )

                model_predict[kk_derv_r] = extended_force
            if vdef.c_differentiable:
                assert vdef.r_differentiable
                # avr: [nf, *def, nall, 3, 3]
                avr = jnp.einsum("f...ai,faj->f...aij", ff, extended_coord)
                if extended_coord_corr is not None:
                    avr = avr + jnp.einsum(
                        "f...ai,faj->f...aij", ff, extended_coord_corr
                    )
                # the correction sums to zero, which does not contribute to global virial
                if do_atomic_virial:

                    def eval_ce(
                        cc_ext: jnp.ndarray,
                        extended_atype: jnp.ndarray,
                        nlist: jnp.ndarray,
                        mapping: jnp.ndarray | None,
                        fparam: jnp.ndarray | None,
                        aparam: jnp.ndarray | None,
                        charge_spin_: jnp.ndarray | None,
                        *,
                        _kk: str = kk,
                        _atom_axis: int = atom_axis - 1,
                    ) -> jnp.ndarray:
                        # This derivative is for the coordinate-weighted atomic
                        # virial correction, so it must run its own transformed
                        # forward; reusing the cached primal atomic_ret would
                        # stop the required derivative through atomic_ret[_kk].
                        # atomic_ret[_kk]: [nf, nloc, *def]
                        atomic_ret = self.atomic_model.forward_common_atomic(
                            cc_ext[None, ...],
                            extended_atype[None, ...],
                            nlist[None, ...],
                            mapping=mapping[None, ...] if mapping is not None else None,
                            fparam=fparam[None, ...] if fparam is not None else None,
                            aparam=aparam[None, ...] if aparam is not None else None,
                            charge_spin=charge_spin_[None, ...]
                            if charge_spin_ is not None
                            else None,
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
                        charge_spin,
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
