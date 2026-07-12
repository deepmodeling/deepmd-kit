# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import (
    Any,
)

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
    tf,
    xp,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class BaseModel(make_base_model()):
    """TF2 model registry with adapters for regular PT SeZM checkpoints."""

    _SEZM_MODEL_TYPES = frozenset({"sezm", "dpa4"})
    _SEZM_ATOMIC_TYPES = frozenset({"sezm_atomic"})

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "BaseModel":
        model_type = str(data.get("type", "standard")).lower()
        if model_type in cls._SEZM_MODEL_TYPES:
            return cls.deserialize(cls._unwrap_pt_sezm_model(data))
        if model_type in cls._SEZM_ATOMIC_TYPES:
            return cls.deserialize(cls._normalize_pt_sezm_atomic(data))
        return super().deserialize(data)

    @staticmethod
    def _unwrap_pt_sezm_model(data: dict[str, Any]) -> dict[str, Any]:
        """Unwrap PT's model-level SeZM schema after validating its extras."""
        check_version_compatibility(int(data.get("@version", 1)), 1, 1)
        if str(data.get("bridging_method", "none")).lower() not in ("none", ""):
            raise NotImplementedError(
                "PT SeZM/DPA4 checkpoints with bridging are not supported in TF2."
            )
        if data.get("lora") is not None:
            raise NotImplementedError(
                "PT SeZM/DPA4 checkpoints with LoRA are not supported in TF2."
            )
        atomic_model = data.get("atomic_model")
        if atomic_model is None:
            raise ValueError("SeZM/DPA4 model data is missing 'atomic_model'.")
        return atomic_model

    @staticmethod
    def _normalize_pt_sezm_atomic(data: dict[str, Any]) -> dict[str, Any]:
        """Convert PT's energy-only ``sezm_atomic`` schema to ``standard``."""
        data = data.copy()
        check_version_compatibility(int(data.get("@version", 2)), 3, 2)
        if data.pop("dens_fitting", None) is not None:
            raise NotImplementedError(
                "PT SeZM/DPA4 checkpoints with a dens head are not supported in TF2."
            )
        active_mode = data.pop("active_mode", None)
        if active_mode not in (None, "ener"):
            raise NotImplementedError(
                f"PT SeZM/DPA4 active_mode {active_mode!r} is not supported in TF2."
            )
        variables = data.get("@variables")
        if isinstance(variables, dict):
            data["@variables"] = {
                key: value
                for key, value in variables.items()
                if key in ("out_bias", "out_std")
            }
        descriptor = data.get("descriptor")
        descriptor_config = (
            descriptor.get("config") if isinstance(descriptor, dict) else None
        )
        if isinstance(descriptor, dict) and (
            descriptor.get("random_gamma")
            or (
                isinstance(descriptor_config, dict)
                and descriptor_config.get("random_gamma")
            )
        ):
            # PT checkpoints may keep the training augmentation enabled, while
            # TF2 conversion produces an inference SavedModel. Inference fixes
            # the local-Z roll, so normalize this field before construction.
            descriptor = descriptor.copy()
            if isinstance(descriptor_config, dict):
                descriptor_config = descriptor_config.copy()
                descriptor_config["random_gamma"] = False
                descriptor["config"] = descriptor_config
            else:
                descriptor["random_gamma"] = False
            data["descriptor"] = descriptor
        data["@version"] = 2
        data["type"] = "standard"
        return data


def _collect_model_predict(
    atomic_ret: dict[str, xp.ndarray],
    atomic_output_def: Any,
) -> tuple[dict[str, xp.ndarray], dict[str, tf.Tensor]]:
    model_predict: dict[str, xp.ndarray] = {}
    reduced_output_tensors: dict[str, tf.Tensor] = {}
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
                model_predict[kk_redu] = xp.sum(vv, axis=atom_axis) / xp.sum(
                    mask, axis=-1, keepdims=True
                )
            else:
                model_predict[kk_redu] = xp.mean(vv, axis=atom_axis)
        else:
            model_predict[kk_redu] = xp.sum(vv, axis=atom_axis)

        if vdef.r_differentiable:
            reduced_output_tensor = to_tf_tensor(model_predict[kk_redu])
            assert reduced_output_tensor is not None
            reduced_output_tensors[kk] = reduced_output_tensor
    return model_predict, reduced_output_tensors


def _negative_coordinate_derivative(
    tape: tf.GradientTape,
    reduced_output_tensor: tf.Tensor,
    coord_tensor: tf.Tensor,
    output_size: int,
) -> tf.Tensor:
    if output_size == 1:
        grad = tape.gradient(
            reduced_output_tensor,
            coord_tensor,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        return -grad[:, tf.newaxis, :, :]
    return -tape.batch_jacobian(reduced_output_tensor, coord_tensor)


def forward_common_atomic(
    self: "BaseModel",
    extended_coord: xp.ndarray,
    extended_atype: xp.ndarray,
    nlist: xp.ndarray,
    mapping: xp.ndarray | None = None,
    fparam: xp.ndarray | None = None,
    aparam: xp.ndarray | None = None,
    do_atomic_virial: bool = False,
    do_deriv_c: bool = True,
    extended_coord_corr: xp.ndarray | None = None,
    comm_dict: dict | None = None,
    charge_spin: xp.ndarray | None = None,
) -> dict[str, xp.ndarray]:
    del comm_dict  # tf2 path has no MPI ghost exchange

    coord_tensor = to_tf_tensor(extended_coord)
    assert coord_tensor is not None
    atomic_output_def = self.atomic_output_def()
    derivative_keys = [
        kk for kk in atomic_output_def.keys() if atomic_output_def[kk].r_differentiable
    ]
    tape: tf.GradientTape | None = None
    if derivative_keys:
        tape = tf.GradientTape(persistent=len(derivative_keys) > 1)
        with tape:
            tape.watch(coord_tensor)
            atomic_ret = self.atomic_model.forward_common_atomic(
                wrap_tensor(coord_tensor),
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
            )
            model_predict, reduced_output_tensors = _collect_model_predict(
                atomic_ret, atomic_output_def
            )
    else:
        atomic_ret = self.atomic_model.forward_common_atomic(
            wrap_tensor(coord_tensor),
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
        )
        model_predict, reduced_output_tensors = _collect_model_predict(
            atomic_ret, atomic_output_def
        )

    for kk in derivative_keys:
        vdef = atomic_output_def[kk]
        kk_derv_r, kk_derv_c = get_deriv_name(kk)
        assert tape is not None
        reduced_output_tensor = reduced_output_tensors[kk]
        ff_tensor = _negative_coordinate_derivative(
            tape,
            reduced_output_tensor,
            coord_tensor,
            vdef.output_size,
        )
        ff = wrap_tensor(ff_tensor)

        # extended_force: [nf, nall, *def, 3]
        def_ndim = len(vdef.shape)
        model_predict[kk_derv_r] = xp.transpose(
            ff, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
        )
        if vdef.r_hessian:
            kk_hessian = get_hessian_name(kk)
            model_predict[kk_hessian] = None

        if vdef.c_differentiable:
            assert vdef.r_differentiable
            if not do_deriv_c:
                model_predict[kk_derv_c] = None
                model_predict[kk_derv_c + "_redu"] = None
                continue
            # avr: [nf, *def, nall, 3, 3]
            avr = xp.einsum("f...ai,faj->f...aij", ff, extended_coord)
            if extended_coord_corr is not None:
                avr = avr + xp.einsum("f...ai,faj->f...aij", ff, extended_coord_corr)
            if do_atomic_virial:
                with tf.GradientTape() as virial_tape:
                    virial_tape.watch(coord_tensor)
                    virial_atomic_ret = self.atomic_model.forward_common_atomic(
                        wrap_tensor(coord_tensor),
                        extended_atype,
                        nlist,
                        mapping=mapping,
                        fparam=fparam,
                        aparam=aparam,
                        charge_spin=charge_spin,
                    )
                    virial_atomic_tensor = to_tf_tensor(virial_atomic_ret[kk])
                    assert virial_atomic_tensor is not None
                    nloc = tf.shape(nlist)[1]
                    loc_coord = tf.stop_gradient(coord_tensor[:, :nloc, :])
                    loc_coord = tf.reshape(
                        loc_coord,
                        [
                            tf.shape(loc_coord)[0],
                            tf.shape(loc_coord)[1],
                            *([1] * def_ndim),
                            3,
                        ],
                    )
                    corr_output = tf.reduce_sum(
                        virial_atomic_tensor[..., tf.newaxis] * loc_coord,
                        axis=1,
                    )
                virial_corr = virial_tape.batch_jacobian(corr_output, coord_tensor)
                virial_corr = tf.transpose(
                    virial_corr,
                    [
                        0,
                        *range(1, def_ndim + 1),
                        def_ndim + 2,
                        def_ndim + 3,
                        def_ndim + 1,
                    ],
                )
                avr = avr + wrap_tensor(virial_corr)
            avr = xp.reshape(avr, [*ff.shape[:-1], 9])
            # extended_virial: [nf, nall, *def, 9]
            extended_virial = xp.transpose(
                avr, [0, def_ndim + 1, *range(1, def_ndim + 1), def_ndim + 2]
            )
            model_predict[kk_derv_c] = extended_virial
            # [nf, *def, 9]
            model_predict[kk_derv_c + "_redu"] = xp.sum(extended_virial, axis=1)
    del tape
    return model_predict
