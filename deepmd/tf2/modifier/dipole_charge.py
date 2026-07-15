# SPDX-License-Identifier: LGPL-3.0-or-later
"""TensorFlow 2 implementation of the dipole-charge modifier."""

from typing import (
    Any,
)

from deepmd.dpmodel.modifier.dipole_charge import (
    DipoleChargeModifierBase,
    compute_ewald_grids,
    ewald_reciprocal_energy,
    extend_dplr_system,
    validate_charge_maps,
)
from deepmd.dpmodel.utils.serialization import (
    load_dp_model,
)
from deepmd.tf2.common import (
    to_tensorflow_array,
    to_tf_tensor,
    unwrap_value,
)
from deepmd.tf2.env import (
    tf,
)
from deepmd.tf2.model.base_model import (
    BaseModel,
)


class DipoleChargeModifier(DipoleChargeModifierBase):
    """Apply dipole-charge corrections with TensorFlow gradient tapes."""

    def __init__(
        self,
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1.0,
        ewald_beta: float = 0.4,
        dipole_model: Any | None = None,
    ) -> None:
        """Load or attach the TF2 dipole model used to create WC positions."""
        super().__init__(
            model_name, model_charge_map, sys_charge_map, ewald_h, ewald_beta
        )
        self.dipole_model = (
            BaseModel.deserialize(load_dp_model(model_name)["model"])
            if dipole_model is None
            else dipole_model
        )
        self.sel_type = [int(value) for value in self.dipole_model.get_sel_type()]
        if len(self.sel_type) != len(self.model_charge_map):
            raise ValueError(
                "model_charge_map length must match the dipole model sel_type length"
            )

    def __call__(
        self,
        coord: Any,
        atype: Any,
        box: Any | None = None,
        fparam: Any | None = None,
        aparam: Any | None = None,
        do_atomic_virial: bool = False,
        charge_spin: Any | None = None,
    ) -> dict[str, tf.Tensor]:
        """Compute dipole-charge energy, force, and virial corrections."""
        if box is None:
            raise RuntimeError("dipole_charge does not support non-periodic systems")
        if do_atomic_virial:
            raise RuntimeError("dipole_charge does not provide atomic virial")
        coord = to_tf_tensor(to_tensorflow_array(coord))
        atype = to_tf_tensor(to_tensorflow_array(atype))
        box = to_tf_tensor(to_tensorflow_array(box))
        assert coord is not None
        assert atype is not None
        assert box is not None
        validate_charge_maps(
            atype,
            self.sel_type,
            self.model_charge_map,
            self.sys_charge_map,
        )
        grids = compute_ewald_grids(box, self.ewald_h)
        with tf.GradientTape() as tape:
            tape.watch(coord)
            strain = tf.zeros((tf.shape(coord)[0], 3, 3), dtype=coord.dtype)
            tape.watch(strain)
            transform = (
                tf.eye(3, batch_shape=[tf.shape(coord)[0]], dtype=coord.dtype) + strain
            )
            strained_coord = coord @ transform
            strained_box = box @ transform
            prediction = self.dipole_model(
                strained_coord,
                atype,
                box=strained_box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=False,
                charge_spin=charge_spin,
            )
            all_coord, all_charge = extend_dplr_system(
                to_tensorflow_array(strained_coord),
                to_tensorflow_array(atype),
                prediction["dipole"],
                self.sel_type,
                self.model_charge_map,
                self.sys_charge_map,
            )
            energy = unwrap_value(
                ewald_reciprocal_energy(
                    all_coord,
                    all_charge,
                    to_tensorflow_array(strained_box),
                    grids,
                    self.ewald_beta,
                )
            )
            total_energy = tf.reduce_sum(energy)
        force_grad, strain_grad = tape.gradient(total_energy, [coord, strain])
        assert force_grad is not None
        assert strain_grad is not None
        return {
            "energy": energy,
            "force": -force_grad,
            "virial": -tf.reshape(tf.transpose(strain_grad, (0, 2, 1)), (-1, 9)),
        }
