# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native-spin energies and magnetic forces through JAX autodifferentiation."""

from deepmd.dpmodel.model.native_spin_model import (
    make_native_spin_model,
)
from deepmd.jax.env import (
    jnp,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.ener_model import (
    EnergyModel,
)


@BaseModel.register("native_spin")
class NativeSpinEnergyModel(make_native_spin_model(EnergyModel)):
    _backbone_model_registry = BaseModel

    def _translate_eager_call(
        self,
        model_ret: dict[str, jnp.ndarray],
        atype: jnp.ndarray,
        do_atomic_virial: bool = False,
    ) -> dict[str, jnp.ndarray]:
        result = {
            "atom_energy": model_ret["energy"],
            "energy": model_ret["energy_redu"],
            "force": model_ret["energy_derv_r"].squeeze(-2),
            "force_mag": model_ret["energy_derv_r_mag"].squeeze(-2),
            "virial": model_ret["energy_derv_c_redu"].squeeze(-2),
            "mask_mag": self._spin_active_mask(atype),
            "mask": model_ret["mask"],
        }
        if do_atomic_virial:
            result["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        return result

    def call_lower(
        self,
        extended_coord: jnp.ndarray,
        extended_atype: jnp.ndarray,
        nlist: jnp.ndarray,
        mapping: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        do_atomic_virial: bool = False,
        charge_spin: jnp.ndarray | None = None,
        spin: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Evaluate extended coordinates with public force and magnetic keys."""
        result = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
            spin=spin,
        )
        output = {
            "atom_energy": result["energy"],
            "energy": result["energy_redu"],
            "extended_force": result["energy_derv_r"].squeeze(-2),
            "extended_force_mag": result["energy_derv_r_mag"].squeeze(-2),
            "virial": result["energy_derv_c_redu"].squeeze(-2),
            "mask_mag": result["mask_mag"],
            "mask": result["mask"],
        }
        if do_atomic_virial:
            output["extended_virial"] = result["energy_derv_c"].squeeze(-2)
        return output
