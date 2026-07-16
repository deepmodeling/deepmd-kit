# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX implementation of the dipole-charge modifier."""

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
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)


class DipoleChargeModifier(DipoleChargeModifierBase):
    """Apply dipole-charge corrections with JAX automatic differentiation."""

    def __init__(
        self,
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1.0,
        ewald_beta: float = 0.4,
        dipole_model: Any | None = None,
    ) -> None:
        """Load or attach the JAX dipole model used to create WC positions."""
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
        coord: jnp.ndarray,
        atype: jnp.ndarray,
        box: jnp.ndarray | None = None,
        fparam: jnp.ndarray | None = None,
        aparam: jnp.ndarray | None = None,
        do_atomic_virial: bool = False,
        charge_spin: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Compute dipole-charge energy, force, and virial corrections."""
        if box is None:
            raise RuntimeError("dipole_charge does not support non-periodic systems")
        if do_atomic_virial:
            raise RuntimeError("dipole_charge does not provide atomic virial")
        coord = jnp.asarray(coord)
        atype = jnp.asarray(atype)
        box = jnp.asarray(box)
        validate_charge_maps(
            atype,
            self.sel_type,
            self.model_charge_map,
            self.sys_charge_map,
        )
        grids = compute_ewald_grids(box, self.ewald_h)

        def energy_fn(force_coord: jnp.ndarray, strain: jnp.ndarray) -> jnp.ndarray:
            """Return per-frame energy while retaining the full gradient path."""
            transform = jnp.eye(3, dtype=coord.dtype)[None, :, :] + strain
            strained_coord = force_coord @ transform
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
                strained_coord,
                atype,
                prediction["dipole"],
                self.sel_type,
                self.model_charge_map,
                self.sys_charge_map,
            )
            return ewald_reciprocal_energy(
                all_coord,
                all_charge,
                strained_box,
                grids,
                self.ewald_beta,
            )

        strain = jnp.zeros((coord.shape[0], 3, 3), dtype=coord.dtype)

        def energy_with_aux(
            force_coord: jnp.ndarray, cell_strain: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """Return the scalar differentiation target and per-frame energies."""
            energy_by_frame = energy_fn(force_coord, cell_strain)
            return jnp.sum(energy_by_frame), energy_by_frame

        (_, energy_by_frame), gradients = jax.value_and_grad(
            energy_with_aux,
            argnums=(0, 1),
            has_aux=True,
        )(coord, strain)
        return {
            "energy": energy_by_frame,
            "force": -gradients[0],
            "virial": -jnp.swapaxes(gradients[1], -1, -2).reshape(coord.shape[0], 9),
        }
