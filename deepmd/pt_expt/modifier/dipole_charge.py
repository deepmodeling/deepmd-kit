# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dipole-charge modifier for the PyTorch exportable backend."""

from typing import (
    Any,
)

import torch

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
from deepmd.pt_expt.model.model import (
    BaseModel,
)


class DipoleChargeModifier(torch.nn.Module, DipoleChargeModifierBase):
    """Apply dipole-charge corrections to a pt_expt dipole model.

    A portable ``.dp`` file is accepted through ``model_name``. Tests and
    embedding workflows may pass an already-deserialized ``dipole_model``.
    """

    def __init__(
        self,
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1.0,
        ewald_beta: float = 0.4,
        use_cache: bool = True,
        dipole_model: Any | None = None,
    ) -> None:
        """Load or attach the exportable dipole model used by the modifier."""
        torch.nn.Module.__init__(self)
        DipoleChargeModifierBase.__init__(
            self,
            model_name,
            model_charge_map,
            sys_charge_map,
            ewald_h,
            ewald_beta,
        )
        if dipole_model is None:
            dipole_model = BaseModel.deserialize(load_dp_model(model_name)["model"])
        self.dipole_model = dipole_model
        self.dipole_model.eval()
        self.sel_type = [int(value) for value in self.dipole_model.get_sel_type()]
        if len(self.sel_type) != len(self.model_charge_map):
            raise ValueError(
                "model_charge_map length must match the dipole model sel_type length"
            )
        self.use_cache = use_cache
        self.modifier_type = "dipole_charge"

    def train(self, mode: bool = True) -> "DipoleChargeModifier":
        """Set modifier mode while keeping the embedded dipole model frozen."""
        super().train(mode)
        self.dipole_model.eval()
        return self

    def serialize(self) -> dict[str, Any]:
        """Serialize the user-facing dipole-charge configuration."""
        data = DipoleChargeModifierBase.serialize(self)
        data["use_cache"] = self.use_cache
        return data

    def _energy_with_grid(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        grids: tuple[tuple[int, int, int], ...],
        fparam: torch.Tensor | None,
        aparam: torch.Tensor | None,
        charge_spin: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate the shared energy core on a precomputed reciprocal grid."""
        prediction = self.dipole_model(
            coord,
            atype,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=False,
            charge_spin=charge_spin,
        )
        all_coord, all_charge = extend_dplr_system(
            coord,
            atype,
            prediction["dipole"],
            self.sel_type,
            self.model_charge_map,
            self.sys_charge_map,
        )
        return ewald_reciprocal_energy(
            all_coord, all_charge, box, grids, self.ewald_beta
        )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute dipole-charge outputs with the shared dpmodel core."""
        if box is None:
            raise RuntimeError("dipole_charge does not support non-periodic systems")
        if do_atomic_virial:
            raise RuntimeError("dipole_charge does not provide atomic virial")
        validate_charge_maps(
            atype,
            self.sel_type,
            self.model_charge_map,
            self.sys_charge_map,
        )
        grids = compute_ewald_grids(box.detach(), self.ewald_h)
        force_coord = coord
        if not force_coord.requires_grad:
            force_coord = force_coord.clone().requires_grad_(True)
        strain = torch.zeros(
            (coord.shape[0], 3, 3), dtype=coord.dtype, device=coord.device
        ).requires_grad_(True)
        transform = torch.eye(3, dtype=coord.dtype, device=coord.device)[None] + strain
        energy = self._energy_with_grid(
            force_coord @ transform,
            atype,
            box @ transform,
            grids,
            fparam,
            aparam,
            charge_spin,
        )
        gradients = torch.autograd.grad(
            energy,
            (force_coord, strain),
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training,
            retain_graph=True,
        )
        return {
            "energy": energy,
            "force": -gradients[0],
            "virial": -gradients[1].transpose(-1, -2).reshape(coord.shape[0], 9),
        }
