# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent helpers for the dipole-charge modifier."""

from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_modifier import (
    make_base_modifier,
)

BaseModifier = make_base_modifier()

ELECTROSTATIC_CONVERSION = 14.39964535475696995031


def compute_ewald_grids(box: Any, spacing: float) -> tuple[tuple[int, int, int], ...]:
    """Compute one fixed even reciprocal grid for every frame's cell."""
    box_array = to_numpy_array(box)
    grids = []
    for frame in range(box_array.shape[0]):
        grid = []
        for axis in range(3):
            size = int(np.ceil(np.linalg.norm(box_array[frame, axis]) / spacing))
            grid.append(size + size % 2)
        grids.append((grid[0], grid[1], grid[2]))
    return tuple(grids)


def validate_charge_maps(
    atype: Any,
    sel_type: list[int],
    model_charge_map: list[float],
    sys_charge_map: list[float],
) -> None:
    """Validate type coverage before entering a backend differentiation graph."""
    atype_array = to_numpy_array(atype)
    real_types = atype_array[atype_array >= 0]
    if real_types.size and np.max(real_types) >= len(sys_charge_map):
        raise ValueError("sys_charge_map does not cover all real atom types")
    if any(atom_type < 0 or atom_type >= len(sys_charge_map) for atom_type in sel_type):
        raise ValueError("sel_type contains an atom type outside sys_charge_map")
    if len(sel_type) != len(model_charge_map):
        raise ValueError(
            "model_charge_map must follow get_sel_type() order and have equal length"
        )


def extend_dplr_system(
    coord: Any,
    atype: Any,
    dipole: Any,
    sel_type: list[int],
    model_charge_map: list[float],
    sys_charge_map: list[float],
) -> tuple[Any, Any]:
    """Append fixed-shape WC slots and construct real/WC charges.

    One WC slot is appended for every input atom. Slots belonging to unselected
    or virtual atoms carry zero charge, avoiding variable-length boolean fancy
    indexing while remaining exactly equivalent in the Ewald sum.
    """
    xp = array_api_compat.array_namespace(coord, atype, dipole)
    device = array_api_compat.device(coord)
    real_mask = atype >= 0
    safe_atype = xp.where(real_mask, atype, xp.zeros_like(atype))
    sys_charge = xp.asarray(sys_charge_map, dtype=coord.dtype, device=device)
    type_index = xp.arange(len(sys_charge_map), dtype=atype.dtype, device=device)
    type_one_hot = xp.astype(safe_atype[..., None] == type_index, coord.dtype)
    real_charge = xp.where(
        real_mask,
        xp.sum(type_one_hot * sys_charge, axis=-1),
        xp.zeros_like(coord[..., 0]),
    )

    wc_charge_by_type = xp.zeros(
        (len(sys_charge_map),), dtype=coord.dtype, device=device
    )
    for index, atom_type in enumerate(sel_type):
        type_mask = type_index == xp.asarray(
            atom_type, dtype=atype.dtype, device=device
        )
        wc_charge_by_type = xp.where(
            type_mask,
            xp.asarray(model_charge_map[index], dtype=coord.dtype, device=device),
            wc_charge_by_type,
        )
    wc_charge = xp.where(
        real_mask,
        xp.sum(type_one_hot * wc_charge_by_type, axis=-1),
        xp.zeros_like(coord[..., 0]),
    )
    selected_mask = wc_charge != 0
    wc_coord = coord + dipole * xp.astype(selected_mask[..., None], coord.dtype)
    return (
        xp.concat((coord, wc_coord), axis=1),
        xp.concat((real_charge, wc_charge), axis=1),
    )


def ewald_reciprocal_energy(
    coord: Any,
    charge: Any,
    box: Any,
    grids: tuple[tuple[int, int, int], ...],
    beta: float,
) -> Any:
    """Evaluate reciprocal Ewald energy using only Array API operations."""
    xp = array_api_compat.array_namespace(coord, charge, box)
    device = array_api_compat.device(coord)
    pi = xp.asarray(np.pi, dtype=coord.dtype, device=device)
    conversion = xp.asarray(ELECTROSTATIC_CONVERSION, dtype=coord.dtype, device=device)
    frame_energy = []
    for frame in range(coord.shape[0]):
        axes = tuple(
            xp.arange(-size // 2, size // 2 + 1, dtype=coord.dtype, device=device)
            for size in grids[frame]
        )
        mesh = xp.meshgrid(*axes, indexing="ij")
        wave_index = xp.reshape(xp.stack(mesh, axis=-1), (-1, 3))
        nonzero = xp.any(wave_index != 0, axis=-1)
        cell = box[frame, ...]
        inverse_box = xp.linalg.inv(cell)
        wave = wave_index @ xp.permute_dims(inverse_box, (1, 0))
        wave2 = xp.sum(wave * wave, axis=-1)
        safe_wave2 = xp.where(
            nonzero,
            wave2,
            xp.ones_like(wave2),
        )
        fractional = coord[frame, ...] @ inverse_box
        phase = 2 * pi * (fractional @ xp.permute_dims(wave_index, (1, 0)))
        sqr = xp.sum(charge[frame, :, None] * xp.cos(phase), axis=0)
        sqi = xp.sum(charge[frame, :, None] * xp.sin(phase), axis=0)
        kernel = xp.exp(-(pi * pi) * safe_wave2 / (beta * beta)) / safe_wave2
        kernel = kernel * xp.astype(nonzero, coord.dtype)
        volume = xp.abs(xp.linalg.det(cell))
        frame_energy.append(
            xp.sum(kernel * (sqr * sqr + sqi * sqi)) * conversion / (2 * pi * volume)
        )
    return xp.stack(frame_energy, axis=0)[:, None]


class DipoleChargeModifierBase:
    """Store the dipole-charge schema and define its atom-selection contract.

    The modifier uses distinct masking concepts. ``real_atom_mask`` excludes
    padding or externally supplied virtual atoms (negative atom types), while
    ``selected_wc_mask`` identifies the real atoms that create Wannier charge
    centers according to the dipole model's ``sel_type``. Neighbor-list masks
    remain the responsibility of the embedded dipole model and must not be
    reused as either of these masks.

    ``model_charge_map`` follows the exact order returned by the embedded
    dipole model's ``get_sel_type()`` method, matching the established
    TensorFlow modifier input contract.
    """

    modifier_type = "dipole_charge"

    def __init__(
        self,
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1.0,
        ewald_beta: float = 0.4,
    ) -> None:
        """Initialize the shared dipole-charge configuration."""
        if not model_name:
            raise ValueError("model_name must identify a dipole model")
        if not model_charge_map:
            raise ValueError("model_charge_map must not be empty")
        if not sys_charge_map:
            raise ValueError("sys_charge_map must not be empty")
        if ewald_h <= 0.0:
            raise ValueError("ewald_h must be positive")
        if ewald_beta <= 0.0:
            raise ValueError("ewald_beta must be positive")
        self.model_name = model_name
        self.model_charge_map = [float(value) for value in model_charge_map]
        self.sys_charge_map = [float(value) for value in sys_charge_map]
        self.ewald_h = float(ewald_h)
        self.ewald_beta = float(ewald_beta)

    def serialize(self) -> dict[str, Any]:
        """Serialize the backend-neutral dipole-charge configuration."""
        return {
            "@class": "Modifier",
            "type": self.modifier_type,
            "@version": 3,
            "model_name": self.model_name,
            "model_charge_map": self.model_charge_map,
            "sys_charge_map": self.sys_charge_map,
            "ewald_h": self.ewald_h,
            "ewald_beta": self.ewald_beta,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DipoleChargeModifierBase":
        """Deserialize a dipole-charge configuration with version validation."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 3, 1)
        data.pop("@class", None)
        data.pop("type", None)
        return cls(**data)

    @staticmethod
    def get_real_atom_mask(atype: np.ndarray) -> np.ndarray:
        """Return the mask of physical atoms accepted by the dipole model."""
        return np.asarray(atype) >= 0

    @staticmethod
    def get_selected_wc_mask(atype: np.ndarray, sel_type: list[int]) -> np.ndarray:
        """Return real atoms whose dipole output creates a virtual WC site."""
        atype = np.asarray(atype)
        return (atype >= 0) & np.isin(atype, np.asarray(sel_type, dtype=np.int64))

    def make_charge_maps(
        self, atype: np.ndarray, sel_type: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map real atom and selected WC types to their configured charges."""
        atype = np.asarray(atype, dtype=np.int64)
        real_mask = self.get_real_atom_mask(atype)
        if np.any(atype[real_mask] >= len(self.sys_charge_map)):
            raise ValueError("sys_charge_map does not cover all real atom types")
        if len(sel_type) != len(self.model_charge_map):
            raise ValueError(
                "model_charge_map length must match the dipole model sel_type length"
            )

        real_charge = np.zeros_like(atype, dtype=np.float64)
        real_charge[real_mask] = np.asarray(self.sys_charge_map)[atype[real_mask]]
        selected_mask = self.get_selected_wc_mask(atype, sel_type)
        selected_type = atype[selected_mask]
        wc_charge_by_type = {
            atom_type: self.model_charge_map[index]
            for index, atom_type in enumerate(sel_type)
        }
        wc_charge = np.asarray(
            [wc_charge_by_type[int(atom_type)] for atom_type in selected_type],
            dtype=np.float64,
        )
        return real_charge, wc_charge


@BaseModifier.register("dipole_charge")
class DipoleChargeModifier(DipoleChargeModifierBase, BaseModifier):
    """Backend-neutral serialized representation of dipole-charge."""
