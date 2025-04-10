# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Union,
)

import numpy as np

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)


class Spin:
    """Class for spin, mainly processes the spin type-related information.
    Atom types can be split into three kinds:
    1. Real types: real atom species, "Fe", "H", "O", etc.
    2. Spin types: atom species with spin, as virtual atoms in input, "Fe_spin", etc.
    3. Placeholder types: atom species without spin, as placeholders in input without contribution,
    also name "H_spin", "O_spin", etc.
    For any types in 2. or 3., the type index is `ntypes` plus index of its corresponding real type.

    Parameters
    ----------
    use_spin: list[bool]
                A list of boolean values indicating whether to use atomic spin for each atom type.
                True for spin and False for not. List of bool values with shape of [ntypes].
    virtual_scale: list[float], float
                The scaling factor to determine the virtual distance
                between a virtual atom representing spin and its corresponding real atom
                for each atom type with spin. This factor is defined as the virtual distance
                divided by the magnitude of atomic spin for each atom type with spin.
                The virtual coordinate is defined as the real coordinate plus spin * virtual_scale.
                List of float values with shape of [ntypes] or [ntypes_spin] or one single float value for all types,
                only used when use_spin is True for each atom type.
    """

    def __init__(
        self,
        use_spin: list[bool],
        virtual_scale: Union[list[float], float],
    ) -> None:
        type_dtype = np.int32
        self.ntypes_real = len(use_spin)
        self.ntypes_spin = use_spin.count(True)
        self.use_spin = np.array(use_spin)
        self.spin_mask = self.use_spin.astype(np.int64)
        self.ntypes_real_and_spin = self.ntypes_real + self.ntypes_spin
        self.ntypes_placeholder = self.ntypes_real - self.ntypes_spin
        self.ntypes_input = 2 * self.ntypes_real  # with placeholder for input types
        self.real_type = np.arange(self.ntypes_real, dtype=type_dtype)
        self.spin_type = self.real_type[self.use_spin] + self.ntypes_real
        self.real_and_spin_type = np.concatenate([self.real_type, self.spin_type])
        self.placeholder_type = (
            np.arange(self.ntypes_real, dtype=type_dtype)[~self.use_spin]
            + self.ntypes_real
        )
        self.spin_placeholder_type = (
            np.arange(self.ntypes_real, dtype=type_dtype) + self.ntypes_real
        )
        self.input_type = np.arange(self.ntypes_real * 2, dtype=type_dtype)
        if isinstance(virtual_scale, list):
            if len(virtual_scale) == self.ntypes_real:
                self.virtual_scale = virtual_scale
            elif len(virtual_scale) == self.ntypes_spin:
                self.virtual_scale = np.zeros(
                    self.ntypes_real, dtype=GLOBAL_NP_FLOAT_PRECISION
                )
                self.virtual_scale[self.use_spin] = virtual_scale
            else:
                raise ValueError(
                    f"Invalid length of virtual_scale for spin atoms"
                    f": Expected {self.ntypes_real} or {self.ntypes_spin} but got {len(virtual_scale)}!"
                )
        elif isinstance(virtual_scale, float):
            self.virtual_scale = [virtual_scale for _ in range(self.ntypes_real)]
        else:
            raise ValueError(f"Invalid virtual scale type: {type(virtual_scale)}")
        self.virtual_scale = np.array(self.virtual_scale)
        self.virtual_scale_mask = (self.virtual_scale * self.use_spin).reshape([-1])
        self.pair_exclude_types = []
        self.init_pair_exclude_types_placeholder()
        self.atom_exclude_types_ps = []
        self.init_atom_exclude_types_placeholder_spin()
        self.atom_exclude_types_p = []
        self.init_atom_exclude_types_placeholder()

    def get_ntypes_real(self) -> int:
        """Returns the number of real atom types."""
        return self.ntypes_real

    def get_ntypes_spin(self) -> int:
        """Returns the number of atom types which contain spin."""
        return self.ntypes_spin

    def get_ntypes_real_and_spin(self) -> int:
        """Returns the number of real atom types and types which contain spin."""
        return self.ntypes_real_and_spin

    def get_ntypes_input(self) -> int:
        """Returns the number of double real atom types for input placeholder."""
        return self.ntypes_input

    def get_use_spin(self) -> list[bool]:
        """Returns the list of whether to use spin for each atom type."""
        return self.use_spin

    def get_virtual_scale(self) -> np.ndarray:
        """Returns the list of magnitude of atomic spin for each atom type."""
        return self.virtual_scale

    def init_pair_exclude_types_placeholder(self) -> None:
        """
        Initialize the pair-wise exclusion types for descriptor.
        The placeholder types for those without spin are excluded.
        """
        ti_grid, tj_grid = np.meshgrid(
            self.placeholder_type, self.input_type, indexing="ij"
        )
        self.pair_exclude_types = (
            np.stack((ti_grid, tj_grid), axis=-1).reshape(-1, 2).tolist()
        )

    def init_atom_exclude_types_placeholder_spin(self) -> None:
        """
        Initialize the atom-wise exclusion types for fitting.
        Both the placeholder types and spin types are excluded.
        """
        self.atom_exclude_types_ps = self.spin_placeholder_type.tolist()

    def init_atom_exclude_types_placeholder(self) -> None:
        """
        Initialize the atom-wise exclusion types for fitting.
        The placeholder types for those without spin are excluded.
        """
        self.atom_exclude_types_p = self.placeholder_type.tolist()

    def get_pair_exclude_types(self, exclude_types=None) -> list[tuple[int, int]]:
        """
        Return the pair-wise exclusion types for descriptor.
        The placeholder types for those without spin are excluded.
        """
        if exclude_types is None:
            return self.pair_exclude_types
        else:
            _exclude_types: list[tuple[int, int]] = copy.deepcopy(
                self.pair_exclude_types
            )
            for tt in exclude_types:
                assert len(tt) == 2
                _exclude_types.append((tt[0], tt[1]))
            return _exclude_types

    def get_atom_exclude_types(self, exclude_types=None) -> list[int]:
        """
        Return the atom-wise exclusion types for fitting before out_def.
        Both the placeholder types and spin types are excluded.
        """
        if exclude_types is None:
            return self.atom_exclude_types_ps
        else:
            _exclude_types: list[int] = copy.deepcopy(self.atom_exclude_types_ps)
            _exclude_types += exclude_types
            _exclude_types = list(set(_exclude_types))
            return _exclude_types

    def get_atom_exclude_types_placeholder(self, exclude_types=None) -> list[int]:
        """
        Return the atom-wise exclusion types for fitting after out_def.
        The placeholder types for those without spin are excluded.
        """
        if exclude_types is None:
            return self.atom_exclude_types_p
        else:
            _exclude_types: list[int] = copy.deepcopy(self.atom_exclude_types_p)
            _exclude_types += exclude_types
            _exclude_types = list(set(_exclude_types))
            return _exclude_types

    def get_spin_mask(self):
        """
        Return the spin mask of shape [ntypes],
        with spin types being 1, and non-spin types being 0.
        """
        return self.spin_mask

    def get_virtual_scale_mask(self):
        """
        Return the virtual scale mask of shape [ntypes],
        with spin types being its virtual scale, and non-spin types being 0.
        """
        return self.virtual_scale_mask

    def serialize(
        self,
    ) -> dict:
        return {
            "use_spin": self.use_spin.tolist(),
            "virtual_scale": self.virtual_scale.tolist(),
        }

    @classmethod
    def deserialize(
        cls,
        data: dict,
    ) -> "Spin":
        return cls(**data)
