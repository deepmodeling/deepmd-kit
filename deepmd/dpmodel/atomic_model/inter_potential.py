# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytical pair potentials for Zone bridging (backend-agnostic port of
``deepmd.pt``'s ``InterPotential``). Lives in the atomic-model package:
the atomic layer owns per-atom energy assembly, where the ZBL term is
injected on the graph route.
"""

from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
    xp_asarray_nodetach,
)
from deepmd.dpmodel.common import (
    NativeOP,
)

# fmt: off
ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
    "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
    "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
    "Og": 118,
}
# fmt: on

# ZBL screening function coefficients
_ZBL_A_COEFF = (0.18175, 0.50986, 0.28022, 0.028171)
_ZBL_B_COEFF = (3.1998, 0.94229, 0.4029, 0.20162)

# Physical constants
_KE_EV_A = 14.3996  # Coulomb constant in eV·Å
_A_BOHR = 0.5291772109  # Bohr radius in Å


class InterPotential(NativeOP):
    """Analytical pair potential for Zone bridging.

    Supports the Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion
    potential, evaluated on the edge form so that its force and virial flow
    through the same edge backward as the learned energy. Each pair (i, j)
    contributes ``V_ZBL(r_ij) / 2`` to both atom i and atom j, avoiding
    double-counting from the symmetric neighbor list. Backend-agnostic
    (array-API) port of the reference implementation in
    ``deepmd.pt.model.model.sezm_model.InterPotential``.

    Parameters
    ----------
    type_map : list[str]
        Element symbols (e.g. ``["O", "H"]``). Index in this list
        corresponds to the ``atype`` integer values.
    mode : str
        Potential formula. Currently only ``"zbl"`` is supported.

    Raises
    ------
    ValueError
        If ``mode`` is not recognized, or if any element in ``type_map`` is
        not found in the periodic table.
    """

    def __init__(self, type_map: list[str], mode: str = "zbl") -> None:
        super().__init__()
        mode = str(mode).upper()
        if mode != "ZBL":
            raise ValueError(f"Unknown InterPotential mode: {mode}")
        self.mode = mode
        self.type_map = list(type_map)
        self.ntypes_real = len(type_map)
        atomic_numbers = []
        for elem in type_map:
            z = ELEMENT_TO_Z.get(elem)
            if z is None:
                raise ValueError(f"Unknown element symbol: {elem}")
            atomic_numbers.append(z)
        self.atomic_numbers = np.asarray(atomic_numbers, dtype=np.float64)

    @staticmethod
    def _zbl_pair_energy(xp: Any, r: Array, zi: Array, zj: Array) -> Array:
        """Compute ZBL pair energy for given distances and nuclear charges.

        Parameters
        ----------
        xp
            The array namespace of ``r``/``zi``/``zj``.
        r : Array
            Pair distances with shape (...) in Å.
        zi : Array
            Nuclear charge of atom i with shape (...).
        zj : Array
            Nuclear charge of atom j with shape (...).

        Returns
        -------
        Array
            Pair energies with shape (...) in eV.
        """
        a_screen = 0.88534 * _A_BOHR / (zi**0.23 + zj**0.23)
        x = r / a_screen
        phi = sum(
            a_k * xp.exp(-b_k * x)
            for a_k, b_k in zip(_ZBL_A_COEFF, _ZBL_B_COEFF, strict=True)
        )
        return _KE_EV_A * zi * zj / r * phi

    def call(
        self,
        edge_vec: Array,
        edge_index: Array,
        atype_flat: Array,
        edge_mask: Array,
        n_node: int,
        real_type_count: int | None = None,
    ) -> Array:
        """Scatter per-edge ZBL half-energies into per-atom energies.

        Parameters
        ----------
        edge_vec : Array
            (E, 3) edge vectors in Å (the autograd leaf on differentiable
            backends: differentiating the returned energy w.r.t. this input
            yields the ZBL force/virial through the shared edge backward).
        edge_index : Array
            (2, E) ``[src, dst]`` edge endpoints (flat node indices).
        atype_flat : Array
            (N,) flat atom types.
        edge_mask : Array
            (E,) valid-edge mask.
        n_node : int
            Total flat node count ``N``.
        real_type_count : int | None
            Count of REAL atom types; types ``>= real_type_count``
            (virtual/placeholder) are wrapped back to their real parent for
            the Z lookup and masked out of the sum. Defaults to
            ``len(type_map)``.

        Returns
        -------
        Array
            Per-atom ZBL energies with shape ``(1, n_node, 1)`` in
            ``edge_vec``'s dtype.
        """
        xp = array_api_compat.array_namespace(edge_vec)
        device = array_api_compat.device(edge_vec)
        if real_type_count is None:
            real_type_count = self.ntypes_real
        src = xp.astype(edge_index[0, :], xp.int64)
        dst = xp.astype(edge_index[1, :], xp.int64)
        r = xp.linalg.vector_norm(xp.astype(edge_vec, xp.float64), axis=-1)
        r = xp.clip(r, min=1e-10)
        # Virtual/placeholder types wrap back to the real parent purely so
        # the Z lookup never indexes out of range; their edges are masked
        # out below.
        atype_i64 = xp.astype(atype_flat, xp.int64)
        atype_for_z = xp.clip(atype_i64, min=0)
        atype_for_z = xp.where(
            atype_for_z >= real_type_count,
            atype_for_z - real_type_count,
            atype_for_z,
        )
        z_all = xp.take(
            xp_asarray_nodetach(
                xp, self.atomic_numbers, dtype=xp.float64, device=device
            ),
            atype_for_z,
            axis=0,
        )
        zi = xp.take(z_all, src, axis=0)
        zj = xp.take(z_all, dst, axis=0)
        pair_e = self._zbl_pair_energy(xp, r, zi, zj)
        node_is_real = atype_i64 < real_type_count
        valid = (
            xp.astype(edge_mask, xp.bool)
            & xp.take(node_is_real, src, axis=0)
            & xp.take(node_is_real, dst, axis=0)
        )
        pair_e = pair_e * xp.astype(valid, pair_e.dtype)
        # Symmetric neighbor list: both directed edges exist, each scatters
        # half into its dst -- atoms i and j each receive V/2.
        from deepmd.dpmodel.utils.neighbor_graph import (
            segment_sum,
        )

        atom_energy = segment_sum(pair_e * 0.5, dst, n_node)
        return xp.astype(xp.reshape(atom_energy, (1, n_node, 1)), edge_vec.dtype)
