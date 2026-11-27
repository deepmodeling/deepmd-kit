# SPDX-License-Identifier: LGPL-3.0-or-later
"""Analytical close-contact pair potentials, defined once for every backend.

Lives in the atomic-model package: the atomic layer owns per-atom energy
assembly, where the analytical term is injected on the graph route.
"""

import functools
import json
import math
from pathlib import (
    Path,
)
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
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.utils import (
    get_promoted_dtype,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.bridging import (
    DEFAULT_TRUNCATION,
)
from deepmd.utils.element_radii import (
    covalent_radii_from_type_map,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
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

# Coefficients of the NLH screening function, one row per unordered element
# pair, in the file beside this module.
#
# Provenance and attribution. The reference data is the open-data package
# K. Nordlund, G. Hobler and S. Lehtola, "Data sets for the publication
# 'Repulsive interatomic potentials calculated at three levels of theory'",
# Zenodo version 1.0, 16 November 2024, doi 10.5281/zenodo.14172633,
# distributed under the Creative Commons Attribution 4.0 International licence
# (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The functional
# form is that of K. Nordlund, S. Lehtola and G. Hobler, Phys. Rev. A 111,
# 032818 (2025), doi 10.1103/PhysRevA.111.032818, with its erratum
# Phys. Rev. A 112, 059901 (2025), doi 10.1103/cdrk-x7my, published under the
# same licence.
#
# The shipped numbers are MODIFIED material: they are this project's own refit
# of that open data and are not the coefficients published by Nordlund,
# Lehtola and Hobler. The fit imposes three physical conditions suited to
# cubic-Taylor truncated zone bridging: non-negative amplitudes (making the
# series completely monotone so the truncated remainder is positive,
# decreasing, convex and C^3 at the truncation radius), amplitudes summing to
# one (exact nuclear Coulomb limit at r -> 0), and live decay rates bounded
# between the universal ZBL outer screening rate and the united-atom 1s
# electron-density decay rate (preventing flat long-range tails and unphysical
# spikes). Three groups of rows use a source other than raw DMol energies:
# B-Ne (Z1 = 5, Z2 = 10) is fitted to the package's Hartree-Fock MP2 screening
# function; Sc-Dy (21, 66), Xe-Pr (54, 59) and Xe-Pb (54, 82), whose raw DMol
# files in the archive are truncated or mislabeled, use the normalized rows of
# ``nlh/nlh_coeffs.dat`` from the same archive; and every pair containing an
# element with Z > 92 uses the normalized Ziegler-Biersack-Littmark universal
# series. Neither the authors of the reference data nor the American Physical
# Society endorse this project.
#
# The same record, machine-readable, is in the file's ``meta`` entry and is
# returned by :func:`nlh_provenance`.
NLH_COEFFICIENTS_FILE = Path(__file__).with_name("nlh_coefficients.npz")
_SERIES_TERMS = 4
_PAIR_ROW = 16
_PAIR_EPS = 1e-10  # Å


def _truncated_pair_energy(radius: Array, row: Array) -> Array:
    r"""Evaluate the cubic Taylor remainder of a screened pair potential.

    With h = R - r, B = A exp(-cR) and K = -V'''(R)/6, multiplication
    by r gives r phi = K h^4 + sum B (exp(ch) - T3(ch)), where T3 is
    the cubic Taylor polynomial of exp at zero. Both contributions are
    positive for h > 0. For ch <= 1 the exponential remainder starts
    explicitly at fourth order; for ch > 1 its first term is evaluated
    as A exp(-cr), so no exponential with a positive argument is needed.

    Parameters
    ----------
    radius : Array
        Regularized separations with shape (E,) in Å.
    row : Array
        Pair constants with shape (E, 16), in the evaluation precision.

    Returns
    -------
    Array
        Pair energies with shape (E,) in eV.
    """
    xp = array_api_compat.array_namespace(radius)
    h = xp.maximum(row[:, 8] - radius, xp.zeros_like(radius))
    y = row[:, 4:8] * h[:, None]
    # At y <= 1 the first omitted term relative to y^4/4! is at
    # most 24/13! (single precision) or 24/19! (double precision).
    z = xp.where(y <= 1.0, y, xp.ones_like(y))
    order = 12 if radius.dtype == xp.float32 else 18
    tail = 1.0 / math.factorial(order)
    for k in range(order - 1, 3, -1):
        tail = 1.0 / math.factorial(k) + z * tail
    cubic = 1.0 + y * (1.0 + y * (0.5 + y / 6.0))
    endpoint = row[:, 12:16]
    remainder = xp.where(
        y <= 1.0,
        endpoint * (z * z) ** 2 * tail,
        row[:, :4] * xp.exp(-row[:, 4:8] * radius[:, None]) - endpoint * cubic,
    )
    return (row[:, 9] * (h * h) ** 2 + xp.sum(remainder, axis=-1)) / radius


def atomic_pair_energies(
    edge_vec: Array,
    edge_index: Array,
    atype: Array,
    edge_mask: Array,
    pair_table: Array,
    n_node: int,
) -> Array:
    """Evaluate directed pairs in table precision and sum their half-energies in FP64.

    Each valid edge contributes to its destination. The result has shape
    (n_node, 1) in eV. The fourth component of the regularized edge vector is
    the pair regularizer in Å, independent of descriptor geometry.
    """
    from deepmd.dpmodel.utils.neighbor_graph import (
        segment_sum,
    )

    xp = array_api_compat.array_namespace(edge_vec)
    table = xp_asarray_nodetach(
        xp, pair_table, device=array_api_compat.device(edge_vec)
    )
    vec = xp.astype(edge_vec, table.dtype, copy=False)
    radius = xp.linalg.vector_norm(
        xp.concat((vec, xp.full_like(vec[:, :1], _PAIR_EPS)), axis=-1), axis=-1
    )
    src = xp.astype(edge_index[0], xp.int64, copy=False)
    dst = xp.astype(edge_index[1], xp.int64, copy=False)
    types = xp.astype(atype, xp.int64, copy=False)
    types = xp.maximum(types, xp.zeros_like(types))
    stride = math.isqrt(table.shape[0])
    pair = xp.take(types, dst, axis=0) * stride + xp.take(types, src, axis=0)
    energy = _truncated_pair_energy(radius, xp.take(table, pair, axis=0))
    half = xp.where(
        xp.astype(edge_mask, xp.bool, copy=False), 0.5 * energy, xp.zeros_like(energy)
    )
    return xp.reshape(
        segment_sum(xp.astype(half, xp.float64, copy=False), dst, n_node), (n_node, 1)
    )


@functools.lru_cache(maxsize=1)
def _nlh_lookup() -> tuple[Array, Array]:
    r"""Load the NLH screening coefficients into charge-indexed lookups.

    Returns
    -------
    amplitude : Array
        Amplitudes :math:`a_k` with shape ``(Z_max + 1, Z_max + 1, 4)``,
        unitless and summing to one along the last axis; symmetric in the two
        charge axes, with row and column zero unused.
    rate : Array
        Decay rates :math:`b_k` with the same shape, in Å⁻¹.
    """
    with np.load(NLH_COEFFICIENTS_FILE, allow_pickle=False) as data:
        pairs, a, b = data["pairs"], data["a"], data["b"]
    # The file stores every unordered pair once, so the largest charge it
    # names is the highest index either axis has to address.
    z_max = int(pairs.max())
    amplitude = np.zeros((z_max + 1, z_max + 1, _SERIES_TERMS), dtype=np.float64)
    rate = np.zeros_like(amplitude)
    z1, z2 = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
    amplitude[z1, z2] = amplitude[z2, z1] = a
    rate[z1, z2] = rate[z2, z1] = b
    return amplitude, rate


def nlh_provenance() -> dict[str, Any]:
    """Read the provenance recorded inside the shipped coefficient file.

    Returns
    -------
    dict
        The reference data and article the coefficients derive from, the
        conditions and objective of the fit, and which reference covers which
        rows.
    """
    with np.load(NLH_COEFFICIENTS_FILE, allow_pickle=False) as data:
        return json.loads(str(data["meta"]))


class InnerPotential(NativeOP):
    r"""Analytical close-contact pair potential.

    Every supported potential is a screened Coulomb series in the pair
    separation,

    .. math::

       V_{ab}(r)=\frac1r\sum_{k=1}^{4}A_{abk}\,e^{-c_{abk}r},

    so one table of eight constants per ordered type pair describes all of
    them. The series ends at :math:`R_{ab}=f\,s_{ab}`, the fraction :math:`f`
    (``truncation``) of the pair's covalent bond length :math:`s_{ab}`, where
    its cubic Taylor polynomial is subtracted:

    .. math::

       \phi_{ab}(r)=V_{ab}(r)-\sum_{k=0}^{3}\frac{V_{ab}^{(k)}(R_{ab})}{k!}
       (r-R_{ab})^k\quad(r<R_{ab}),\qquad\phi_{ab}(r)=0\quad(r\ge R_{ab}).

    Every term of the series is completely monotone, so :math:`V^{(4)}>0`, and
    by Taylor's theorem :math:`\phi(r)=\int_r^R V^{(4)}(t)\,(t-r)^3/6\,dt` is
    non-negative, decreasing and convex, with value and first three
    derivatives zero at :math:`R` (C³). The subtracted polynomial is bounded at
    the origin, so the nuclear asymptote of :math:`V` is kept. A pair term
    that decreases and is convex in the separation adds a positive pressure and
    bulk modulus along every uniform compression, whatever the learned model
    does there, and it is exactly zero at long range.

    The constants are built in double precision and stored in the descriptor's
    compute precision. The term is evaluated on the edge form so its force
    and virial flow through the same edge backward as the learned energy.
    Each pair (i, j) contributes ``phi(r_ij) / 2`` to both
    atom i and atom j, avoiding double-counting from the symmetric neighbor
    list. The class is written against the array API, so every backend
    evaluates the same definition on its own array type.

    Parameters
    ----------
    type_map : list[str]
        Element symbols (e.g. ``["O", "H"]``). Index in this list
        corresponds to the ``atype`` integer values.
    mode : str
        Potential formula: ``"zbl"`` for the Ziegler-Biersack-Littmark
        universal potential, ``"nlh"`` for the bundled Nordlund-Lehtola-Hobler
        coefficients. Case-insensitive.
    truncation : float
        Radius at which the potential ends, as a fraction of each pair's
        covalent bond length.
    precision : str
        Descriptor precision. Half-precision descriptors evaluate geometry in FP32.

    Raises
    ------
    ValueError
        If ``mode`` is not recognized, if any element in ``type_map`` is
        not found in the periodic table, or if ``truncation`` is not finite
        and positive.
    """

    # Rebuilt from ``type_map``, ``mode`` and ``truncation`` by ``__init__``
    # and ``change_type_map``, so a wrapped backend keeps them out of its
    # checkpoints.
    CONFIG_DERIVED_ARRAYS = ("pair_table",)

    SUPPORTED_MODES = ("ZBL", "NLH")

    def __init__(
        self,
        type_map: list[str],
        mode: str = "zbl",
        truncation: float = DEFAULT_TRUNCATION,
        precision: str = "float64",
    ) -> None:
        super().__init__()
        mode = str(mode).upper()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unknown InnerPotential mode: {mode}")
        truncation = float(truncation)
        if not (math.isfinite(truncation) and truncation > 0.0):
            raise ValueError(
                f"`truncation` must be finite and positive, got {truncation}"
            )
        self.mode = mode
        self.truncation = truncation
        self.precision = np.dtype(get_promoted_dtype(PRECISION_DICT[precision])).name
        self.change_type_map(type_map)

    def _build_table(self, type_map: list[str]) -> Array:
        """Build the aligned rows [A, c, R, -V'''(R)/6, padding, A exp(-cR)]."""
        series = self._series_constants(type_map)
        table = np.zeros((series.shape[0], _PAIR_ROW), dtype=np.float64)
        table[:, :8] = series
        radii = covalent_radii_from_type_map(type_map)
        ends = table[:, 8].reshape(len(type_map) + 1, len(type_map) + 1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ends[:-1, :-1] = self.truncation * (radii[:, None] + radii[None, :])
            end = table[:, 8:9]
            inverse = 1.0 / np.where(end > 0, end, 1.0)
            z = series[:, 4:] * end
            endpoint = series[:, :4] * np.exp(-z)
            table[:, 12:16] = endpoint
            cubic = 1 + z * (1 + z * (0.5 + z / 6))
            table[:, 9] = (endpoint * cubic).sum(axis=1) * inverse[:, 0] ** 4
            table = table.astype(self.precision)
        if not np.isfinite(table).all():
            raise ValueError(
                f"Bridging truncation {self.truncation} produces non-finite pair "
                f"coefficients in {self.precision} for type map {type_map}."
            )
        return self._in_namespace_of(table, getattr(self, "pair_table", None))

    def _series_constants(self, type_map: list[str]) -> np.ndarray:
        """Build the four amplitudes and inverse screening lengths per pair in FP64.

        ZBL scales the universal screening rates by the inverse screening
        length of each pair. NLH supplies the pair-specific rates directly.
        Both series have amplitudes in eV Å and rates in inverse Å.
        """
        charge = self._lookup_from_type_map(type_map)
        ntypes = charge.shape[0]
        if self.mode == "NLH":
            lut_a, lut_b = _nlh_lookup()
            index = charge.astype(np.int64)
            amplitude = lut_a[index[:, None], index[None, :]]
            rate = lut_b[index[:, None], index[None, :]]
        else:
            screening = (
                0.88534 * _A_BOHR / (charge[:, None] ** 0.23 + charge[None, :] ** 0.23)
            )
            amplitude = np.broadcast_to(
                np.asarray(_ZBL_A_COEFF), (ntypes, ntypes, _SERIES_TERMS)
            )
            rate = np.asarray(_ZBL_B_COEFF) / screening[:, :, None]
        table = np.zeros((ntypes + 1, ntypes + 1, 2 * _SERIES_TERMS), dtype=np.float64)
        table[:ntypes, :ntypes, :_SERIES_TERMS] = (
            _KE_EV_A * (charge[:, None] * charge[None, :])[:, :, None] * amplitude
        )
        table[:ntypes, :ntypes, _SERIES_TERMS:] = rate
        return table.reshape(-1, 2 * _SERIES_TERMS)

    @staticmethod
    def _in_namespace_of(table: np.ndarray, like: Array | None) -> Array:
        """Place a NumPy table in the namespace and device of another array.

        Parameters
        ----------
        table : np.ndarray
            The table to place.
        like : Array, optional
            The array whose namespace and device the result adopts. ``None``
            returns the table unchanged.

        Returns
        -------
        Array
            The table, in ``like``'s namespace and on its device.
        """
        if like is None:
            return table
        xp = array_api_compat.array_namespace(like)
        return xp.asarray(table, device=array_api_compat.device(like))

    @staticmethod
    def _lookup_from_type_map(type_map: list[str]) -> np.ndarray:
        """Build the per-type nuclear-charge lookup from element symbols.

        Parameters
        ----------
        type_map : list[str]
            Element symbols; index corresponds to ``atype`` values.

        Returns
        -------
        np.ndarray
            Nuclear charges, shape ``(len(type_map),)``.

        Raises
        ------
        ValueError
            If an element symbol is not in :data:`ELEMENT_TO_Z`.
        """
        atomic_numbers = []
        for elem in type_map:
            z = ELEMENT_TO_Z.get(elem)
            if z is None:
                raise ValueError(f"Unknown element symbol: {elem}")
            atomic_numbers.append(z)
        return np.asarray(atomic_numbers, dtype=np.float64)

    def change_type_map(self, type_map: list[str]) -> None:
        """Rebuild the coefficient tables for a new type map.

        This class owns the coefficient tables, so it owns every update of
        them: the symbols, their count and the tables are one piece of state
        and are replaced together. Reordering, adding and dropping elements are
        all covered, because a table is rebuilt from the symbols rather than
        permuted, so no index bookkeeping can drift.

        Parameters
        ----------
        type_map : list[str]
            The new element symbols.
        """
        table = self._build_table(type_map)
        self.type_map = list(type_map)
        self.ntypes = len(type_map)
        self.pair_table = table

    def call(
        self,
        edge_vec: Array,
        edge_index: Array,
        atype_flat: Array,
        edge_mask: Array,
        n_node: int,
    ) -> Array:
        """Scatter per-edge analytical half-energies into per-atom energies.

        Parameters
        ----------
        edge_vec : Array
            (E, 3) edge vectors in Å (the autograd leaf on differentiable
            backends: differentiating the returned energy w.r.t. this input
            yields the analytical force/virial through the shared edge
            backward).
        edge_index : Array
            (2, E) ``[src, dst]`` edge endpoints (flat node indices).
        atype_flat : Array
            (N,) flat atom types.
        edge_mask : Array
            (E,) valid-edge mask.
        n_node : int
            Total flat node count ``N``.

        Returns
        -------
        Array
            Per-atom analytical energies with shape ``(1, n_node, 1)`` in
            ``edge_vec``'s dtype.
        """
        xp = array_api_compat.array_namespace(edge_vec)
        energy = atomic_pair_energies(
            edge_vec, edge_index, atype_flat, edge_mask, self.pair_table, n_node
        )
        return xp.astype(xp.reshape(energy, (1, n_node, 1)), edge_vec.dtype, copy=False)


def check_end_inside_cutoff(
    type_map: list[str], truncation: float, rcut: float
) -> None:
    """Refuse an analytical potential that the neighbor graph would cut off.

    A pair whose end radius lies at or beyond ``rcut`` leaves the graph while
    its term is still nonzero, which would make the energy jump.

    Parameters
    ----------
    type_map : list[str]
        Element symbols of the types.
    truncation : float
        End radius as a multiple of each pair's covalent bond length.
    rcut : float
        Cutoff radius of the neighbor graph in Å.

    Raises
    ------
    ValueError
        If the end radius of the largest pair reaches ``rcut``.
    """
    end = truncation * 2.0 * float(np.max(covalent_radii_from_type_map(list(type_map))))
    if end >= rcut:
        raise ValueError(
            f"The analytical potential of the largest pair ends at {end:.3f} Å, "
            f"at or beyond the cutoff {rcut} Å; enlarge `rcut` or lower the "
            "truncation (`bridging_truncation`)."
        )


@BaseAtomicModel.register("inner_potential")
class InnerPotentialAtomicModel(BaseAtomicModel):
    """Analytical close-contact pair potential as an ATOMIC MODEL.

    First-principles composition design: the analytical term maps local
    atomic environments to per-atom energies -- exactly the atomic-model
    contract -- so a bridged model is a SUM of two atomic energy models
    (the learned one and this one) via
    :class:`~deepmd.dpmodel.atomic_model.linear_atomic_model.LinearEnergyAtomicModel`
    with ``weights="sum"``, not a flag on the learned model. Graph-route
    only: the term is evaluated on the shared ``graph.edge_vec`` leaf so
    its force/virial ride the same edge backward as the learned energy;
    the dense (nlist) route raises.

    Parameters
    ----------
    type_map : list[str]
        Element symbols; index corresponds to ``atype`` values.
    rcut : float
        Cut-off radius this model declares (the composition uses the max
        over children; pass the learned model's).
    mode : str
        Potential formula, ``"zbl"`` or ``"nlh"``. Case-insensitive.
    sel : list[int] | int
        Neighbor selection this model declares (composition bookkeeping).
    truncation : float
        Radius at which the potential ends, as a fraction of each pair's
        covalent bond length; see :class:`InnerPotential`.
    precision : str
        Precision of the learned descriptor sharing the neighbor graph.

    Raises
    ------
    ValueError
        If the potential ends at or beyond ``rcut`` for some pair.
    """

    def __init__(
        self,
        type_map: list[str],
        rcut: float,
        mode: str = "zbl",
        sel: "list[int] | int" = 0,
        truncation: float = DEFAULT_TRUNCATION,
        precision: str = "float64",
        **kwargs: Any,
    ) -> None:
        super().__init__(type_map, **kwargs)
        self.potential = InnerPotential(
            type_map=list(type_map),
            mode=mode,
            truncation=truncation,
            precision=precision,
        )
        self.mode = self.potential.mode
        self.rcut = float(rcut)
        self.sel = (
            [int(s) for s in sel] if isinstance(sel, (list, tuple)) else [int(sel)]
        )
        check_end_inside_cutoff(self.type_map, self.potential.truncation, self.rcut)
        super().init_out_stat()

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        """Remap the output statistics, exclusions and ordered-pair coefficient table.

        Parameters
        ----------
        type_map : list[str]
            The new element symbols.
        model_with_new_type_stat : optional
            Model with statistics for the new types (unused: an analytical
            term has no fitted statistics).
        """
        check_end_inside_cutoff(type_map, self.potential.truncation, self.rcut)
        self.potential.change_type_map(type_map)
        super().change_type_map(
            type_map, model_with_new_type_stat=model_with_new_type_stat
        )

    def fitting_output_def(self) -> FittingOutputDef:
        """Per-atom analytical energy: reducible and fully differentiable."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                )
            ]
        )

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    def get_sel(self) -> list[int]:
        """Get the neighbor selection."""
        return self.sel

    def get_nsel(self) -> int:
        """Get the total neighbor selection."""
        return sum(self.sel)

    def mixed_types(self) -> bool:
        """The analytical term is type-agnostic in layout (mixed types)."""
        return True

    def has_message_passing(self) -> bool:
        """No message passing in an analytical pair term."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """No nlist ordering requirement (graph-route only)."""
        return False

    def get_dim_fparam(self) -> int:
        """No frame parameters."""
        return 0

    def get_dim_aparam(self) -> int:
        """No atomic parameters."""
        return 0

    def get_sel_type(self) -> list[int]:
        """All atom types contribute."""
        return []

    def is_aparam_nall(self) -> bool:
        """No atomic parameters."""
        return False

    def requires_full_neighbor_list(self) -> bool:
        """The graph-only pair potential retains every neighbor in range."""
        return True

    def uses_graph_lower(self) -> bool:
        """Graph-only term: the NeighborGraph lower is its sole evaluation
        route, so it supports the graph route like any graph-capable atomic
        model.
        """
        return True

    def graph_edge_dtype(self) -> str:
        """The term accepts single-precision edges, so under the composition
        rule the learned sibling decides the dtype of the shared edge tensor.
        """
        return "float32"

    def fused_decomposition(self) -> tuple[None, InnerPotential]:
        """The term is a pair potential with no learned part."""
        return None, self.potential

    def supports_canonical_graph(self) -> bool:
        """The term sums over the edges of any graph, without pair exclusion."""
        return self.pair_excl is None

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Analytical term: nothing to tabulate."""

    def compression_needs_min_nbor_dist(self) -> bool:
        """Analytical term: compression reads no neighbor statistics."""
        return False

    def forward_atomic(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """Dense route unsupported: the term rides the NeighborGraph route only."""
        raise NotImplementedError(
            "InnerPotentialAtomicModel rides the NeighborGraph route only; "
            "the dense (nlist) route has no injection site for the term"
        )

    def forward_atomic_graph(
        self,
        graph: Any,
        atype: Any,
        fparam: Any = None,
        aparam: Any = None,
        charge_spin: Any = None,
        spin: Any = None,
        comm_dict: dict | None = None,
        topologies: dict[str, Any] | None = None,
        node_mask: Any = None,
    ) -> dict:
        """Evaluate the analytical per-atom energy on the flat node axis.

        ``fparam``/``aparam``/``charge_spin``/``spin``/``comm_dict``/
        ``topologies``/``node_mask`` are accepted for pipeline-signature
        compatibility and ignored (the analytical term conditions on
        geometry and types only; the caller masks its outputs).

        Parameters
        ----------
        graph
            neighbor graph; ``graph.edge_vec`` is the differentiable edge
            leaf on autograd backends.
        atype
            flat local atom types. N

        Returns
        -------
        dict
            ``{"energy": (N, 1)}`` per-atom analytical energies.
        """
        import array_api_compat

        xp = array_api_compat.array_namespace(graph.edge_vec)
        n_node = atype.shape[0]
        energy = self.potential.call(
            graph.edge_vec,
            graph.edge_index,
            atype,
            graph.edge_mask,
            n_node=n_node,
        )
        return {"energy": xp.reshape(energy, (n_node, 1))}

    def serialize(self) -> dict:
        data = super().serialize()
        data.update(
            {
                "@class": "Model",
                "type": "inner_potential",
                "@version": 2,
                "mode": self.mode,
                "rcut": self.rcut,
                "sel": self.sel,
                "truncation": self.potential.truncation,
                "precision": self.potential.precision,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "InnerPotentialAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 2)
        data.pop("@class", None)
        data.pop("type", None)
        return super().deserialize(data)

    def set_case_embd(self, case_idx: int) -> None:
        """No case embedding in an analytical term."""

    def compute_or_load_stat(
        self,
        sampled_func: Any,
        stat_file_path: Any = None,
        compute_or_load_out_stat: bool = True,
        preset_observed_type: "list[str] | None" = None,
    ) -> None:
        """Analytical term: no statistics to compute."""
