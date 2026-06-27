# SPDX-License-Identifier: LGPL-3.0-or-later
"""Neuroevolution potential (NEP) descriptor.

This is the array-API reference implementation of the NEP descriptor introduced
by Fan et al. and implemented in GPUMD. The descriptor maps the local atomic
environment of every atom onto a fixed-length vector that is rotationally and
permutationally invariant, which is subsequently consumed by a per-element
fitting network to predict the atomic energy.

The descriptor is composed of two parts:

* radial part: for every radial expansion index ``n`` the descriptor sums a
  Chebyshev radial function over all neighbours within ``rcut_radial``.
* angular part: for every angular expansion index ``n`` and every angular order
  ``L`` the descriptor contracts the real solid harmonics of the neighbour
  directions (within ``rcut_angular``) into ``3``-body (and optionally ``4``- and
  ``5``-body) rotational invariants.

The expansion coefficients that turn the Chebyshev basis into the radial and
angular embeddings are the only trainable parameters of the descriptor. Because
the mapping ``g_n = sum_k c_{nk} f_k`` is linear in the basis, every centre/
neighbour type pair is represented as a single bias-free linear layer, stored in
a :class:`NetworkCollection`. This reuses the existing serialization and
backend-conversion machinery without any bespoke parameter handling.

References
----------
Z. Fan et al., "Neuroevolution potentials", J. Chem. Phys. 157, 114801 (2022).
The numerical conventions (cutoff, Chebyshev recursion, solid-harmonic
normalisation constants) follow GPUMD ``src/utilities/nep_utilities.cuh``.
"""

import math
from collections.abc import (
    Callable,
)
from typing import (
    Any,
    NoReturn,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_take_along_axis,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)

# Solid-harmonic normalisation constants, indexed by the flattened (L, m) layout
# ``L * L - 1 + k`` with ``k in [0, 2 * L]`` for ``L = 1 .. 8``. Copied verbatim
# from GPUMD ``nep_utilities.cuh`` (the authoritative reference).
NUM_OF_ABC = 80  # sum_{L=1}^{8} (2 L + 1)
C3B = [
    0.238732414637843,
    0.119366207318922,
    0.119366207318922,
    0.099471839432435,
    0.596831036594608,
    0.596831036594608,
    0.149207759148652,
    0.149207759148652,
    0.139260575205408,
    0.104445431404056,
    0.104445431404056,
    1.044454314040563,
    1.044454314040563,
    0.174075719006761,
    0.174075719006761,
    0.011190581936149,
    0.223811638722978,
    0.223811638722978,
    0.111905819361489,
    0.111905819361489,
    1.566681471060845,
    1.566681471060845,
    0.195835183882606,
    0.195835183882606,
    0.013677377921960,
    0.102580334414698,
    0.102580334414698,
    2.872249363611549,
    2.872249363611549,
    0.119677056817148,
    0.119677056817148,
    2.154187022708661,
    2.154187022708661,
    0.215418702270866,
    0.215418702270866,
    0.004041043476943,
    0.169723826031592,
    0.169723826031592,
    0.106077391269745,
    0.106077391269745,
    0.424309565078979,
    0.424309565078979,
    0.127292869523694,
    0.127292869523694,
    2.800443129521260,
    2.800443129521260,
    0.233370260793438,
    0.233370260793438,
    0.004662742473395,
    0.004079899664221,
    0.004079899664221,
    0.024479397985326,
    0.024479397985326,
    0.012239698992663,
    0.012239698992663,
    0.538546755677165,
    0.538546755677165,
    0.134636688919291,
    0.134636688919291,
    3.500553911901575,
    3.500553911901575,
    0.250039565135827,
    0.250039565135827,
    0.000082569397966,
    0.005944996653579,
    0.005944996653579,
    0.104037441437634,
    0.104037441437634,
    0.762941237209318,
    0.762941237209318,
    0.114441185581398,
    0.114441185581398,
    5.950941650232678,
    5.950941650232678,
    0.141689086910302,
    0.141689086910302,
    4.250672607309055,
    4.250672607309055,
    0.265667037956816,
    0.265667037956816,
]
C4B = [
    -0.007499480826664,
    -0.134990654879954,
    0.067495327439977,
    0.404971964639861,
    -0.809943929279723,
]
C5B = [0.026596810706114, 0.053193621412227, 0.026596810706114]

# Polynomial coefficients of the real solid harmonics in powers of ``z``.
# ``Z_COEFFICIENTS[L][n1][n2]`` multiplies ``z**n2`` for the harmonic of order
# ``L`` and azimuthal index ``n1``. Mirrors GPUMD ``Z_COEFFICIENT_{L}``.
Z_COEFFICIENTS: dict[int, list[list[float]]] = {
    1: [[0.0, 1.0], [1.0, 0.0]],
    2: [[-1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    3: [
        [0.0, -3.0, 0.0, 5.0],
        [-1.0, 0.0, 5.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ],
    4: [
        [3.0, 0.0, -30.0, 0.0, 35.0],
        [0.0, -3.0, 0.0, 7.0, 0.0],
        [-1.0, 0.0, 7.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    5: [
        [0.0, 15.0, 0.0, -70.0, 0.0, 63.0],
        [1.0, 0.0, -14.0, 0.0, 21.0, 0.0],
        [0.0, -1.0, 0.0, 3.0, 0.0, 0.0],
        [-1.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    6: [
        [-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0],
        [0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0],
        [1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0],
        [0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    7: [
        [0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0],
        [-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0],
        [0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0],
        [3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0],
        [0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    8: [
        [35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0],
        [0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0],
        [-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
}


def _nep_cutoff(distance: Array, rcut: float) -> Array:
    """Smooth NEP cutoff ``0.5 cos(pi r / rc) + 0.5`` for ``r < rc``, else ``0``.

    Parameters
    ----------
    distance : Array
        Pairwise distances, arbitrary shape.
    rcut : float
        Cutoff radius.

    Returns
    -------
    Array
        Cutoff values with the same shape as ``distance``.
    """
    xp = array_api_compat.array_namespace(distance)
    fc = 0.5 * xp.cos(math.pi * distance / rcut) + 0.5
    return xp.where(distance < rcut, fc, xp.zeros_like(fc))


def _chebyshev_basis(distance: Array, rcut: float, k_max: int) -> Array:
    """Evaluate the NEP Chebyshev radial basis functions.

    The ``k``-th basis is ``f_k(r) = 0.5 (T_k(x) + 1) f_c(r)`` with
    ``x = 2 (r / rc - 1)^2 - 1`` and ``T_k`` the Chebyshev polynomial of the
    first kind, following GPUMD ``find_fn``.

    Parameters
    ----------
    distance : Array
        Pairwise distances with shape ``(...,)``.
    rcut : float
        Cutoff radius.
    k_max : int
        Number of basis functions (``basis_size + 1``).

    Returns
    -------
    Array
        Basis values with shape ``(..., k_max)``.
    """
    xp = array_api_compat.array_namespace(distance)
    fc = _nep_cutoff(distance, rcut)
    x = 2.0 * (distance / rcut - 1.0) ** 2 - 1.0
    polys = [xp.ones_like(x)]
    if k_max > 1:
        polys.append(x)
        for _ in range(2, k_max):
            polys.append(2.0 * x * polys[-1] - polys[-2])
    # (..., k_max)
    poly = xp.stack(polys, axis=-1)
    return (poly + 1.0) * 0.5 * fc[..., None]


def _gather_neighbor_diff(coord_ext: Array, nlist: Array) -> tuple[Array, Array]:
    """Gather neighbour relative vectors for every centre atom.

    Parameters
    ----------
    coord_ext : Array
        Extended coordinates with shape ``(nf, nall * 3)`` or ``(nf, nall, 3)``.
    nlist : Array
        Neighbour list with shape ``(nf, nloc, nnei)``; negative entries denote
        padding.

    Returns
    -------
    diff : Array
        Relative vectors with shape ``(nf, nloc, nnei, 3)``, zeroed for padding.
    mask : Array
        Boolean validity mask with shape ``(nf, nloc, nnei)``.
    """
    xp = array_api_compat.array_namespace(coord_ext, nlist)
    nf, nloc, nnei = nlist.shape
    coord = xp.reshape(coord_ext, (nf, -1, 3))
    mask = nlist >= 0
    nlist_safe = nlist * xp.astype(mask, nlist.dtype)
    index = xp.tile(xp.reshape(nlist_safe, (nf, nloc * nnei, 1)), (1, 1, 3))
    # nf x nloc x nnei x 3
    coord_r = xp.reshape(xp_take_along_axis(coord, index, 1), (nf, nloc, nnei, 3))
    coord_l = xp.reshape(coord[:, :nloc, :], (nf, nloc, 1, 3))
    diff = (coord_r - coord_l) * xp.astype(mask[..., None], coord.dtype)
    return diff, mask


def _real_solid_harmonics(unit_vec: Array, l_max: int) -> Array:
    """Evaluate the (un-normalised) real solid harmonics of neighbour directions.

    Parameters
    ----------
    unit_vec : Array
        Unit direction vectors with shape ``(..., 3)``.
    l_max : int
        Maximum angular order ``L``.

    Returns
    -------
    Array
        Harmonics with shape ``(..., (l_max + 1) ** 2 - 1)`` ordered by the
        flattened ``L * L - 1 + k`` layout, matching GPUMD ``accumulate_s``.
    """
    xp = array_api_compat.array_namespace(unit_vec)
    x = unit_vec[..., 0]
    y = unit_vec[..., 1]
    z = unit_vec[..., 2]
    # === Step 1. Powers of z and complex powers (x + i y) ** m ===
    z_pow = [xp.ones_like(z)]
    for _ in range(1, l_max + 1):
        z_pow.append(z_pow[-1] * z)
    xy_real = [xp.ones_like(x)]
    xy_imag = [xp.zeros_like(y)]
    for _ in range(1, l_max + 1):
        xy_real.append(xy_real[-1] * x - xy_imag[-1] * y)
        xy_imag.append(xy_real[-2] * y + xy_imag[-1] * x)
    # === Step 2. Assemble harmonics column by column ===
    cols: list[Array] = []
    for ll in range(1, l_max + 1):
        z_coeff = Z_COEFFICIENTS[ll]
        for n1 in range(ll + 1):
            n2_start = 0 if (ll + n1) % 2 == 0 else 1
            z_factor = xp.zeros_like(z)
            for n2 in range(n2_start, ll - n1 + 1, 2):
                z_factor = z_factor + z_coeff[n1][n2] * z_pow[n2]
            if n1 == 0:
                cols.append(z_factor)
            else:
                cols.append(z_factor * xy_real[n1])
                cols.append(z_factor * xy_imag[n1])
    # (..., (l_max + 1) ** 2 - 1)
    return xp.stack(cols, axis=-1)


class NepEmbeddingCoeff(NativeOP):
    r"""Dense per-type-pair NEP expansion coefficients.

    Maps the Chebyshev basis of every edge onto the radial/angular embedding

    .. math::
        g^{ij}_n = \sum_k c_{t_i t_j n k}\, f_k(r_{ij}),

    where the coefficients :math:`c` are the only trainable parameters of the
    descriptor. They are stored as a single dense tensor of shape
    ``(ntypes, ntypes, n_desc, k_max)`` and applied through a gathered batched
    contraction, so the forward cost is ``O(n_edges)`` and independent of
    ``ntypes`` — unlike a per-pair network collection whose cost and object
    count grow as ``ntypes**2``.

    Parameters
    ----------
    ntypes : int
        Number of element types.
    n_desc : int
        Number of expansion channels (radial/angular descriptor dimension).
    k_max : int
        Chebyshev basis size.
    trainable : bool, default=True
        Whether the coefficients are trainable.
    precision : str
        Floating point precision of the coefficients.
    seed : int or list[int], optional
        Random seed for initialization.
    """

    def __init__(
        self,
        ntypes: int,
        n_desc: int,
        k_max: int,
        trainable: bool = True,
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None = None,
    ) -> None:
        self.ntypes = ntypes
        self.n_desc = n_desc
        self.k_max = k_max
        self.trainable = trainable
        self.precision = precision
        prec = PRECISION_DICT[precision]
        rng = np.random.default_rng(seed)
        self.coeff = rng.normal(
            scale=1.0 / np.sqrt(k_max + n_desc),
            size=(ntypes, ntypes, n_desc, k_max),
        ).astype(prec)

    def call(self, fn: Array, pair_index: Array) -> Array:
        """Gather the type-pair coefficients and contract with the basis.

        Parameters
        ----------
        fn : Array
            Chebyshev basis values with shape ``(..., k_max)``.
        pair_index : Array
            Flattened ordered type-pair index ``t_i * ntypes + t_j`` with shape
            ``(...)`` matching the leading axes of ``fn``.

        Returns
        -------
        Array
            The embedding ``g`` with shape ``(..., n_desc)``.
        """
        xp = array_api_compat.array_namespace(fn, pair_index)
        coeff = xp.reshape(
            self.coeff[...], (self.ntypes * self.ntypes, self.n_desc, self.k_max)
        )
        lead = pair_index.shape
        gathered = xp.take(coeff, xp.reshape(pair_index, (-1,)), axis=0)
        gathered = xp.reshape(gathered, (*lead, self.n_desc, self.k_max))
        # (..., n_desc, k_max) @ (..., k_max, 1) -> (..., n_desc)
        return xp.matmul(gathered, fn[..., None])[..., 0]

    def serialize(self) -> dict:
        """Serialize the coefficient table to dict."""
        return {
            "@class": "NepEmbeddingCoeff",
            "@version": 1,
            "ntypes": self.ntypes,
            "n_desc": self.n_desc,
            "k_max": self.k_max,
            "trainable": self.trainable,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "@variables": {"coeff": to_numpy_array(self.coeff)},
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NepEmbeddingCoeff":
        """Deserialize the coefficient table from dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        variables = data.pop("@variables")
        obj = cls(**data)
        obj.coeff = variables["coeff"]
        return obj


@BaseDescriptor.register("nep")
class DescrptNep(NativeOP, BaseDescriptor):
    r"""The NEP (neuroevolution potential) descriptor.

    The descriptor of atom :math:`i` concatenates a radial part and an angular
    part. The radial part reads

    .. math::
        q^i_n = \sum_{j \neq i} g_n(r_{ij}),
        \qquad g_n(r_{ij}) = \sum_{k} c^{t_i t_j}_{nk} f_k(r_{ij}),

    where :math:`f_k` are the Chebyshev radial basis functions. The angular part
    contracts the real solid harmonics :math:`Y_{Lm}` of the neighbour
    directions into rotational invariants

    .. math::
        s^i_{nLm} = \sum_{j \neq i} g_n(r_{ij}) Y_{Lm}(\hat{r}_{ij}),
        \qquad q^i_{nL} = \sum_{m} C^{(3)}_{Lm} (s^i_{nLm})^2,

    optionally augmented with ``4``-body and ``5``-body invariants. The full
    descriptor is normalised element-wise by a fixed scaler.

    Parameters
    ----------
    rcut_radial
        Cut-off radius of the radial part :math:`r_c^R`.
    rcut_angular
        Cut-off radius of the angular part :math:`r_c^A`. Must not exceed
        ``rcut_radial``.
    sel : list[int]
        ``sel[i]`` is the maximum number of type-``i`` neighbours within
        ``rcut_radial``.
    n_max_radial
        Maximum radial expansion index; the radial dimension is
        ``n_max_radial + 1``.
    n_max_angular
        Maximum angular expansion index; the angular radial dimension is
        ``n_max_angular + 1``.
    basis_size_radial
        Maximum radial Chebyshev index; the radial basis size is
        ``basis_size_radial + 1``.
    basis_size_angular
        Maximum angular Chebyshev index; the angular basis size is
        ``basis_size_angular + 1``.
    l_max
        Maximum angular order :math:`L` of the ``3``-body invariants.
    l_max_4body
        ``2`` to include the ``4``-body invariant, ``0`` to disable it.
    l_max_5body
        ``1`` to include the ``5``-body invariant, ``0`` to disable it. Requires
        ``l_max_4body == 2``.
    trainable
        Whether the expansion coefficients are trainable.
    precision
        Floating point precision of the expansion coefficients. Defaults to
        ``"float32"`` to match GPUMD; ``"float64"`` may be used for higher
        training precision, but note that GPUMD inference of an exported
        ``nep.txt`` is always single precision.
    type_map : list[str], optional
        The name of each type of atoms.
    seed : int, optional
        Random seed for parameter initialization.
    ntypes : int, optional
        Number of element types. Only for interface compatibility; inferred from
        ``sel``.

    Limitations
    -----------
    To stay consistent with GPUMD (so a trained model can be exported to
    ``nep.txt`` and run in GPUMD), this descriptor always uses a per-element
    fitting network and does not support type exclusion, per-type cutoffs, or the
    experimental higher-order angular invariants (``q_112``, ``q_123``,
    ``q_233``, ``q_134``).
    """

    _update_sel_cls = UpdateSel

    def __init__(
        self,
        rcut_radial: float = 8.0,
        rcut_angular: float = 4.0,
        sel: list[int] | None = None,
        n_max_radial: int = 6,
        n_max_angular: int = 6,
        basis_size_radial: int = 6,
        basis_size_angular: int = 6,
        l_max: int = 4,
        l_max_4body: int = 2,
        l_max_5body: int = 0,
        trainable: bool = True,
        precision: str = "float32",
        type_map: list[str] | None = None,
        seed: int | list[int] | None = None,
        ntypes: int | None = None,  # to be compat with input
    ) -> None:
        del ntypes  # inferred from sel
        if sel is None:
            raise ValueError("sel must be provided for the NEP descriptor")
        if rcut_angular > rcut_radial:
            raise ValueError("rcut_angular must not exceed rcut_radial")
        if l_max < 1 or l_max > 8:
            raise ValueError("l_max must be in [1, 8]")
        if l_max_4body not in (0, 2):
            raise ValueError("l_max_4body must be 0 or 2")
        if l_max_5body not in (0, 1):
            raise ValueError("l_max_5body must be 0 or 1")
        if l_max_5body == 1 and l_max_4body != 2:
            raise ValueError("the 5-body invariant requires l_max_4body == 2")

        self.rcut_radial = rcut_radial
        self.rcut_angular = rcut_angular
        self.sel = sel
        self.ntypes = len(sel)
        self.n_max_radial = n_max_radial
        self.n_max_angular = n_max_angular
        self.basis_size_radial = basis_size_radial
        self.basis_size_angular = basis_size_angular
        self.l_max = l_max
        self.l_max_4body = l_max_4body
        self.l_max_5body = l_max_5body
        self.trainable = trainable
        self.precision = precision
        self.type_map = type_map
        self.seed = seed

        # Derived dimensions following the NEP / GPUMD naming convention.
        self.n_desc_radial = n_max_radial + 1
        self.n_desc_angular = n_max_angular + 1
        self.k_max_radial = basis_size_radial + 1
        self.k_max_angular = basis_size_angular + 1
        self.num_l = l_max + (l_max_4body == 2) + (l_max_5body == 1)
        self.n_abc = (l_max + 1) ** 2 - 1

        # The Chebyshev->embedding coefficients are stored as a single dense
        # tensor per part and applied through a gathered contraction, keeping the
        # forward cost O(n_edges) regardless of the number of element types.
        self.radial_coeff = NepEmbeddingCoeff(
            self.ntypes,
            self.n_desc_radial,
            self.k_max_radial,
            trainable=self.trainable,
            precision=self.precision,
            seed=child_seed(self.seed, 0),
        )
        self.angular_coeff = NepEmbeddingCoeff(
            self.ntypes,
            self.n_desc_angular,
            self.k_max_angular,
            trainable=self.trainable,
            precision=self.precision,
            seed=child_seed(self.seed, 1),
        )

        self.sel_cumsum = [0, *np.cumsum(self.sel).tolist()]
        self.nnei = sum(self.sel)

        prec = PRECISION_DICT[self.precision]
        dim_out = self.get_dim_out()
        self.davg = np.zeros([dim_out], dtype=prec)
        self.dstd = np.ones([dim_out], dtype=prec)

    def __setitem__(self, key: str, value: Array) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.davg = value
        elif key in ("std", "data_std", "dstd"):
            self.dstd = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> Array:
        if key in ("avg", "data_avg", "davg"):
            return self.davg
        elif key in ("std", "data_std", "dstd"):
            return self.dstd
        else:
            raise KeyError(key)

    @property
    def dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return self.get_dim_out()

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return self.n_desc_radial + self.n_desc_angular * self.num_l

    def get_dim_emb(self) -> int:
        """Returns the embedding (g2) dimension; the NEP descriptor has none."""
        return 0

    def get_rcut(self) -> float:
        """Returns the (largest) cut-off radius."""
        return self.rcut_radial

    def get_rcut_smth(self) -> float:
        """Returns the smoothing onset radius.

        NEP uses a single cosine cutoff with no smoothing region; this value is
        only reported to the neighbor-list infrastructure and does not enter the
        descriptor.
        """
        return self.rcut_radial

    def get_sel(self) -> list[int]:
        """Returns the number of selected neighbors for each type."""
        return self.sel

    def mixed_types(self) -> bool:
        """NEP uses a type-resolved neighbor list and per-element fitting.

        This matches GPUMD, which always assigns a separate energy network to
        each element, so an exported ``nep.txt`` reproduces the trained model.
        """
        return False

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return False

    def has_message_passing_across_ranks(self) -> bool:
        """Returns whether per-layer node embeddings need MPI ghost exchange."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor needs sorted nlist for `forward_lower`."""
        return False

    def get_env_protection(self) -> float:
        """Returns the environment protection; unused by NEP (always zero)."""
        return 0.0

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Get the name to each type of atoms."""
        return self.type_map

    def share_params(
        self, base_class: Any, shared_level: Any, resume: bool = False
    ) -> NoReturn:
        """Share the parameters with the base class during multitask training."""
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> NoReturn:
        """Change the type related params to new ones."""
        raise NotImplementedError(
            "Descriptor nep does not support changing for type related params!"
        )

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """Compute the descriptor scaler ``1 / (q_max - q_min)`` from data.

        The raw (unscaled) descriptor is evaluated on the sampled frames and the
        per-channel range determines the scaler, following the NEP convention.
        The computation runs in the array namespace of ``self.dstd`` so that it
        works identically across the numpy, torch, and jax backends.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            Sampled data systems, or a lazy callable returning them.
        path : Optional[DPPath]
            Unused; kept for interface compatibility.
        """
        from deepmd.dpmodel.utils.nlist import (
            extend_input_and_build_neighbor_list,
        )

        del path
        sampled = merged() if callable(merged) else merged
        if not sampled:
            return
        xp = array_api_compat.array_namespace(self.dstd)
        device = array_api_compat.device(self.dstd)
        dtype = self.dstd.dtype
        prec = PRECISION_DICT[self.precision]

        # Evaluate the raw descriptor with the scaler temporarily disabled. The
        # per-channel range is accumulated in numpy so that the resulting scaler
        # is a detached constant: the descriptor output depends on the trainable
        # coefficients, and storing a graph-carrying tensor as ``dstd`` would tie
        # every training step to the (freed) statistics graph.
        davg_bak, dstd_bak = self.davg, self.dstd
        self.davg = xp.zeros_like(self.dstd)
        self.dstd = xp.ones_like(self.dstd)
        q_min = None
        q_max = None
        try:
            for system in sampled:
                coord = xp.asarray(
                    to_numpy_array(system["coord"]), dtype=dtype, device=device
                )
                atype = xp.asarray(to_numpy_array(system["atype"]), device=device)
                atype = xp.reshape(atype, (coord.shape[0], -1))
                coord = xp.reshape(coord, (atype.shape[0], -1, 3))
                box = system.get("box", None)
                if box is not None:
                    box = xp.asarray(to_numpy_array(box), dtype=dtype, device=device)
                coord_ext, atype_ext, _, nlist = extend_input_and_build_neighbor_list(
                    coord,
                    atype,
                    self.get_rcut(),
                    self.get_sel(),
                    mixed_types=self.mixed_types(),
                    box=box,
                )
                q = to_numpy_array(self.call(coord_ext, atype_ext, nlist)[0]).reshape(
                    -1, self.get_dim_out()
                )
                if q.shape[0] == 0:
                    continue
                batch_min = np.min(q, axis=0)
                batch_max = np.max(q, axis=0)
                q_min = batch_min if q_min is None else np.minimum(q_min, batch_min)
                q_max = batch_max if q_max is None else np.maximum(q_max, batch_max)
        finally:
            self.davg, self.dstd = davg_bak, dstd_bak
        if q_min is None:
            return
        diff = q_max - q_min
        dstd = np.where(diff > 1e-12, diff, np.ones_like(diff)).astype(prec)
        self.davg = np.zeros_like(dstd)
        self.dstd = dstd

    def set_stat_mean_and_stddev(self, mean: Array, stddev: Array) -> None:
        """Update mean and stddev (the descriptor scaler) for descriptor."""
        self.davg = mean
        self.dstd = stddev

    def get_stat_mean_and_stddev(self) -> tuple[Array, Array]:
        """Get mean and stddev (the descriptor scaler) for descriptor."""
        return self.davg, self.dstd

    @cast_precision
    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        comm_dict: dict | None = None,
        charge_spin: Array | None = None,
    ) -> tuple[Array, None, None, None, Array]:
        """Compute the NEP descriptor.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nall x 3)
        atype_ext
            The extended atom types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            Not used by this descriptor.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x dim_out
        rot_mat
            ``None``; the NEP descriptor exposes no equivariant representation.
        g2
            ``None``.
        h2
            ``None``.
        sw
            The radial cutoff switch. shape: nf x nloc x nnei
        """
        del mapping, fparam, comm_dict, charge_spin
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nf, nloc, nnei = nlist.shape
        m = nf * nloc

        # === Step 1. Neighbour geometry (relative vectors and distances) ===
        # ``diff`` is zeroed for padded neighbours. ``safe_for_vector_norm`` keeps
        # the gradient finite (zero) at the padded entries where the distance is 0.
        diff, mask = _gather_neighbor_diff(coord_ext, nlist)
        dist = safe_for_vector_norm(diff, axis=-1)
        safe_dist = xp.where(dist > 0.0, dist, xp.ones_like(dist))
        # nf x nloc x nnei x 3
        unit_vec = diff / safe_dist[..., None]

        valid = xp.astype(mask, diff.dtype)
        radial_valid = valid * xp.astype(dist < self.rcut_radial, diff.dtype)
        angular_valid = valid * xp.astype(dist < self.rcut_angular, diff.dtype)

        # === Step 2. Ordered type-pair index for every edge ===
        # The neighbour types are gathered from ``atype_ext`` to index the dense
        # coefficient table by the ordered (centre, neighbour) type pair.
        nlist_safe = nlist * xp.astype(mask, nlist.dtype)
        neighbor_type = xp_take_along_axis(
            atype_ext, xp.reshape(nlist_safe, (nf, nloc * nnei)), 1
        )
        neighbor_type = xp.reshape(neighbor_type, (nf, nloc, nnei))
        atype_center = atype_ext[:, :nloc]
        pair_index = atype_center[:, :, None] * self.ntypes + neighbor_type
        pair_index = xp.astype(xp.reshape(pair_index, (m, nnei)), xp.int64)

        # === Step 3. Radial part: q_n = sum_j g_n(r_ij) ===
        fn_radial = _chebyshev_basis(dist, self.rcut_radial, self.k_max_radial)
        fn_radial = xp.reshape(fn_radial * radial_valid[..., None], (m, nnei, -1))
        gn_radial = self.radial_coeff.call(fn_radial, pair_index)
        q_radial = xp.sum(gn_radial, axis=1)

        # === Step 4. Angular part: s_{n,abc} = sum_j g_n(r_ij) Y_abc(r_hat_ij) ===
        fn_angular = _chebyshev_basis(dist, self.rcut_angular, self.k_max_angular)
        fn_angular = xp.reshape(fn_angular * angular_valid[..., None], (m, nnei, -1))
        harmonics = xp.reshape(
            _real_solid_harmonics(unit_vec, self.l_max), (m, nnei, self.n_abc)
        )
        gn_angular = self.angular_coeff.call(fn_angular, pair_index)
        # (m, n_desc_angular, n_abc) = (m, n_desc, nnei) @ (m, nnei, n_abc)
        sum_s = xp.matmul(xp.permute_dims(gn_angular, (0, 2, 1)), harmonics)
        q_angular = self._angular_invariants(sum_s)

        # === Step 5. Assemble, normalise, and reshape ===
        # angular layout is L-major to match the GPUMD descriptor ordering.
        q_angular = xp.reshape(
            xp.permute_dims(q_angular, (0, 2, 1)),
            (m, self.n_desc_angular * self.num_l),
        )
        descriptor = xp.concat([q_radial, q_angular], axis=-1)
        descriptor = (descriptor - self.davg[...]) / self.dstd[...]
        descriptor = xp.reshape(descriptor, (nf, nloc, self.get_dim_out()))

        sw = _nep_cutoff(dist, self.rcut_radial) * valid
        return descriptor, None, None, None, sw

    def _angular_invariants(self, sum_s: Array) -> Array:
        """Contract harmonic sums into rotational invariants.

        Parameters
        ----------
        sum_s : Array
            Harmonic sums with shape ``(m, n_desc_angular, n_abc)``.

        Returns
        -------
        Array
            Invariants with shape ``(m, n_desc_angular, num_l)``, ordered as the
            ``3``-body terms ``L = 1 .. l_max`` followed by the optional
            ``4``-body and ``5``-body terms.
        """
        xp = array_api_compat.array_namespace(sum_s)
        q_list: list[Array] = []
        # 3-body: q_L = C3B[start] s[start]^2 + 2 sum_{k>=1} C3B[start+k] s[start+k]^2
        for ll in range(1, self.l_max + 1):
            start = ll * ll - 1
            q = C3B[start] * sum_s[:, :, start] * sum_s[:, :, start]
            for k in range(1, 2 * ll + 1):
                comp = sum_s[:, :, start + k]
                q = q + 2.0 * C3B[start + k] * comp * comp
            q_list.append(q)
        # 4-body invariant from the L = 2 components (s[3:8]).
        if self.l_max_4body == 2:
            s3 = sum_s[:, :, 3]
            s4 = sum_s[:, :, 4]
            s5 = sum_s[:, :, 5]
            s6 = sum_s[:, :, 6]
            s7 = sum_s[:, :, 7]
            q4 = (
                C4B[0] * s3 * s3 * s3
                + C4B[1] * s3 * (s4 * s4 + s5 * s5)
                + C4B[2] * s3 * (s6 * s6 + s7 * s7)
                + C4B[3] * s6 * (s5 * s5 - s4 * s4)
                + C4B[4] * s4 * s5 * s7
            )
            q_list.append(q4)
        # 5-body invariant from the L = 1 components (s[0:3]).
        if self.l_max_5body == 1:
            s0_sq = sum_s[:, :, 0] * sum_s[:, :, 0]
            s12_sq = sum_s[:, :, 1] * sum_s[:, :, 1] + sum_s[:, :, 2] * sum_s[:, :, 2]
            q5 = (
                C5B[0] * s0_sq * s0_sq
                + C5B[1] * s0_sq * s12_sq
                + C5B[2] * s12_sq * s12_sq
            )
            q_list.append(q5)
        return xp.stack(q_list, axis=-1)

    def serialize(self) -> dict:
        """Serialize the descriptor to dict."""
        return {
            "@class": "Descriptor",
            "type": "nep",
            "@version": 1,
            "rcut_radial": self.rcut_radial,
            "rcut_angular": self.rcut_angular,
            "sel": self.sel,
            "n_max_radial": self.n_max_radial,
            "n_max_angular": self.n_max_angular,
            "basis_size_radial": self.basis_size_radial,
            "basis_size_angular": self.basis_size_angular,
            "l_max": self.l_max,
            "l_max_4body": self.l_max_4body,
            "l_max_5body": self.l_max_5body,
            "trainable": self.trainable,
            "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            "radial_coeff": self.radial_coeff.serialize(),
            "angular_coeff": self.angular_coeff.serialize(),
            "@variables": {
                "davg": to_numpy_array(self.davg),
                "dstd": to_numpy_array(self.dstd),
            },
            "type_map": self.type_map,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptNep":
        """Deserialize from dict."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        radial_coeff = data.pop("radial_coeff")
        angular_coeff = data.pop("angular_coeff")
        obj = cls(**data)
        obj["davg"] = variables["davg"]
        obj["dstd"] = variables["dstd"]
        obj.radial_coeff = NepEmbeddingCoeff.deserialize(radial_coeff)
        obj.angular_coeff = NepEmbeddingCoeff.deserialize(angular_coeff)
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = cls._update_sel_cls().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["rcut_radial"],
            local_jdata_cpy["sel"],
            False,
        )
        local_jdata_cpy["sel"] = sel
        return local_jdata_cpy, min_nbor_dist
