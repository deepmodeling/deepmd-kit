# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Reciprocal-space energy of latent charges in periodic cells.

For the charges :math:`q_i^c` of a periodic cell of volume :math:`V` and a
pair kernel :math:`K` with Fourier transform :math:`\hat K`, the energy of the
lattice of images,

.. math::
    E = k_e \Big[ \frac{1}{2V} \sum_{\mathbf{k} \neq \mathbf{0}} \sum_c
    \hat K(\mathbf{k}) \, |\hat\rho_c(\mathbf{k})|^2
    - \frac{1}{2} K(0) \sum_{i,c} (q_i^c)^2 \Big],
    \qquad
    \hat\rho_c(\mathbf{k}) = \sum_i q_i^c e^{i \mathbf{k} \cdot \mathbf{r}_i},

is the sum over the wave numbers of the cell minus the interaction of every
charge with itself. The :math:`\mathbf{k} = \mathbf{0}` mode is dropped, the
tin-foil convention, which for a channel with a net charge amounts to a
uniform neutralizing background. The energy is attributed to the atoms as
:math:`E_i = \frac{1}{2} \sum_c q_i^c \phi_i^c`, with the potential
:math:`\phi_i^c = \partial E / \partial q_i^c` that the other charges and the
images create at atom :math:`i`, as the pair sum of a non-periodic system
splits every pair term evenly between its atoms.

Two evaluations serve the topologies of
:mod:`~deepmd.dpmodel.utils.neighbor_graph.reciprocal_graph`:
:func:`kspace_energy` sums over explicit wave numbers, and
:func:`mesh_energy` evaluates the same sum by the smooth particle-mesh Ewald
method on a mesh.
"""

import math
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    KGrid,
    MeshGrid,
)

from .kernel import (
    COULOMB_CONSTANT,
    BaseInteractionKernel,
)


def _self_potential(
    kernel: BaseInteractionKernel, charge: Array, periodic_node: Array
) -> Array:
    """The potential of every charge at itself, k_e K(0) q, on periodic nodes."""
    xp = array_api_compat.array_namespace(charge)
    return (
        COULOMB_CONSTANT
        * kernel.contact_value(charge)
        * charge
        * xp.astype(periodic_node, charge.dtype)[:, None]
    )


def _strained_rows(xp: Any, rows: Array, strain: Array) -> Array:
    r"""Wave numbers of the strained cells, to first order in the strain.

    The strain :math:`\varepsilon` of a frame maps its cell :math:`B` to
    :math:`B S^\mathsf{T}`, :math:`S = I + \varepsilon`, and a wave number,
    a row :math:`\mathbf{k}`, to :math:`\mathbf{k} S^{-1}`. The strain of a
    topology is zero, and the energy is differentiated once with respect to
    it, for the virial; the value and the first derivative at zero strain
    are those of :math:`S^{-1} = I - \varepsilon`, which maps the rows here
    and keeps linear algebra out of the energy and its derivatives.

    Parameters
    ----------
    rows
        Wave numbers or reciprocal lattice vectors with shape ``(nf, ..., 3)``.
    strain
        The strain of every frame with shape ``(nf, 3, 3)``.

    Returns
    -------
    Array
        The strained wave numbers, of the shape of ``rows``.
    """
    eps = xp.reshape(strain, (strain.shape[0],) + (1,) * (rows.ndim - 2) + (3, 3))
    return rows - xp.sum(rows[..., :, None] * eps, axis=-2)


def _strained_volume(volume: Array, strain: Array) -> Array:
    r"""Volumes of the strained cells, :math:`V \det S = V (1 + \mathrm{tr}\,\varepsilon)` to first order."""
    return volume * (1.0 + strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2])


def kspace_energy(
    charge: Array,
    k_grid: KGrid,
    kernel: BaseInteractionKernel,
) -> Array:
    r"""Energy of the periodic frames over explicit wave numbers, per atom.

    The structure factors of the wave numbers of every frame come from the
    phases of its nodes, and the potential at a node from the same phases:
    :math:`\phi_i = \frac{k_e}{V} \sum_{\mathbf{k}} \hat K(\mathbf{k})
    [\cos(\mathbf{k} \cdot \mathbf{r}_i) \mathrm{Re}\,\hat\rho +
    \sin(\mathbf{k} \cdot \mathbf{r}_i) \mathrm{Im}\,\hat\rho] - k_e K(0) q_i`.
    Every frame pairs the wave numbers of its row of :attr:`KGrid.vec` with
    the nodes of its row of :attr:`KGrid.slot`, in products batched over the
    frames and the blocks of their wave numbers. The geometry is strained by
    :attr:`KGrid.strain` (see :class:`KGrid`).

    Parameters
    ----------
    charge
        Node charges with shape ``(N, C)``, zero on the nodes that carry none.
    k_grid
        The wave numbers and the nodes of the periodic frames.
    kernel
        The pair kernel.

    Returns
    -------
    Array
        The energy of every node with shape ``(N,)``, in eV; zero in a frame
        without wave numbers.
    """
    xp = array_api_compat.array_namespace(charge)
    device = array_api_compat.device(charge)
    dtype = charge.dtype
    n_node, n_channel = charge.shape
    nf, n_slot = k_grid.slot.shape
    # === Step 1. The nodes of every frame, (nf, ns, 3) and (nf, ns, C) ===
    real = k_grid.slot >= 0
    index = xp.reshape(xp.where(real, k_grid.slot, xp.zeros_like(k_grid.slot)), (-1,))
    coord = xp.take(xp.astype(k_grid.coord, dtype), index, axis=0)
    coord = xp.reshape(coord, (nf, n_slot, 3))
    charge_slot = xp.reshape(xp.take(charge, index, axis=0), (nf, n_slot, n_channel))
    charge_slot = charge_slot * xp.astype(real, dtype)[:, :, None]
    # === Step 2. Strained wave numbers and volumes: k' = S^-T k, V' = V det S ===
    # Every product over the wave numbers of a frame broadcasts the per-frame
    # factor over the blocks, (nf, nb, kb, .), so that its sum runs over the
    # points of a block.
    strain = xp.astype(k_grid.strain, dtype)
    k_vec = xp.astype(k_grid.vec, dtype)
    strained_k = _strained_rows(xp, k_vec, strain)
    volume = _strained_volume(xp.astype(k_grid.volume, dtype), strain)
    # === Step 3. Structure factors, (nf, nb, kb, C) ===
    # The phase k' . r' = k . r does not change under the strain.
    phase = xp.matmul(k_vec, xp.matrix_transpose(coord)[:, None, :, :])
    cos_phase = xp.cos(phase)
    sin_phase = xp.sin(phase)
    rho_real = xp.matmul(cos_phase, charge_slot[:, None, :, :])
    rho_imag = xp.matmul(sin_phase, charge_slot[:, None, :, :])
    # === Step 4. Potential at the nodes of every frame, (nf, ns, C) ===
    weight = COULOMB_CONSTANT * kernel.kspace(xp.sum(strained_k * strained_k, axis=-1))
    weight = weight / volume[:, None, None]
    weight = xp.where(k_grid.mask, weight, xp.zeros_like(weight))[..., None]
    block_potential = xp.matmul(xp.matrix_transpose(cos_phase), weight * rho_real)
    block_potential = block_potential + xp.matmul(
        xp.matrix_transpose(sin_phase), weight * rho_imag
    )
    potential = xp.sum(block_potential, axis=1)
    # The self term belongs to the lattice sum of a frame with wave numbers;
    # a non-periodic batch evaluated by a periodic module carries none.
    has_points = xp.astype(xp.any(k_grid.mask, axis=(1, 2)), dtype)[:, None, None]
    potential = (
        potential
        - COULOMB_CONSTANT * kernel.contact_value(charge) * charge_slot * has_points
    )
    slot_energy = 0.5 * xp.sum(charge_slot * potential, axis=-1)
    # === Step 5. Energy on the node axis ===
    # A padding slot carries no charge and adds a zero to the first node.
    energy = xp.zeros((n_node,), dtype=dtype, device=device)
    return xp_add_at(energy, index, xp.reshape(slot_energy, (-1,)))


def bspline_integer_weights(order: int) -> np.ndarray:
    r"""The cardinal B-spline of order ``order`` at :math:`1, 2, \dots, p - 1`.

    These are the weights of the assignment factor
    :math:`D(\theta) = \sum_{l=0}^{p-2} M_p(l + 1) e^{i \theta l}` of the
    smooth particle-mesh Ewald method.
    """
    return np.asarray(
        [
            sum(
                (-1.0) ** j * math.comb(order, j) * max(l + 1.0 - j, 0.0) ** (order - 1)
                for j in range(order + 1)
            )
            / math.factorial(order - 1)
            for l in range(order - 1)
        ]
    )


def _bspline(xp: Any, x: Array, order: int) -> Array:
    r"""The cardinal B-spline :math:`M_p(x)` of order :math:`p`, elementwise.

    :math:`M_p(x) = \frac{1}{(p-1)!} \sum_{j=0}^{p} (-1)^j \binom{p}{j}
    \max(x - j, 0)^{p-1}`, a piecewise polynomial supported on
    :math:`[0, p]` with :math:`p - 2` continuous derivatives.
    """
    value = None
    for j in range(order + 1):
        term = ((-1.0) ** j * math.comb(order, j)) * xp.maximum(
            x - j, xp.zeros_like(x)
        ) ** (order - 1)
        value = term if value is None else value + term
    return value / math.factorial(order - 1)


def _inverse3(xp: Any, m: Array) -> Array:
    """Inverse of a batch of 3x3 matrices, its columns the cross products of the rows over the determinant."""
    a, b, c = m[..., 0, :], m[..., 1, :], m[..., 2, :]
    bc = xp.linalg.cross(b, c)
    ca = xp.linalg.cross(c, a)
    ab = xp.linalg.cross(a, b)
    det = xp.sum(a * bc, axis=-1)
    return xp.stack([bc, ca, ab], axis=-1) / det[..., None, None]


def _stencil(
    xp: Any, frac: Array, dims: tuple[Any, Any, Any], order: int
) -> tuple[Array, Array]:
    r"""The mesh points every node spreads to and their weights.

    Along axis :math:`d` the scaled coordinate :math:`u = n_d s_d` of the
    fractional coordinate :math:`s_d \in [0, 1)` reaches the ``order`` points
    :math:`\lfloor u \rfloor - p + 1, \dots, \lfloor u \rfloor` with the
    weights :math:`M_p(u - m)`; the stencil of a node is the outer product of
    its three axes.

    Returns
    -------
    tuple[Array, Array]
        The flat mesh indices and the weights, both with shape
        ``(N, order**3)``.
    """
    device = array_api_compat.device(frac)
    n_node = frac.shape[0]
    shift = xp.arange(order, dtype=frac.dtype, device=device)[None, :]
    offset = xp.arange(order, dtype=xp.int64, device=device)[None, :]
    index = None
    weight = None
    for d, n_d in enumerate(dims):
        u = frac[:, d] * n_d
        base = xp.floor(u)
        # The distance of point base - p + 1 + j from u is t + p - 1 - j.
        w = _bspline(xp, (u - base)[:, None] + (order - 1) - shift, order)
        m = (xp.astype(base, xp.int64)[:, None] - (order - 1) + offset) % n_d
        if index is None:
            index, weight = m, w
        else:
            index = xp.reshape(index[:, :, None] * n_d + m[:, None, :], (n_node, -1))
            weight = xp.reshape(weight[:, :, None] * w[:, None, :], (n_node, -1))
    return index, weight


def _assignment_factor(
    m_axes: list[Array], dims: tuple[Any, Any, Any], order: int
) -> list[Array]:
    r"""The inverse squared assignment factor :math:`|D_d(m)|^{-2}` of every axis.

    The B-spline spreading multiplies every mode of the charge mesh by
    :math:`\prod_d D_d(m_d)`, :math:`D_d(m) = \sum_l M_p(l + 1)
    e^{2\pi i m l / n_d}`, which the influence function divides out.
    """
    xp = array_api_compat.array_namespace(m_axes[0])
    weights = bspline_integer_weights(order)
    factors = []
    for m, n_d in zip(m_axes, dims, strict=True):
        theta = (2.0 * math.pi) * m / n_d
        real = xp.zeros_like(theta)
        imag = xp.zeros_like(theta)
        for l, w in enumerate(weights):
            real = real + float(w) * xp.cos(theta * l)
            imag = imag + float(w) * xp.sin(theta * l)
        factors.append(1.0 / (real * real + imag * imag))
    return factors


def mesh_energy(
    charge: Array,
    mesh: MeshGrid,
    frame_id: Array,
    kernel: BaseInteractionKernel,
    order: int,
) -> Array:
    r"""Energy of the periodic frames by the smooth particle-mesh Ewald method, per atom.

    The charges spread to the mesh of every frame with B-splines of order
    ``order`` along the fractional coordinates; a real FFT gives the half
    spectrum :math:`Q(\mathbf{m})` of the charge mesh. The influence function

    .. math::
        G(\mathbf{m}) = \frac{k_e}{V} \hat K(\mathbf{k}_\mathbf{m})
        \prod_d |D_d(m_d)|^{-2},
        \qquad \mathbf{k}_\mathbf{m} = 2\pi B^{-1} \mathbf{m},

    corrects for the spreading, so that the potential on the mesh is the
    inverse transform of :math:`G Q` and the potential at a node its
    interpolation by the same B-splines. :math:`|\mathbf{k}|^2` is the
    quadratic form :math:`(2\pi)^2 \mathbf{m}^\mathsf{T} W \mathbf{m}` of the
    reciprocal metric :math:`W = A^\mathsf{T} A`, :math:`A = S^{-\mathsf{T}}
    B^{-1}` of the strained cell, and the strain scales the volume by
    :math:`\det S`; the fractional coordinates do not depend on it. The zero
    mode carries no weight, nor do the modes at the Nyquist frequency of an
    axis, which have no unique wave number.

    Parameters
    ----------
    charge
        Node charges with shape ``(N, C)``, zero on the nodes that carry none.
    mesh
        The cells and the mesh of the periodic frames.
    frame_id
        The frame of every node with shape ``(N,)``.
    kernel
        The pair kernel.
    order
        The order of the B-splines.

    Returns
    -------
    Array
        The energy of every node with shape ``(N,)``, in eV; zero in a frame
        masked out by :attr:`MeshGrid.mask`.
    """
    xp = array_api_compat.array_namespace(charge)
    device = array_api_compat.device(charge)
    dtype = charge.dtype
    nf = mesh.box.shape[0]
    n_node, n_channel = charge.shape
    m_full = [xp.astype(m, dtype) for m in (mesh.mode_1, mesh.mode_2, mesh.mode_3)]
    n1, n2, n3 = (m.shape[0] for m in m_full)
    n_mesh = n1 * n2 * n3
    box = xp.astype(mesh.box, dtype)
    volume = xp.astype(mesh.volume, dtype)
    # === Step 1. Stencils of the nodes on the mesh ===
    box_inv = _inverse3(xp, box)
    # frac_d = sum_j r_j (B^-1)_jd, elementwise rather than a batched product
    # of 3x3 matrices.
    frac = xp.sum(
        xp.astype(mesh.coord, dtype)[:, :, None] * xp.take(box_inv, frame_id, axis=0),
        axis=1,
    )
    frac = frac - xp.floor(frac)
    index, weight = _stencil(xp, frac, (n1, n2, n3), order)
    # Flat targets on the frame- and channel-major meshes, (N, C, order**3).
    channel = xp.arange(n_channel, dtype=xp.int64, device=device)
    first = (frame_id[:, None] * n_channel + channel[None, :]) * n_mesh
    target = first[:, :, None] + index[:, None, :]
    # === Step 2. Charge mesh and its half spectrum ===
    q_mesh = xp.zeros((nf * n_channel * n_mesh,), dtype=dtype, device=device)
    q_mesh = xp_add_at(
        q_mesh,
        xp.reshape(target, (-1,)),
        xp.reshape(charge[:, :, None] * weight[:, None, :], (-1,)),
    )
    q_mesh = xp.reshape(q_mesh, (nf * n_channel, n1, n2, n3))
    spectrum = xp.fft.rfftn(q_mesh, axes=(-3, -2, -1))
    # === Step 3. Influence function on the half spectrum, (nf, n1, n2, h3) ===
    m3 = m_full[2][: n3 // 2 + 1]
    m_half = [m_full[0], m_full[1], m3]
    strain = xp.astype(mesh.strain, dtype)
    # The rows of A^T = B^-T S^-1 are the reciprocal vectors of the strained
    # cell, strained like wave numbers; W = A^T A.
    a_t = _strained_rows(xp, xp.matrix_transpose(box_inv), strain)
    metric = xp.sum(a_t[:, :, None, :] * a_t[:, None, :, :], axis=-1)
    volume = _strained_volume(volume, strain)
    g1 = m_half[0][:, None, None]
    g2 = m_half[1][None, :, None]
    g3 = m_half[2][None, None, :]
    # Six monomials of the mode indices times six entries of the metric of
    # every frame, elementwise.
    monomials = (g1 * g1, g2 * g2, g3 * g3, g1 * g2, g1 * g3, g2 * g3)
    entries = (
        metric[:, 0, 0],
        metric[:, 1, 1],
        metric[:, 2, 2],
        2.0 * metric[:, 0, 1],
        2.0 * metric[:, 0, 2],
        2.0 * metric[:, 1, 2],
    )
    k_sq = None
    for entry, monomial in zip(entries, monomials, strict=True):
        term = entry[:, None, None, None] * monomial[None, ...]
        k_sq = term if k_sq is None else k_sq + term
    k_sq = (2.0 * math.pi) ** 2 * k_sq
    # The zero mode and the Nyquist planes carry no weight.
    keep = (
        (2.0 * xp.abs(g1) != n1) & (2.0 * xp.abs(g2) != n2) & (2.0 * xp.abs(g3) != n3)
    )
    keep = keep & ((g1 != 0.0) | (g2 != 0.0) | (g3 != 0.0))
    keep = keep[None, ...] & mesh.mask[:, None, None, None]
    safe_k_sq = xp.where(keep, k_sq, xp.ones_like(k_sq))
    shape = safe_k_sq.shape
    # One row per frame and first mode index, along which the gradients of
    # the kernel parameters sum first.
    rows = xp.reshape(safe_k_sq, (nf * n1, n2 * (n3 // 2 + 1)))
    transform = xp.reshape(kernel.kspace(rows), shape)
    f1, f2, f3 = _assignment_factor(m_half, (n1, n2, n3), order)
    factor = f1[:, None, None] * f2[None, :, None] * f3[None, None, :]
    influence = (COULOMB_CONSTANT * transform * factor[None, ...]) / volume[
        :, None, None, None
    ]
    influence = xp.where(keep, influence, xp.zeros_like(influence))
    # === Step 4. Potential on the mesh and at the nodes ===
    spectrum = xp.reshape(spectrum, (nf, n_channel, n1, n2, n3 // 2 + 1))
    potential_mesh = xp.fft.irfftn(
        spectrum * influence[:, None, ...],
        s=(n1, n2, n3),
        axes=(-3, -2, -1),
        norm="forward",
    )
    stencil_potential = xp.take(
        xp.reshape(potential_mesh, (-1,)), xp.reshape(target, (-1,)), axis=0
    )
    potential = xp.sum(
        xp.reshape(stencil_potential, (n_node, n_channel, -1)) * weight[:, None, :],
        axis=-1,
    )
    periodic_node = xp.take(mesh.mask, frame_id, axis=0)
    potential = potential - _self_potential(kernel, charge, periodic_node)
    return 0.5 * xp.sum(charge * potential, axis=-1)
