# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Cartesian rank-2 tensor-product mixers for SeZM.

For message-passing degree ``lmax <= 2`` the per-channel spherical-harmonic
feature ``(l = 0, 1, 2)`` is isomorphic to a rank-2 Cartesian tensor (a ``3x3``
matrix) that decomposes into a scalar (the trace), a vector (the antisymmetric
part), and a symmetric-traceless tensor. A matrix product of two such tensors
mixes these irreducible components while staying SO(3)-equivariant in the global
frame, because ``(R X R^T)(R Y R^T) = R (X Y) R^T`` for any rotation ``R``. This
replaces the rotate-to-local / ``SO2Linear`` stack / rotate-back core of
:class:`SO2Convolution` without constructing any Wigner-D rotation.

Two placements share the same scaffold (a per-degree channel linear, a gated
nonlinearity, and a residual stack) but differ in the right operand of the
``3x3`` product:

* :class:`EdgeCartesianTensorProduct` runs per edge, before aggregation. The
  right operand is the edge tensor
  ``T_e = f_iso I + f_aniso A(r_hat) + f_sym S(r_hat)``, whose per-degree radial
  weights ``f_*`` carry the edge condition. Because ``T_e`` depends only on the
  edge direction it is shared across channels, so the product is evaluated
  through channel-shared packed operators (below) without materializing any
  ``3x3`` matrix per channel. With ``n_layers = 0`` the message is the single
  modulation ``x @ T_e`` (no learnable channel-mixing layers); ``n_layers > 0``
  refines it with the residual stack.
* :class:`NodeCartesianTensorProduct` runs per node, after aggregation. It
  couples the aggregated message with the destination node feature through the
  product of ``linear(message)`` with ``node`` lifted by the orthonormal basis,
  serving as the Cartesian counterpart of the ``message_node`` grid product.
  Both operands are per-channel, so the product is the literal ``3x3`` form. The
  one-sided product ``linear(message) @ node`` is SO(3)-equivariant; the
  symmetrized product ``linear(message) @ node + node @ linear(message)``
  additionally preserves the parity of each irreducible component.

Placing the product per node makes its cost scale with the number of nodes
rather than the number of edges, which is the regime where the Cartesian form is
cheaper than the per-edge SO(2) rotation.

Channel-shared edge evaluation
------------------------------
A literal ``to_cart -> Y @ T_e -> from_cart`` round trip materializes a ``3x3``
matrix for every (edge, channel) pair and runs both basis changes once per
layer, which is memory-bandwidth and kernel-launch bound. Instead, for a fixed
edge the map ``y -> from_cart(to_cart(y) @ T_e)`` is linear in the packed
coefficient ``y`` and splits, by linearity of ``T_e``, into

    m = (f_iso / sqrt(3)) y + f_aniso (K_A y) + f_sym (K_S y),

where ``K_A`` and ``K_S`` are ``(D, D)`` packed-basis operators for
"right-multiply by ``A(r_hat)`` / ``S(r_hat)``". They depend only on the edge
direction, hence are shared across channels: building them once per edge turns
the per-layer geometry into a single channel-batched ``bmm(K, y)`` instead of
two per-channel basis changes plus a per-channel ``3x3`` product. The identity
component collapses to a scalar rescaling because the basis is orthonormal
(``<B_p, B_d> = delta_{pd}``).

With ``B`` the orthonormal packed-to-Cartesian basis, the projection from an
edge component ``G`` to its packed right-multiply operator is
``K_G[p, d] = sum_{k,j} W[p, d, k, j] G[k, j]`` with the fixed tensor
``W[p, d, k, j] = sum_i B[p, i, j] B[d, i, k]``. The per-degree overall scale of
``B`` is arbitrary (absorbed by the learnable layers), so it is chosen
orthonormal for an exact, transpose-free round trip.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
)

import torch
import torch.nn as nn

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.utils import (
    env,
)

from .activation import (
    GatedActivation,
)
from .indexing import (
    get_so3_dim_of_lmax,
)
from .so3 import (
    SO3Linear,
)
from .utils import (
    get_promoted_dtype,
    safe_norm,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


def build_cartesian_basis(
    lmax: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the orthonormal ``3x3`` basis aligned with the SeZM packed (l, m) layout.

    Entry ``basis[d]`` is the Cartesian image of the d-th packed
    spherical-harmonic coefficient, ordered ``l = 0``, ``l = 1`` (``m = -1, 0,
    +1``), ``l = 2`` (``m = -2 .. +2``). The basis is orthonormal under the
    Frobenius inner product; the inverse map reuses the same basis and the
    coefficient round trip is exact.

    The convention-critical part is the within-degree sign and ordering: it
    matches the SeZM ``WignerDCalculator`` (``l = 1`` follows its
    ``l1_perm``/``l1_sign``), which makes the basis intertwine the packed
    Wigner-D rotation with the Cartesian rotation ``X -> R X R^T``. The
    per-degree overall scale is free and absorbed by the learnable layers.

    Parameters
    ----------
    lmax : int
        Message-passing degree, must be 1 or 2.
    dtype : torch.dtype
        Output dtype.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Basis with shape ``(D, 3, 3)`` where ``D = (lmax + 1) ** 2``.

    Raises
    ------
    ValueError
        If ``lmax`` is not 1 or 2.
    """
    if lmax not in (1, 2):
        raise ValueError("Cartesian tensor product requires lmax in {1, 2}")
    a = 1.0 / math.sqrt(2.0)
    b = 1.0 / math.sqrt(3.0)
    c = 1.0 / math.sqrt(6.0)
    matrices: list[list[list[float]]] = [
        # l = 0 : isotropic (trace)
        [[b, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, b]],
        # l = 1 : antisymmetric, m = -1, 0, +1
        [[0.0, 0.0, -a], [0.0, 0.0, 0.0], [a, 0.0, 0.0]],
        [[0.0, a, 0.0], [-a, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, -a], [0.0, a, 0.0]],
    ]
    if lmax == 2:
        matrices += [
            # l = 2 : symmetric traceless, m = -2 .. +2
            [[0.0, -a, 0.0], [-a, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, a], [0.0, a, 0.0]],
            [[-c, 0.0, 0.0], [0.0, -c, 0.0], [0.0, 0.0, 2.0 * c]],
            [[0.0, 0.0, -a], [0.0, 0.0, 0.0], [-a, 0.0, 0.0]],
            [[a, 0.0, 0.0], [0.0, -a, 0.0], [0.0, 0.0, 0.0]],
        ]
    return torch.tensor(matrices, dtype=dtype, device=device)


def build_edge_cartesian_tensors(
    r_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the antisymmetric and symmetric-traceless edge tensors from unit vectors.

    Parameters
    ----------
    r_hat : torch.Tensor
        Unit edge vectors with shape ``(E, 3)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple containing (A0, S0), each with shape (E, 3, 3).
        - A0: The antisymmetric (l=1, vector) part, computed as skew(r_hat).
        - S0: The symmetric traceless (l=2, tensor) part, given by r_hat r_hat^T minus the identity matrix divided by 3.
        Both are 3x3 matrices which transform via matrix conjugation (M -> R M R^T) under rotation of r_hat, but occupy different irreducible SO(3) subspaces (l=1 for A0, l=2 for S0).
    """
    rx, ry, rz = r_hat[:, 0], r_hat[:, 1], r_hat[:, 2]
    zero = torch.zeros_like(rx)
    a0 = torch.stack(
        [
            torch.stack([zero, -rz, ry], dim=-1),
            torch.stack([rz, zero, -rx], dim=-1),
            torch.stack([-ry, rx, zero], dim=-1),
        ],
        dim=-2,
    )  # (E, 3, 3)
    eye = torch.eye(3, dtype=r_hat.dtype, device=r_hat.device)
    s0 = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2) - eye / 3.0  # (E, 3, 3)
    return a0, s0


class _CartesianTensorProduct(nn.Module):
    """
    Shared scaffold for the Cartesian rank-2 tensor-product mixers.

    Holds the per-degree channel linears, the gated nonlinearities, and the
    residual layer loop. Subclasses register the geometry buffer they need and
    define ``forward``; the only per-layer difference is the equivariant ``3x3``
    product supplied to :meth:`_run_layers`.

    Parameters
    ----------
    lmax : int
        Message-passing degree, must be 1 or 2.
    focus_dim : int
        Channel width per focus stream.
    n_focus : int
        Number of focus streams; the flattened channel width is
        ``n_focus * focus_dim``.
    n_layers : int
        Number of stacked tensor-product layers.
    activation_function : str
        Activation function for the intermediate gated nonlinearities.
    mlp_bias : bool
        Whether the per-degree channel linear carries an ``l = 0`` bias.
    dtype : torch.dtype
        Parameter dtype.
    seed : int | list[int] | None
        Base seed for deterministic initialization.
    trainable : bool
        Whether parameters are trainable.

    Raises
    ------
    ValueError
        If ``lmax`` is not 1 or 2, or ``n_layers`` is negative.
    """

    def __init__(
        self,
        *,
        lmax: int,
        focus_dim: int,
        n_focus: int,
        n_layers: int,
        activation_function: str,
        mlp_bias: bool,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__()
        if lmax not in (1, 2):
            raise ValueError("`lmax` must be 1 or 2 for the Cartesian tensor product")
        self.lmax = int(lmax)
        self.focus_dim = int(focus_dim)
        self.n_focus = int(n_focus)
        self.n_layers = int(n_layers)
        if self.n_layers < 0:
            raise ValueError("`n_layers` must be >= 0")
        self.ebed_dim = get_so3_dim_of_lmax(self.lmax)
        self.c_wide = self.n_focus * self.focus_dim
        self.dtype = dtype
        self.device = env.DEVICE
        self.compute_dtype = get_promoted_dtype(self.dtype)

        # Separate seed namespaces so the linear and activation seeds never
        # collide regardless of ``n_layers``.
        seed_linears = child_seed(seed, 0)
        seed_activations = child_seed(seed, 1)

        # === Step 1. Per-degree channel linears (cross-degree mixing comes from
        # the matrix product, not the linear) ===
        self.linears = nn.ModuleList(
            SO3Linear(
                lmax=self.lmax,
                in_channels=self.focus_dim,
                out_channels=self.focus_dim,
                n_focus=self.n_focus,
                dtype=self.dtype,
                mlp_bias=mlp_bias,
                trainable=trainable,
                seed=child_seed(seed_linears, i),
            )
            for i in range(self.n_layers)
        )

        # === Step 2. Gated nonlinearities; the last layer stays linear to mirror
        # the trailing identity of the SO(2) mixing stack ===
        activations: list[nn.Module] = []
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                activations.append(
                    GatedActivation(
                        lmax=self.lmax,
                        channels=self.focus_dim,
                        n_focus=self.n_focus,
                        dtype=self.compute_dtype,
                        activation_function=activation_function,
                        mlp_bias=mlp_bias,
                        layout="ndfc",
                        trainable=trainable,
                        seed=child_seed(seed_activations, i),
                    )
                )
            else:
                activations.append(nn.Identity())
        self.activations = nn.ModuleList(activations)

    def _run_layers(
        self,
        h: torch.Tensor,
        apply_product: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Run the residual tensor-product stack in packed ``(B, D, C_wide)`` layout.

        Each layer mixes channels per degree (``linear``), forms the equivariant
        ``3x3`` product (``apply_product``), and adds a gated-nonlinear residual.

        Parameters
        ----------
        h : torch.Tensor
            Input features with shape ``(B, D, C_wide)``.
        apply_product : Callable[[torch.Tensor], torch.Tensor]
            Maps the per-degree channel-mixed feature ``y`` to the equivariant
            product term, both in ``(B, D, C_wide)`` layout.

        Returns
        -------
        torch.Tensor
            Mixed features with shape ``(B, D, C_wide)``.
        """
        n = h.shape[0]
        d, f, cf, cw = self.ebed_dim, self.n_focus, self.focus_dim, self.c_wide
        for linear, activation in zip(self.linears, self.activations, strict=True):
            y = linear(h.reshape(n, d, f, cf)).reshape(n, d, cw)
            m = apply_product(y)
            h = h + activation(m.reshape(n, d, f, cf)).reshape(n, d, cw)
        return h


class EdgeCartesianTensorProduct(_CartesianTensorProduct):
    """
    Edge-wise Cartesian rank-2 tensor-product mixer (SO(3)-equivariant).

    Per edge, the source spherical-harmonic feature is mixed with the edge tensor
    ``T_e = f_iso I + f_aniso A(r_hat) + f_sym S(r_hat)``, whose per-degree radial
    weights ``f_*`` carry the edge condition. The product is evaluated through
    channel-shared packed operators (see the module docstring) so no ``3x3``
    matrix is materialized per channel. Stacking ``n_layers`` such products
    supplies the cross-degree mixing that the local-frame ``SO2Linear`` provided,
    but in the global frame and without any Wigner-D rotation.

    Parameters
    ----------
    lmax : int
        Message-passing degree, must be 1 or 2.
    focus_dim : int
        Channel width per focus stream.
    n_focus : int
        Number of focus streams; the flattened channel width is
        ``n_focus * focus_dim``.
    n_layers : int
        Number of stacked tensor-product layers.
    activation_function : str
        Activation function for the intermediate gated nonlinearities.
    mlp_bias : bool
        Whether the per-degree channel linear carries an ``l = 0`` bias.
    eps : float
        Epsilon for the edge-vector normalization.
    dtype : torch.dtype
        Parameter dtype.
    seed : int | list[int] | None
        Base seed for deterministic initialization.
    trainable : bool
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        focus_dim: int,
        n_focus: int,
        n_layers: int,
        activation_function: str,
        mlp_bias: bool,
        eps: float,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__(
            lmax=lmax,
            focus_dim=focus_dim,
            n_focus=n_focus,
            n_layers=n_layers,
            activation_function=activation_function,
            mlp_bias=mlp_bias,
            dtype=dtype,
            seed=seed,
            trainable=trainable,
        )
        self.eps = float(eps)

        # Non-persistent: a deterministic constant rebuilt on construction and
        # moved with the module, so it never enters the serialized state. The
        # orthonormal basis ``B`` is contracted into the right-multiply
        # projection ``W[p, d, k, j] = sum_i B[p, i, j] B[d, i, k]`` that maps an
        # edge component to its channel-shared packed operator (see ``forward``).
        basis = build_cartesian_basis(self.lmax, dtype=self.dtype, device=self.device)
        self.register_buffer(
            "right_mult_proj",
            torch.einsum("pij,dik->pdkj", basis, basis),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_vec: torch.Tensor,
        rad_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Source node features in packed SO(3) layout with shape
            ``(E, D, C_wide)``, where ``D = (lmax + 1) ** 2`` and
            ``C_wide = n_focus * focus_dim``.
        edge_vec : torch.Tensor
            Edge vectors with shape ``(E, 3)``, in Å.
        rad_feat : torch.Tensor
            Per-degree radial weights with shape ``(E, lmax + 1, C_wide)``,
            already projected to the hidden width.

        Returns
        -------
        torch.Tensor
            Edge messages in packed SO(3) layout with shape ``(E, D, C_wide)``.
        """
        d = self.ebed_dim
        proj = self.right_mult_proj.to(dtype=x.dtype)

        # === Step 1. Channel-shared packed operators for the edge tensor ===
        # A(r_hat) and S(r_hat) are rescaled to unit Frobenius norm (raw norms are
        # sqrt(2) and sqrt(2/3)) so the per-degree radial weights modulate
        # components of equal magnitude. Each is projected into a packed
        # right-multiply operator of shape (E, D, D), shared by all channels; the
        # identity component reduces to the scalar ``c_iso`` rescaling in Step 3.
        r_hat = edge_vec / safe_norm(edge_vec, self.eps)  # (E, 3)
        a0, s0 = build_edge_cartesian_tensors(r_hat.to(dtype=x.dtype))
        a_hat = a0 / math.sqrt(2.0)
        k_op = torch.einsum("pdkj,ekj->epd", proj, a_hat)  # (E, D, D)
        if self.lmax == 2:
            s_hat = s0 / math.sqrt(2.0 / 3.0)
            k_sym = torch.einsum("pdkj,ekj->epd", proj, s_hat)  # (E, D, D)
            # Stack so one batched matmul per layer covers both components.
            k_op = torch.cat((k_op, k_sym), dim=1)  # (E, 2D, D)

        # === Step 2. Per-degree radial weights, broadcast over the degree axis ===
        c_iso = (rad_feat[:, 0, :] / math.sqrt(3.0)).unsqueeze(1)  # (E, 1, C_wide)
        c_aniso = rad_feat[:, 1, :].unsqueeze(1)  # (E, 1, C_wide)
        c_sym = (
            rad_feat[:, 2, :].unsqueeze(1) if self.lmax == 2 else None
        )  # (E, 1, C_wide)

        # === Step 3. Edge-tensor modulation, optionally refined by a mixing stack ===
        def apply_product(y: torch.Tensor) -> torch.Tensor:
            ky = torch.bmm(k_op, y)  # (E, lmax * D, C_wide)
            m = c_iso * y + c_aniso * ky[:, :d, :]
            if c_sym is not None:
                m = m + c_sym * ky[:, d:, :]
            return m

        # ``n_layers == 0`` keeps only the edge-condition modulation ``x @ T_e``
        # (radial scale + directional cross-degree coupling), with no learnable
        # channel-mixing layers; ``n_layers > 0`` refines it with the residual
        # stack of per-degree channel linears.
        if self.n_layers == 0:
            return apply_product(x)

        return self._run_layers(x, apply_product)


class NodeCartesianTensorProduct(_CartesianTensorProduct):
    """
    Node-wise Cartesian rank-2 tensor-product mixer (SO(3)-equivariant).

    Applied per node after aggregation, this couples the aggregated message with
    the destination node feature, serving as the Cartesian counterpart of the
    ``message_node`` grid product. The node feature is the fixed operator and the
    message is the residual stream, so each layer forms the product of
    ``linear(message)`` with ``node`` lifted by the orthonormal basis, then adds
    a gated-nonlinear residual. There is no per-edge geometry, so the cost scales
    with the number of nodes instead of the number of edges.

    The ``symmetric`` flag selects the product form. The one-sided product
    ``linear(message) @ node`` is SO(3)-equivariant and cheapest. The symmetrized
    product ``linear(message) @ node + node @ linear(message)`` additionally
    gives each irreducible component a definite parity under spatial inversion
    (even scalar and symmetric-traceless parts, odd skew-symmetric part), which
    the one-sided product mixes, at the cost of a second matrix product.

    Parameters
    ----------
    lmax : int
        Node degree, must be 1 or 2.
    focus_dim : int
        Channel width per focus stream.
    n_focus : int
        Number of focus streams; the flattened channel width is
        ``n_focus * focus_dim``.
    n_layers : int
        Number of stacked tensor-product layers.
    symmetric : bool
        If True, use the parity-preserving symmetrized product ``Y N + N Y``;
        if False, use the one-sided product ``Y N``.
    activation_function : str
        Activation function for the intermediate gated nonlinearities.
    mlp_bias : bool
        Whether the per-degree channel linear carries an ``l = 0`` bias.
    dtype : torch.dtype
        Parameter dtype.
    seed : int | list[int] | None
        Base seed for deterministic initialization.
    trainable : bool
        Whether parameters are trainable.
    """

    def __init__(
        self,
        *,
        lmax: int,
        focus_dim: int,
        n_focus: int,
        n_layers: int,
        symmetric: bool,
        activation_function: str,
        mlp_bias: bool,
        dtype: torch.dtype,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
        super().__init__(
            lmax=lmax,
            focus_dim=focus_dim,
            n_focus=n_focus,
            n_layers=n_layers,
            activation_function=activation_function,
            mlp_bias=mlp_bias,
            dtype=dtype,
            seed=seed,
            trainable=trainable,
        )
        self.symmetric = bool(symmetric)
        self.register_buffer(
            "basis",
            build_cartesian_basis(self.lmax, dtype=self.dtype, device=self.device),
            persistent=False,
        )

    def forward(self, message: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        message : torch.Tensor
            Aggregated message in packed SO(3) layout with shape
            ``(N, D, C_wide)``, where ``D = (lmax + 1) ** 2`` and
            ``C_wide = n_focus * focus_dim``. This is the residual stream.
        node : torch.Tensor
            Destination node feature in the same packed layout and shape. It is
            the fixed right operand of the product across all layers.

        Returns
        -------
        torch.Tensor
            Mixed message in packed SO(3) layout with shape ``(N, D, C_wide)``.
        """
        basis = self.basis.to(dtype=message.dtype)

        # The node feature is the fixed per-node operator; lifting it to its
        # per-(node, channel) 3x3 form once lets every layer reuse it.
        node_cart = torch.einsum("ndc,dij->ncij", node, basis)  # (N, C_wide, 3, 3)

        def apply_product(y: torch.Tensor) -> torch.Tensor:
            y_cart = torch.einsum("ndc,dij->ncij", y, basis)  # (N, C_wide, 3, 3)
            m_cart = torch.matmul(y_cart, node_cart)
            if self.symmetric:
                m_cart = m_cart + torch.matmul(node_cart, y_cart)
            return torch.einsum("ncij,dij->ndc", m_cart, basis)  # (N, D, C_wide)

        return self._run_layers(message, apply_product)
