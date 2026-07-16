# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Cartesian rank-2 tensor-product mixers for DPA4/SeZM.

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

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.cartesian``.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    xp_asarray_nodetach,
)
from deepmd.dpmodel.utils.network import (
    Identity,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.version import (
    check_version_compatibility,
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
    dtype: Any,
) -> np.ndarray:
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
    dtype : np.dtype
        Output dtype.

    Returns
    -------
    np.ndarray
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
    return np.array(matrices, dtype=dtype)


def build_edge_cartesian_tensors(
    r_hat: Any,
) -> tuple[Any, Any]:
    """
    Build the antisymmetric and symmetric-traceless edge tensors from unit vectors.

    Parameters
    ----------
    r_hat : Array
        Unit edge vectors with shape ``(E, 3)``.

    Returns
    -------
    tuple[Array, Array]
        Tuple containing (A0, S0), each with shape (E, 3, 3).
        - A0: The antisymmetric (l=1, vector) part, computed as skew(r_hat).
        - S0: The symmetric traceless (l=2, tensor) part, given by r_hat r_hat^T minus the identity matrix divided by 3.
        Both are 3x3 matrices which transform via matrix conjugation (M -> R M R^T) under rotation of r_hat, but occupy different irreducible SO(3) subspaces (l=1 for A0, l=2 for S0).
    """
    xp = array_api_compat.array_namespace(r_hat)
    rx, ry, rz = r_hat[:, 0], r_hat[:, 1], r_hat[:, 2]
    zero = xp.zeros_like(rx)
    a0 = xp.stack(
        [
            xp.stack([zero, -rz, ry], axis=-1),
            xp.stack([rz, zero, -rx], axis=-1),
            xp.stack([-ry, rx, zero], axis=-1),
        ],
        axis=-2,
    )  # (E, 3, 3)
    eye = xp.eye(3, dtype=r_hat.dtype, device=array_api_compat.device(r_hat))
    s0 = r_hat[..., None] * r_hat[..., None, :] - eye / 3.0  # (E, 3, 3)
    return a0, s0


class _CartesianTensorProduct(NativeOP):
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
    precision : str
        Parameter precision.
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
        precision: str = DEFAULT_PRECISION,
        seed: int | list[int] | None,
        trainable: bool,
    ) -> None:
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
        self.precision = precision
        self.compute_precision = str(
            np.dtype(get_promoted_dtype(PRECISION_DICT[self.precision])).name
        )
        self.activation_function = str(activation_function)
        self.mlp_bias = bool(mlp_bias)
        self.trainable = bool(trainable)

        # Separate seed namespaces so the linear and activation seeds never
        # collide regardless of ``n_layers``.
        seed_linears = child_seed(seed, 0)
        seed_activations = child_seed(seed, 1)

        # === Step 1. Per-degree channel linears (cross-degree mixing comes from
        # the matrix product, not the linear) ===
        self.linears = [
            SO3Linear(
                lmax=self.lmax,
                in_channels=self.focus_dim,
                out_channels=self.focus_dim,
                n_focus=self.n_focus,
                precision=self.precision,
                mlp_bias=mlp_bias,
                trainable=trainable,
                seed=child_seed(seed_linears, i),
            )
            for i in range(self.n_layers)
        ]

        # === Step 2. Gated nonlinearities; the last layer stays linear to mirror
        # the trailing identity of the SO(2) mixing stack ===
        activations: list[NativeOP] = []
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                activations.append(
                    GatedActivation(
                        lmax=self.lmax,
                        channels=self.focus_dim,
                        n_focus=self.n_focus,
                        precision=self.compute_precision,
                        activation_function=activation_function,
                        mlp_bias=mlp_bias,
                        layout="ndfc",
                        trainable=trainable,
                        seed=child_seed(seed_activations, i),
                    )
                )
            else:
                activations.append(Identity())
        self.activations = activations

    def _run_layers(
        self,
        h: Any,
        apply_product: Callable[[Any], Any],
    ) -> Any:
        """
        Run the residual tensor-product stack in packed ``(B, D, C_wide)`` layout.

        Each layer mixes channels per degree (``linear``), forms the equivariant
        ``3x3`` product (``apply_product``), and adds a gated-nonlinear residual.

        Parameters
        ----------
        h : Array
            Input features with shape ``(B, D, C_wide)``.
        apply_product : Callable[[Array], Array]
            Maps the per-degree channel-mixed feature ``y`` to the equivariant
            product term, both in ``(B, D, C_wide)`` layout.

        Returns
        -------
        Array
            Mixed features with shape ``(B, D, C_wide)``.
        """
        xp = array_api_compat.array_namespace(h)
        n = h.shape[0]
        d, f, cf, cw = self.ebed_dim, self.n_focus, self.focus_dim, self.c_wide
        for linear, activation in zip(self.linears, self.activations, strict=True):
            y = xp.reshape(linear(xp.reshape(h, (n, d, f, cf))), (n, d, cw))
            m = apply_product(y)
            h = h + xp.reshape(activation(xp.reshape(m, (n, d, f, cf))), (n, d, cw))
        return h

    def _sub_modules(self) -> list[tuple[str, NativeOP]]:
        """Sub-modules with their pt module names."""
        subs: list[tuple[str, NativeOP]] = []
        for i, linear in enumerate(self.linears):
            subs.append((f"linears.{i}", linear))
        for i, activation in enumerate(self.activations):
            subs.append((f"activations.{i}", activation))
        return subs

    def _variables(self) -> dict[str, Any]:
        """Variables keyed by the pt ``state_dict`` key names."""
        variables: dict[str, Any] = {}
        for prefix, sub in self._sub_modules():
            for key, value in sub.serialize().get("@variables", {}).items():
                variables[f"{prefix}.{key}"] = value
        return variables

    def _load_variables(self, variables: dict[str, Any]) -> None:
        """Load variables keyed by the pt ``state_dict`` key names."""
        for attr, sub in self._sub_modules():
            full = f"{attr}."
            sv = {
                key[len(full) :]: value
                for key, value in variables.items()
                if key.startswith(full)
            }
            data = sub.serialize()
            data["@variables"] = sv
            list_name, _, idx = attr.partition(".")
            getattr(self, list_name)[int(idx)] = type(sub).deserialize(data)


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
    precision : str
        Parameter precision.
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
        precision: str = DEFAULT_PRECISION,
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
            precision=precision,
            seed=seed,
            trainable=trainable,
        )
        self.eps = float(eps)

        # Non-persistent: a deterministic constant rebuilt on construction, so it
        # never enters the serialized state. The orthonormal basis ``B`` is
        # contracted into the right-multiply projection
        # ``W[p, d, k, j] = sum_i B[p, i, j] B[d, i, k]`` that maps an edge
        # component to its channel-shared packed operator (see ``call``).
        basis = build_cartesian_basis(
            self.lmax, dtype=PRECISION_DICT[self.precision.lower()]
        )
        self.right_mult_proj = np.einsum("pij,dik->pdkj", basis, basis)

    def call(
        self,
        x: Any,
        edge_vec: Any,
        rad_feat: Any,
    ) -> Any:
        """
        Parameters
        ----------
        x : Array
            Source node features in packed SO(3) layout with shape
            ``(E, D, C_wide)``, where ``D = (lmax + 1) ** 2`` and
            ``C_wide = n_focus * focus_dim``.
        edge_vec : Array
            Edge vectors with shape ``(E, 3)``, in Å.
        rad_feat : Array
            Per-degree radial weights with shape ``(E, lmax + 1, C_wide)``,
            already projected to the hidden width.

        Returns
        -------
        Array
            Edge messages in packed SO(3) layout with shape ``(E, D, C_wide)``.
        """
        xp = array_api_compat.array_namespace(x)
        device = array_api_compat.device(x)
        d = self.ebed_dim
        proj = xp.astype(
            xp_asarray_nodetach(xp, self.right_mult_proj[...], device=device),
            x.dtype,
        )

        # === Step 1. Channel-shared packed operators for the edge tensor ===
        # A(r_hat) and S(r_hat) are rescaled to unit Frobenius norm (raw norms are
        # sqrt(2) and sqrt(2/3)) so the per-degree radial weights modulate
        # components of equal magnitude. Each is projected into a packed
        # right-multiply operator of shape (E, D, D), shared by all channels; the
        # identity component reduces to the scalar ``c_iso`` rescaling in Step 3.
        r_hat = edge_vec / safe_norm(edge_vec, self.eps)  # (E, 3)
        a0, s0 = build_edge_cartesian_tensors(xp.astype(r_hat, x.dtype))
        a_hat = a0 / math.sqrt(2.0)
        # einsum "pdkj,ekj->epd" as a flattened matmul contracting the Cartesian
        # (k, j) entries against ``proj`` reshaped to ``(D * D, 3 * 3)``.
        k_op = xp.reshape(
            xp.matmul(
                xp.reshape(a_hat, (a_hat.shape[0], -1)),
                xp.permute_dims(xp.reshape(proj, (d * d, -1)), (1, 0)),
            ),
            (a_hat.shape[0], d, d),
        )  # (E, D, D)
        if self.lmax == 2:
            s_hat = s0 / math.sqrt(2.0 / 3.0)
            k_sym = xp.reshape(
                xp.matmul(
                    xp.reshape(s_hat, (s_hat.shape[0], -1)),
                    xp.permute_dims(xp.reshape(proj, (d * d, -1)), (1, 0)),
                ),
                (s_hat.shape[0], d, d),
            )  # (E, D, D)
            # Stack so one batched matmul per layer covers both components.
            k_op = xp.concat((k_op, k_sym), axis=1)  # (E, 2D, D)

        # === Step 2. Per-degree radial weights, broadcast over the degree axis ===
        c_iso = (rad_feat[:, 0, :] / math.sqrt(3.0))[:, None, :]  # (E, 1, C_wide)
        c_aniso = rad_feat[:, 1, :][:, None, :]  # (E, 1, C_wide)
        c_sym = (
            rad_feat[:, 2, :][:, None, :] if self.lmax == 2 else None
        )  # (E, 1, C_wide)

        # === Step 3. Edge-tensor modulation, optionally refined by a mixing stack ===
        def apply_product(y: Any) -> Any:
            ky = xp.matmul(k_op, y)  # (E, lmax * D, C_wide)
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

    def serialize(self) -> dict[str, Any]:
        """Serialize the EdgeCartesianTensorProduct to a dict."""
        return {
            "@class": "EdgeCartesianTensorProduct",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "focus_dim": self.focus_dim,
                "n_focus": self.n_focus,
                "n_layers": self.n_layers,
                "activation_function": self.activation_function,
                "mlp_bias": self.mlp_bias,
                "eps": self.eps,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> EdgeCartesianTensorProduct:
        """Deserialize an EdgeCartesianTensorProduct from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "EdgeCartesianTensorProduct":
            raise ValueError(
                f"Invalid class for EdgeCartesianTensorProduct: {data_cls}"
            )
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = dict(data.pop("config"))
        variables = data.pop("@variables")
        config["precision"] = str(config["precision"])
        obj = cls(**config)
        obj._load_variables(variables)
        return obj


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
    precision : str
        Parameter precision.
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
        precision: str = DEFAULT_PRECISION,
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
            precision=precision,
            seed=seed,
            trainable=trainable,
        )
        self.symmetric = bool(symmetric)
        self.basis = build_cartesian_basis(
            self.lmax, dtype=PRECISION_DICT[self.precision.lower()]
        )

    def call(self, message: Any, node: Any) -> Any:
        """
        Parameters
        ----------
        message : Array
            Aggregated message in packed SO(3) layout with shape
            ``(N, D, C_wide)``, where ``D = (lmax + 1) ** 2`` and
            ``C_wide = n_focus * focus_dim``. This is the residual stream.
        node : Array
            Destination node feature in the same packed layout and shape. It is
            the fixed right operand of the product across all layers.

        Returns
        -------
        Array
            Mixed message in packed SO(3) layout with shape ``(N, D, C_wide)``.
        """
        xp = array_api_compat.array_namespace(message)
        device = array_api_compat.device(message)
        basis = xp.astype(
            xp_asarray_nodetach(xp, self.basis[...], device=device),
            message.dtype,
        )

        # The node feature is the fixed per-node operator; lifting it to its
        # per-(node, channel) 3x3 form once lets every layer reuse it.
        # einsum "ndc,dij->ncij" as a flattened matmul over the degree axis.
        node_cart = xp.reshape(
            xp.matmul(
                xp.permute_dims(node, (0, 2, 1)),
                xp.reshape(basis, (basis.shape[0], -1)),
            ),
            (node.shape[0], node.shape[2], 3, 3),
        )  # (N, C_wide, 3, 3)

        def apply_product(y: Any) -> Any:
            # einsum "ndc,dij->ncij" as a flattened matmul over the degree axis.
            y_cart = xp.reshape(
                xp.matmul(
                    xp.permute_dims(y, (0, 2, 1)),
                    xp.reshape(basis, (basis.shape[0], -1)),
                ),
                (y.shape[0], y.shape[2], 3, 3),
            )  # (N, C_wide, 3, 3)
            m_cart = xp.matmul(y_cart, node_cart)
            if self.symmetric:
                m_cart = m_cart + xp.matmul(node_cart, y_cart)
            # einsum "ncij,dij->ndc" as a flattened matmul over the Cartesian
            # (i, j) entries, then transpose back to packed (N, D, C_wide).
            return xp.permute_dims(
                xp.matmul(
                    xp.reshape(m_cart, (m_cart.shape[0], m_cart.shape[1], -1)),
                    xp.permute_dims(xp.reshape(basis, (basis.shape[0], -1)), (1, 0)),
                ),
                (0, 2, 1),
            )  # (N, D, C_wide)

        return self._run_layers(message, apply_product)

    def serialize(self) -> dict[str, Any]:
        """Serialize the NodeCartesianTensorProduct to a dict."""
        return {
            "@class": "NodeCartesianTensorProduct",
            "@version": 1,
            "config": {
                "lmax": self.lmax,
                "focus_dim": self.focus_dim,
                "n_focus": self.n_focus,
                "n_layers": self.n_layers,
                "symmetric": self.symmetric,
                "activation_function": self.activation_function,
                "mlp_bias": self.mlp_bias,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
                "trainable": self.trainable,
                "seed": None,
            },
            "@variables": self._variables(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> NodeCartesianTensorProduct:
        """Deserialize a NodeCartesianTensorProduct from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "NodeCartesianTensorProduct":
            raise ValueError(
                f"Invalid class for NodeCartesianTensorProduct: {data_cls}"
            )
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = dict(data.pop("config"))
        variables = data.pop("@variables")
        config["precision"] = str(config["precision"])
        obj = cls(**config)
        obj._load_variables(variables)
        return obj
