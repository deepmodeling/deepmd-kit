# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Radial building blocks for the DPA4/SeZM descriptor.

This module defines the cutoff envelope, inner-distance clamp, radial basis,
and radial multilayer perceptron used by SeZM.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.radial``.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
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
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.bridging import (
    window_midpoint,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .norm import (
    RMSNorm,
)


class RadialMLP(NativeOP):
    """
    Radial MLP with optional channel RMSNorm and configurable activation.

    Parameters
    ----------
    mlp_layers : list[int]
        Layer sizes including input and output dimensions.
        E.g., [in_dim, hidden1, hidden2, out_dim].
    activation_function : str
        Activation function name (e.g., "silu", "tanh", "gelu").
    precision : str
        Floating point precision for the linear layers.
    trainable : bool
        Whether the parameters are trainable.
    radial_norm : bool
        Whether to insert a channel RMSNorm in each hidden layer.

    Architecture
    ------------
    ``radial_norm=True``  : Linear → RMSNorm → Activation for each hidden layer.
    ``radial_norm=False`` : Linear → Activation for each hidden layer.
    The final layer is always a plain Linear (no norm, no activation).

    Notes
    -----
    All bias terms are disabled (Linear bias=False, RMSNorm bias-free) to
    guarantee ``RadialMLP(0) = 0``. This is required because the compile path
    pads masked edges with zero ``edge_rbf``; any non-zero bias would leak
    spurious features into GIE scatter, causing energy divergence between
    compile and non-compile paths.

    The hidden RMSNorm normalizes each edge's radial features by their own RMS.
    When the input ``edge_rbf`` includes the C^3 cutoff envelope, it vanishes
    at ``rcut``. The RMSNorm divides that envelope out, and its ``eps``
    floor is crossed as the edge approaches ``rcut``. On a sparse neighborhood
    (e.g. a dimer) this floor-crossing produces a sharp kink in the potential
    energy surface just inside the cutoff. Setting ``radial_norm=False`` drops
    the RMSNorm so the radial features vanish smoothly with the envelope, which
    restores C^3 smoothness at the cutoff.
    """

    def __init__(
        self,
        mlp_layers: list[int],
        *,
        activation_function: str = "silu",
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        radial_norm: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if len(mlp_layers) < 2:
            raise ValueError("`mlp_layers` must have at least 2 elements")
        self.mlp_layers = list(mlp_layers)
        self.activation_function = str(activation_function)
        self.precision = precision
        self.trainable = bool(trainable)
        self.radial_norm = bool(radial_norm)

        modules: list = []
        n_layers = len(mlp_layers)
        for i in range(n_layers - 1):
            linear = NativeLayer(
                mlp_layers[i],
                mlp_layers[i + 1],
                bias=False,
                activation_function=None,
                precision=self.precision,
                seed=child_seed(seed, i),
                trainable=trainable,
            )
            modules.append(linear)
            # Last layer: no RMSNorm/activation
            if i < n_layers - 2:
                if self.radial_norm:
                    modules.append(
                        RMSNorm(
                            channels=mlp_layers[i + 1],
                            precision=self.precision,
                            trainable=trainable,
                        )
                    )
                modules.append(get_activation_fn(self.activation_function))

        self.net = modules

    def call(self, x: Any) -> Any:
        """
        Forward pass.

        Parameters
        ----------
        x : Array
            Input array with shape (..., mlp_layers[0]).

        Returns
        -------
        Array
            Output array with shape (..., mlp_layers[-1]).
        """
        for layer in self.net:
            x = layer(x)
        return x

    def serialize(self) -> dict[str, Any]:
        """Serialize the RadialMLP to a dict."""
        variables: dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.net):
            if isinstance(layer, NativeLayer):
                variables[f"{idx}.matrix"] = to_numpy_array(layer.w)
            elif isinstance(layer, RMSNorm):
                variables[f"{idx}.adam_scale"] = to_numpy_array(layer.adam_scale)
        return {
            "@class": "RadialMLP",
            "@version": 1,
            "mlp_layers": self.mlp_layers.copy(),
            "activation_function": self.activation_function,
            "dtype": np.dtype(PRECISION_DICT[self.precision]).name,
            "trainable": self.trainable,
            "radial_norm": self.radial_norm,
            "@variables": variables,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RadialMLP:
        """Deserialize a RadialMLP from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RadialMLP":
            raise ValueError(f"Invalid class for RadialMLP: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        variables = data.pop("@variables")
        data["precision"] = data.pop("dtype")
        obj = cls(**data)
        prec = PRECISION_DICT[obj.precision.lower()]
        for key, value in variables.items():
            idx, _, name = key.partition(".")
            layer = obj.net[int(idx)]
            if name == "matrix":
                layer.w = np.asarray(value, dtype=prec)
            else:
                layer.adam_scale = np.asarray(value, dtype=prec)
        return obj


class C3CutoffEnvelope(NativeOP):
    """
    C^3-continuous polynomial cutoff envelope function.

    This envelope provides a smooth transition to zero at the cutoff radius,
    ensuring continuity of the function value and the first three derivatives.

    Notes
    -----
    For scaled distance ``x = r / rcut`` and ``u = 1 - x``, the envelope is
    evaluated in the cancellation-free form::

        E_p(x) = u^4 * sum(comb(k + 3, 3) * x^k, k=0..p-1),  for x < 1
        E_p(x) = 0,                                             for x >= 1

    This positive-coefficient factorization satisfies::

        E(0) = 1,    E(1) = 0
        E'(1) = 0,   E''(1) = 0,   E'''(1) = 0

    For the default exponent ``p=5``::

        E_5(x) = u^4 * (1 + 4*x + 10*x^2 + 20*x^3 + 35*x^4)

    Parameters
    ----------
    rcut : float
        Cutoff radius in Å.
    exponent : int, optional
        Polynomial exponent (p), must be positive. Default is 5.

    Attributes
    ----------
    rcut : float
        Cutoff radius in Å.
    p : float
        Polynomial exponent.
    """

    def __init__(
        self,
        rcut: float,
        exponent: int = 5,
        *,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        if rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        if exponent <= 0:
            raise ValueError("`exponent` must be positive")
        self.rcut = float(rcut)
        self.p = int(exponent)
        self.precision = precision
        self._series_coefficients = tuple(
            float(math.comb(k + 3, 3)) for k in range(self.p)
        )

    def call(self, dst: Any) -> Any:
        """Compute the envelope value for given distances."""
        xp = array_api_compat.array_namespace(dst)
        device = array_api_compat.device(dst)
        u = xp.clip((self.rcut - dst) / self.rcut, min=0.0, max=1.0)
        x = 1.0 - u
        series = xp.full(
            x.shape,
            self._series_coefficients[-1],
            dtype=x.dtype,
            device=device,
        )
        for coefficient in reversed(self._series_coefficients[:-1]):
            series = coefficient + x * series
        return u**4 * series


def pair_contact(
    contact_radius: Any, center_type: Any, neighbor_type: Any, dtype: Any
) -> Any:
    """
    Gather the length scale of each edge from the types at its ends.

    Parameters
    ----------
    contact_radius : Any
        Per-type length scales in Å with shape (ntypes + 1,), the trailing
        entry belonging to the padding type.
    center_type : Any
        Type index of the center atom of every edge, with shape (E,).
    neighbor_type : Any
        Type index of the neighbor atom of every edge, with shape (E,).
    dtype : Any
        Floating-point dtype of the result. The edge distances are divided by
        that result, so this is the dtype the geometry is reduced in rather
        than the dtype of the incoming coordinates.

    Returns
    -------
    Any
        Contact distance of each edge in Å, with shape (E, 1).
    """
    xp = array_api_compat.array_namespace(center_type)
    radius = xp.asarray(
        contact_radius, dtype=dtype, device=array_api_compat.device(center_type)
    )
    contact = xp.take(radius, center_type) + xp.take(radius, neighbor_type)
    return contact[:, None]


class InnerClamp(NativeOP):
    """
    C3-continuous distance clamp of the DPA4 bridging window.

    The window of a pair is two dimensionless fractions of that pair's own
    length scale ``s``, the sum of the covalent radii of its two elements, so
    one window shape serves every element combination. In the reduced distance
    ``u = r / s`` the clamp freezes everything below the freeze point
    ``f_z`` at that value and returns to the identity at ``f_outer``::

        ũ = f_z                                             if u <= f_z
        ũ = f_z + (f_outer - f_z) * h(t)                    if f_z < u < f_outer
        ũ = u                                               if u >= f_outer

        h(t) = 20t^4 - 45t^5 + 36t^6 - 10t^7,  t = (u - f_z) / (f_outer - f_z)
        f_z = f_inner + 0.4 (f_outer - f_inner)

    and the clamped distance is ``r̃ = s * ũ``. Boundary conditions
    ``h(0)=0``, ``h(1)=1``, ``h'(0)=0``, ``h'(1)=1``, ``h''(0)=h''(1)=0``,
    ``h'''(0)=h'''(1)=0`` give C3 continuity: ``dr̃/dr = 0`` at the freeze
    point and ``dr̃/dr = 1`` at the outer radius (identity zone), with
    matched second and third derivatives at both.

    The freeze point sits two fifths into the window, one tenth of the width
    below the frame-filter point at the midpoint, where the training data of a
    bridged model end. The displayed distance therefore still moves at the
    filter point, with slope ``h'(1/6) = 0.22``, so the network can fit the
    labels just above it through the pair's own distance; frozen at the filter
    point itself, the learned pair energy could only follow the switch
    amplitude, whose slope at the midpoint pins the wall to a stiffness the
    labels do not have. Below the filter point the displayed distance keeps
    falling by ``0.6 h(1/6) = 0.62 %`` of the window width (5 mÅ for a C-C
    pair) before it freezes; that band is the only range of displayed
    distances no retained frame covers, and its shallowness is what bounds the
    network's extrapolation there. The switch of the window
    (:class:`BridgingSwitch`) spans the whole window, so below the freeze point
    the learned energy of the pair follows the switch amplitude alone.

    Parameters
    ----------
    f_inner : float
        Inner radius of the window as a fraction of the pair length scale.
    f_outer : float
        Outer radius as a fraction of the pair length scale. At or above it the
        clamp is the identity.

    Raises
    ------
    ValueError
        If ``f_inner >= f_outer`` or either is non-positive.
    """

    #: Position of the freeze point inside the window, as a fraction of its width.
    FREEZE_FRACTION = 0.4

    def __init__(self, f_inner: float, f_outer: float) -> None:
        if f_inner <= 0 or f_outer <= 0:
            raise ValueError("f_inner and f_outer must be positive")
        if f_inner >= f_outer:
            raise ValueError(f"f_inner ({f_inner}) must be < f_outer ({f_outer})")
        self.f_inner = float(f_inner)
        self.f_outer = float(f_outer)
        self.f_freeze = self.f_inner + self.FREEZE_FRACTION * (
            self.f_outer - self.f_inner
        )

    def call(self, r: Any, contact: Any) -> Any:
        """
        Apply the distance clamp.

        Parameters
        ----------
        r : Array
            Pair distances with shape (...) or (..., 1) in Å.
        contact : Array
            Length scale of each pair in Å, broadcastable against ``r``.

        Returns
        -------
        Array
            Clamped distances r̃ with the same shape as ``r``.
        """
        xp = array_api_compat.array_namespace(r)
        u = r / contact
        t = xp.clip(
            (u - self.f_freeze) / (self.f_outer - self.f_freeze), min=0.0, max=1.0
        )
        t2 = t * t
        t4 = t2 * t2
        # h(t) = 20t^4 - 45t^5 + 36t^6 - 10t^7
        # Satisfies:
        #   h(0)=0, h(1)=1
        #   h'(0)=0, h'(1)=1
        #   h''(0)=0, h''(1)=0
        #   h'''(0)=0, h'''(1)=0
        h = t4 * (20.0 + t * (-45.0 + t * (36.0 - 10.0 * t)))
        interpolated = contact * (self.f_freeze + (self.f_outer - self.f_freeze) * h)
        # Identity zone: u >= f_outer returns r directly.
        # Both branches have matching first three derivatives there,
        # so xp.where preserves C3 continuity here.
        return xp.where(u >= self.f_outer, r, interpolated)


class BridgingSwitch(NativeOP):
    r"""
    C3-continuous switching amplitude for the SeZM bridging zone.

    ``BridgingSwitch`` returns a per-edge scalar amplitude in ``[0, 1]``
    that measures how far an edge sits outside the frozen zone. It is
    the elementary piece the Source Freeze Propagation Gate (SFPG)
    aggregates into a per-node "non-frozen confidence" via a product
    over each source node's outgoing edges.

    The window of a pair is two dimensionless fractions of that pair's own
    length scale ``s``, the sum of the covalent radii of its two elements, so
    one window shape serves every element combination. In the reduced distance
    ``u = r / s``::

        w = 0                                            if u <= f_inner  (frozen)
        w = h((u - f_inner) / (f_outer - f_inner))       if f_inner < u < f_outer  (transition)
        w = 1                                            if u >= f_outer  (normal)

        h(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7

    Boundary conditions at ``t=0`` and ``t=1``::

        h(0)   = h'(0)   = h''(0)   = h'''(0)   = 0
        h(1)=1, h'(1)    = h''(1)   = h'''(1)   = 0

    The vanishing first three derivatives at both endpoints give
    ``w \in C^3(\mathbb{R}_{\ge 0})`` with zero slope/curvature at both
    radii of every pair, so forces (first derivatives) and the
    force derivatives consumed by second-order training stay continuous
    across both zone boundaries.

    The surrounding infrastructure (``compute_source_gates``) owns the
    per-node product reduction and the per-edge leave-one-out product; this
    module only encodes the scalar amplitude shape.

    Parameters
    ----------
    f_inner : float
        Inner radius as a fraction of the pair length scale. At or below it
        ``w = 0``.
    f_outer : float
        Outer radius as a fraction of the pair length scale. At or above it
        ``w = 1``.

    Raises
    ------
    ValueError
        If ``f_inner <= 0``, ``f_outer <= 0``, or ``f_inner >= f_outer``.
    """

    def __init__(self, f_inner: float, f_outer: float) -> None:
        if f_inner <= 0 or f_outer <= 0:
            raise ValueError("f_inner and f_outer must be positive")
        if f_inner >= f_outer:
            raise ValueError(f"f_inner ({f_inner}) must be < f_outer ({f_outer})")
        self.f_inner = float(f_inner)
        self.f_outer = float(f_outer)

    def call(self, r: Any, contact: Any) -> Any:
        """
        Evaluate the C3 switching amplitude.

        Parameters
        ----------
        r : Array
            Pair distances with shape (...) or (..., 1) in Å.
        contact : Array
            Length scale of each pair in Å, broadcastable against ``r``.

        Returns
        -------
        Array
            Switching amplitudes in ``[0, 1]`` with the same shape as ``r``.
        """
        xp = array_api_compat.array_namespace(r)
        t = xp.clip(
            (r / contact - self.f_inner) / (self.f_outer - self.f_inner),
            min=0.0,
            max=1.0,
        )
        t2 = t * t
        t4 = t2 * t2
        # h(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7  (Horner form).
        # Degree-7 smootherstep: the unique polynomial of this degree that
        # hits ``w = 0`` at the inner radius and ``w = 1`` at the outer one
        # together with C3 flatness at both.
        return t4 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))


class BridgingClamp(NativeOP):
    r"""
    C4-continuous distance clamp whose slope is the bridging switch.

    The clamp shares the window of :class:`BridgingSwitch` and is its integral:
    the distance the descriptor sees moves with the true one at exactly the
    rate at which the switch has opened. In the reduced distance ``u = r / s``,
    with ``s`` the length scale of the pair::

        ũ = f_mid                                          if u <= f_inner
        ũ = f_mid + (f_outer - f_inner) * S(t)             if f_inner < u < f_outer
        ũ = u                                              if u >= f_outer

        S(t) = 7 t^5 - 14 t^6 + 10 t^7 - 2.5 t^8,  t = (u - f_inner) / (f_outer - f_inner)

    and the clamped distance is ``r̃ = s * ũ``. ``S`` is the antiderivative of
    the septic smootherstep ``h`` of the switch, so ``dr̃/dr = h(t)``: zero
    below the inner radius, one above the outer radius, never above one in
    between. Since ``S(1) = 1/2``, continuity at the outer radius fixes the
    frozen value at the window midpoint ``f_mid = (f_inner + f_outer) / 2``.
    The vanishing first three derivatives of ``h`` at both ends make ``r̃`` a
    C4 function of ``r``.

    Used together with the switch, the clamp keeps the radial features of a
    close pair inside the range the training frames cover. The frame filter of
    a bridged model keeps pairs down to the window midpoint, the clamped
    distance never falls below it, and through the lower half of the window it
    rises by less than 7 % of the window width (``S(1/2) = 35/512``), so there
    the learned energy follows the switch amplitude.

    Parameters
    ----------
    f_inner : float
        Inner radius as a fraction of the pair length scale. At or below it the
        distance the descriptor sees is frozen at the window midpoint.
    f_outer : float
        Outer radius as a fraction of the pair length scale. At or above it the
        clamp is the identity.

    Raises
    ------
    ValueError
        If ``f_inner >= f_outer`` or either is non-positive.
    """

    def __init__(self, f_inner: float, f_outer: float) -> None:
        if f_inner <= 0 or f_outer <= 0:
            raise ValueError("f_inner and f_outer must be positive")
        if f_inner >= f_outer:
            raise ValueError(f"f_inner ({f_inner}) must be < f_outer ({f_outer})")
        self.f_inner = float(f_inner)
        self.f_outer = float(f_outer)

    def call(self, r: Any, contact: Any) -> Any:
        """
        Apply the distance clamp.

        Parameters
        ----------
        r : Array
            Pair distances with shape (...) or (..., 1) in Å.
        contact : Array
            Length scale of each pair in Å, broadcastable against ``r``.

        Returns
        -------
        Array
            Clamped distances r̃ with the same shape as ``r``.
        """
        xp = array_api_compat.array_namespace(r)
        width = self.f_outer - self.f_inner
        u = r / contact
        t = xp.clip((u - self.f_inner) / width, min=0.0, max=1.0)
        t2 = t * t
        # S(t) = 7 t^5 - 14 t^6 + 10 t^7 - 2.5 t^8  (Horner form), the
        # antiderivative of the switch profile with S(0) = 0 and S(1) = 1/2.
        rise = t2 * t2 * t * (7.0 + t * (-14.0 + t * (10.0 - 2.5 * t)))
        interpolated = contact * (
            window_midpoint(self.f_inner, self.f_outer) + width * rise
        )
        # Identity zone: both branches share their first four derivatives at
        # the outer radius, so xp.where preserves the continuity there.
        return xp.where(u >= self.f_outer, r, interpolated)


def parse_basis_type(basis_type: str) -> tuple[str, bool]:
    """
    Split a radial basis type into its family and its ``/fix`` flag.

    Parameters
    ----------
    basis_type : str
        One of ``"bessel"``, ``"gaussian"``, ``"bessel/fix"`` or
        ``"gaussian/fix"`` (case-insensitive).

    Returns
    -------
    tuple[str, bool]
        The basis family (``"bessel"`` or ``"gaussian"``) and whether the
        basis parameters are held fixed during training.

    Raises
    ------
    ValueError
        If the basis type is not one of the four supported values.
    """
    family, _, suffix = str(basis_type).lower().partition("/")
    if family not in ("bessel", "gaussian") or suffix not in ("", "fix"):
        raise ValueError(
            "`basis_type` must be 'bessel', 'gaussian', 'bessel/fix' or "
            f"'gaussian/fix', got '{basis_type}'"
        )
    return family, suffix == "fix"


class RadialBasis(NativeOP):
    """
    Radial basis with an optional C^3 cutoff envelope.

    The trainable radial parameters are stored in ``adam_freqs`` so HybridMuon
    routes them to Adam without weight decay.

    Notes
    -----
    The Bessel basis uses PyTorch's sinc function for numerical stability::

        phi_n(r) = w_n * sinc(w_n * r / π)

    where ``torch.sinc(z) = sin(π*z) / (π*z)``. This is mathematically
    equivalent to the standard form ``sin(w_n * r) / r``, but sinc handles
    the r->0 limit via Taylor expansion, providing continuous gradients
    without explicit epsilon clamping.

    The ``r -> 0`` limit is finite::

        lim_{r->0} w_n * sinc(w_n * r / π) = w_n

    The initial Bessel frequencies follow a common spacing::

        w_n = n * π / rcut, for n = 1..n_radial (in 1/Å)

    A positive ``exponent`` multiplies the C^3 cutoff envelope directly into
    the output. Zero selects the raw basis without constructing an envelope.

    Parameters
    ----------
    rcut : float
        Cutoff radius in Å.
    n_radial : int
        Number of basis functions.
    basis_type : str, optional
        Radial basis type. Supported values are ``"bessel"``, ``"gaussian"``,
        ``"bessel/fix"`` and ``"gaussian/fix"``; the ``/fix`` forms are
        evaluated like their family and differ only in training, where the
        backends keep their frequencies or centres fixed.
    precision : str
        Floating-point precision for the radial basis frequencies and outputs.
    exponent : int, optional
        Exponent for the C^3 cutoff envelope polynomial. Zero disables the
        envelope. Default is 7.
    """

    def __init__(
        self,
        rcut: float,
        basis_type: str = "bessel",
        n_radial: int = 10,
        precision: str = DEFAULT_PRECISION,
        exponent: int = 7,
    ) -> None:
        self.rcut = float(rcut)
        if self.rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        self.n_radial = int(n_radial)
        if self.n_radial <= 0:
            raise ValueError("`n_radial` must be positive")
        self.basis_type = str(basis_type).lower()
        # Parameter promotion in every backend consumes this trainability flag.
        self.basis_family, fixed = parse_basis_type(self.basis_type)
        self.trainable = not fixed
        self.precision = precision
        self.exponent = int(exponent)
        prec = PRECISION_DICT[self.precision.lower()]
        self.pi_tensor = math.pi

        # Frequencies: n*π/rcut, n=1..n_radial
        # Shape: (1, n_radial), stored as a trainable array.
        if self.basis_family == "bessel":
            freqs = np.arange(1, self.n_radial + 1, dtype=prec) * (math.pi / self.rcut)
        else:
            freqs = np.linspace(0.0, self.rcut, self.n_radial, dtype=prec)
        self.adam_freqs = np.reshape(freqs.astype(prec), (1, self.n_radial))
        gaussian_width = self.rcut / max(self.n_radial - 1, 1)
        self.gaussian_coeff = -0.5 / (gaussian_width * gaussian_width)

        self.envelope = (
            C3CutoffEnvelope(
                rcut=self.rcut,
                exponent=self.exponent,
                precision=self.precision,
            )
            if self.exponent != 0
            else None
        )

    def call(self, r: Any) -> Any:
        """
        Compute radial basis functions.

        Parameters
        ----------
        r : Array
            Pair distances with shape (N, 1) in Å, where N is the number of pairs.

        Returns
        -------
        Array
            Radial basis with shape ``(N, n_radial)``. When
            ``exponent > 0``, the output includes the C³ envelope and
            vanishes smoothly at ``rcut``; otherwise it is the raw basis.
        """
        xp = array_api_compat.array_namespace(r)
        freqs = xp_asarray_nodetach(
            xp, self.adam_freqs[...], device=array_api_compat.device(r)
        )
        # === Step 1. Radial basis ===
        # Shape: (N, 1) * (1, n_radial) -> (N, n_radial)
        if self.basis_family == "bessel":
            # phi_n(r) = w_n * sinc(w_n * r / π)
            x = r * freqs  # (N, n_rbf)
            # torch.sinc(z) = sin(π z) / (π z) with sinc(0) = 1. The array API
            # has no sinc, so evaluate it directly with a guarded denominator so
            # the r -> 0 limit and its gradient stay finite.
            z = x / self.pi_tensor
            pz = self.pi_tensor * z
            zero = z == 0.0
            safe_pz = xp.where(zero, xp.ones_like(pz), pz)
            sinc = xp.where(zero, xp.ones_like(pz), xp.sin(safe_pz) / safe_pz)
            raw = freqs * sinc  # (N, n_rbf)
        else:
            dr = r - freqs  # (N, n_rbf)
            raw = xp.exp(dr * dr * self.gaussian_coeff)  # (N, n_rbf)

        # === Step 2. Apply the optional C³ envelope ===
        if self.envelope is not None:
            return raw * self.envelope(r)
        return raw

    def serialize(self) -> dict[str, Any]:
        """Serialize RadialBasis including trainable frequencies."""
        return {
            "@class": "RadialBasis",
            "@version": 2,
            "config": {
                "rcut": self.rcut,
                "basis_type": self.basis_type,
                "n_radial": self.n_radial,
                "exponent": self.exponent,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            },
            "@variables": {"adam_freqs": to_numpy_array(self.adam_freqs)},
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RadialBasis:
        """Deserialize RadialBasis including trainable frequencies."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "RadialBasis":
            raise ValueError(f"Invalid class for RadialBasis: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 2, 1)
        config = data.pop("config", data)
        variables = data.pop("@variables", None)
        precision = str(config["precision"])
        obj = cls(
            rcut=float(config["rcut"]),
            n_radial=int(config["n_radial"]),
            basis_type=str(config.get("basis_type", "bessel")),
            exponent=(
                0
                if version == 1 and config.get("apply_envelope") is False
                else int(config.get("exponent", 7))
            ),
            precision=precision,
        )
        if variables is not None:
            prec = PRECISION_DICT[precision.lower()]
            obj.adam_freqs = np.asarray(variables["adam_freqs"], dtype=prec)
        return obj
