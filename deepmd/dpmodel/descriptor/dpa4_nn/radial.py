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
from deepmd.utils.version import (
    check_version_compatibility,
)

from .norm import (
    RMSNorm,
)


class RadialMLP(NativeOP):
    """
    Radial MLP with channel RMSNorm and configurable activation.

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

    Architecture
    ------------
    Linear → RMSNorm → Activation for all hidden layers,
    with the final layer being a plain Linear (no norm, no activation).

    Notes
    -----
    All bias terms are disabled (Linear bias=False, RMSNorm bias-free) to
    guarantee ``RadialMLP(0) = 0``. This is required because the compile path
    pads masked edges with zero ``edge_rbf``; any non-zero bias would leak
    spurious features into GIE scatter, causing energy divergence between
    compile and non-compile paths.
    """

    def __init__(
        self,
        mlp_layers: list[int],
        *,
        activation_function: str = "silu",
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        if len(mlp_layers) < 2:
            raise ValueError("`mlp_layers` must have at least 2 elements")
        self.mlp_layers = list(mlp_layers)
        self.activation_function = str(activation_function)
        self.precision = precision
        self.trainable = bool(trainable)

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
    The envelope function is defined for scaled distance ``x = r / rcut`` as::

        E(x) = 1 + x^p * (a + b*x + c*x^2 + d*x^3),  for x < 1
        E(x) = 0,                                     for x >= 1

    where the coefficients are chosen to satisfy::

        E(0) = 1,    E(1) = 0
        E'(1) = 0,   E''(1) = 0,   E'''(1) = 0

    This ensures C^3 continuity at the cutoff boundary. The coefficients are::

        a = -(p + 1)(p + 2)(p + 3) / 6
        b = p(p + 2)(p + 3) / 2
        c = -p(p + 1)(p + 3) / 2
        d = p(p + 1)(p + 2) / 6

    For the default exponent p=5, the coefficients are a=-56, b=140, c=-120,
    d=35::

        E(x) = 1 + x^5 * (-56 + 140*x - 120*x^2 + 35*x^3)
             = 1 - 56*x^5 + 140*x^6 - 120*x^7 + 35*x^8

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
    a : float
        Quadratic coefficient for x^p term.
    b : float
        Linear coefficient for x^(p+1) term.
    c : float
        Quadratic coefficient for x^(p+2) term.
    d : float
        Cubic coefficient for x^(p+3) term.
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
        self.coeff_a = -((self.p + 1) * (self.p + 2) * (self.p + 3)) / 6.0
        self.coeff_b = (self.p * (self.p + 2) * (self.p + 3)) / 2.0
        self.coeff_c = -(self.p * (self.p + 1) * (self.p + 3)) / 2.0
        self.coeff_d = (self.p * (self.p + 1) * (self.p + 2)) / 6.0

    def call(self, dst: Any) -> Any:
        """Compute the envelope value for given distances."""
        xp = array_api_compat.array_namespace(dst)
        d_scaled = xp.clip(dst / self.rcut, min=0.0, max=1.0)
        poly = self.coeff_a + d_scaled * (
            self.coeff_b + d_scaled * (self.coeff_c + d_scaled * self.coeff_d)
        )
        env_val = 1 + d_scaled**self.p * poly
        return env_val * xp.astype(d_scaled < 1.0, dst.dtype)


class InnerClamp(NativeOP):
    """
    C3-continuous inner distance clamping for zone bridging.

    Applies a septic Hermite polynomial transition that freezes distances
    below ``r_inner`` to the constant ``r_inner``, then smoothly transitions
    back to identity at ``r_outer``::

        r̃(r) = r_inner                                    if r <= r_inner
        r̃(r) = r_inner + (r_outer - r_inner) * h(t)       if r_inner < r < r_outer
        r̃(r) = r                                          if r >= r_outer

        h(t) = 20t^4 - 45t^5 + 36t^6 - 10t^7,  t = (r - r_inner) / (r_outer - r_inner)

    Boundary conditions:
    ``h(0)=0``, ``h(1)=1``, ``h'(0)=0``, ``h'(1)=1``,
    ``h''(0)=0``, ``h''(1)=0``, ``h'''(0)=0``, ``h'''(1)=0``.
    This ensures C3 continuity: ``dr̃/dr = 0`` at r_inner (frozen zone) and
    ``dr̃/dr = 1`` at r_outer (identity zone), with matched second and third
    derivatives at both boundaries.

    Parameters
    ----------
    r_inner : float
        Freeze radius in Å. Distances below this are clamped to ``r_inner``.
    r_outer : float
        Outer boundary of the transition zone in Å. Above this, ``r̃ = r``.

    Raises
    ------
    ValueError
        If ``r_inner >= r_outer`` or either is non-positive.
    """

    def __init__(self, r_inner: float, r_outer: float) -> None:
        if r_inner <= 0 or r_outer <= 0:
            raise ValueError("r_inner and r_outer must be positive")
        if r_inner >= r_outer:
            raise ValueError(f"r_inner ({r_inner}) must be < r_outer ({r_outer})")
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)

    def call(self, r: Any) -> Any:
        """
        Apply inner distance clamping.

        Parameters
        ----------
        r : Array
            Pair distances with shape (...) or (..., 1) in Å.

        Returns
        -------
        Array
            Clamped distances r̃ with the same shape as input.
        """
        xp = array_api_compat.array_namespace(r)
        t = xp.clip(
            (r - self.r_inner) / (self.r_outer - self.r_inner), min=0.0, max=1.0
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
        interpolated = self.r_inner + (self.r_outer - self.r_inner) * h
        # Identity zone: r >= r_outer returns r directly.
        # Both branches have matching first three derivatives at r_outer,
        # so xp.where preserves C3 continuity here.
        return xp.where(r >= self.r_outer, r, interpolated)


class BridgingSwitch(NativeOP):
    r"""
    C3-continuous switching amplitude for the SeZM bridging zone.

    ``BridgingSwitch`` returns a per-edge scalar amplitude in ``[0, 1]``
    that measures how far an edge sits outside the frozen zone. It is
    the elementary piece the Source Freeze Propagation Gate (SFPG)
    aggregates into a per-node "non-frozen confidence" via a product
    over each source node's outgoing edges::

        w(r) = 0                                             if r <= r_inner  (frozen)
        w(r) = h((r - r_inner) / (r_outer - r_inner))        if r_inner < r < r_outer  (transition)
        w(r) = 1                                             if r >= r_outer  (normal)

        h(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7

    Boundary conditions at ``t=0`` and ``t=1``::

        h(0)   = h'(0)   = h''(0)   = h'''(0)   = 0
        h(1)=1, h'(1)    = h''(1)   = h'''(1)   = 0

    The vanishing first three derivatives at both endpoints give
    ``w \in C^3(\mathbb{R}_{\ge 0})`` with zero slope/curvature at
    ``r_inner`` and ``r_outer``, so forces (first derivatives) and the
    force derivatives consumed by second-order training stay continuous
    across both zone boundaries.

    The surrounding infrastructure (``compute_edge_src_gate``) owns the
    per-node product reduction and broadcast; this module only encodes
    the scalar amplitude shape.

    Parameters
    ----------
    r_inner : float
        Inner radius in Å. At or below this distance ``w = 0``.
    r_outer : float
        Outer radius in Å. At or above this distance ``w = 1``.

    Raises
    ------
    ValueError
        If ``r_inner <= 0``, ``r_outer <= 0``, or ``r_inner >= r_outer``.
    """

    def __init__(self, r_inner: float, r_outer: float) -> None:
        if r_inner <= 0 or r_outer <= 0:
            raise ValueError("r_inner and r_outer must be positive")
        if r_inner >= r_outer:
            raise ValueError(f"r_inner ({r_inner}) must be < r_outer ({r_outer})")
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)

    def call(self, r: Any) -> Any:
        """
        Evaluate the C3 switching amplitude.

        Parameters
        ----------
        r : Array
            Pair distances with shape (...) or (..., 1) in Å.

        Returns
        -------
        Array
            Switching amplitudes in ``[0, 1]`` with the same shape as input.
        """
        xp = array_api_compat.array_namespace(r)
        t = xp.clip(
            (r - self.r_inner) / (self.r_outer - self.r_inner), min=0.0, max=1.0
        )
        t2 = t * t
        t4 = t2 * t2
        # h(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7  (Horner form).
        # Degree-7 smootherstep: the unique polynomial of this degree that
        # hits ``w(r_inner)=0, w(r_outer)=1`` together with C3 flatness at
        # both radii.
        return t4 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))


class RadialBasis(NativeOP):
    """
    Radial basis with C^3 cutoff envelope.

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

    The C^3 cutoff envelope is multiplied directly into the output to ensure
    strict smoothness at ``rcut``.

    Parameters
    ----------
    rcut : float
        Cutoff radius in Å.
    n_radial : int
        Number of basis functions.
    basis_type : str, optional
        Radial basis type. Supported values are ``"bessel"`` and ``"gaussian"``.
    precision : str
        Floating-point precision for the radial basis frequencies and outputs.
    exponent : int, optional
        Exponent for the C^3 cutoff envelope polynomial. Default is 7.
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
        if self.basis_type not in ("bessel", "gaussian"):
            raise ValueError("`basis_type` must be either 'bessel' or 'gaussian'")
        self.precision = precision
        self.exponent = int(exponent)
        prec = PRECISION_DICT[self.precision.lower()]
        self.pi_tensor = math.pi

        # Frequencies: n*π/rcut, n=1..n_radial
        # Shape: (1, n_radial), stored as a trainable array.
        if self.basis_type == "bessel":
            freqs = np.arange(1, self.n_radial + 1, dtype=prec) * (math.pi / self.rcut)
        else:
            freqs = np.linspace(0.0, self.rcut, self.n_radial, dtype=prec)
        self.adam_freqs = np.reshape(freqs.astype(prec), (1, self.n_radial))
        gaussian_width = self.rcut / max(self.n_radial - 1, 1)
        self.gaussian_coeff = -0.5 / (gaussian_width * gaussian_width)

        self.envelope = C3CutoffEnvelope(
            rcut=self.rcut,
            exponent=self.exponent,
            precision=self.precision,
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
            Radial basis multiplied by C^3 cutoff envelope with shape (N, n_rbf).
            The output is smoothly truncated to zero at r = rcut.
        """
        xp = array_api_compat.array_namespace(r)
        freqs = xp_asarray_nodetach(
            xp, self.adam_freqs[...], device=array_api_compat.device(r)
        )
        # === Step 1. Radial basis ===
        # Shape: (N, 1) * (1, n_radial) -> (N, n_radial)
        if self.basis_type == "bessel":
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

        # === Step 2. Apply C^3 envelope for smooth cutoff ===
        envelope = self.envelope(r)  # (N, 1)
        return raw * envelope

    def serialize(self) -> dict[str, Any]:
        """Serialize RadialBasis including trainable frequencies."""
        return {
            "@class": "RadialBasis",
            "@version": 1,
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
        check_version_compatibility(version, 1, 1)
        config = data.pop("config", data)
        variables = data.pop("@variables", None)
        precision = str(config["precision"])
        obj = cls(
            rcut=float(config["rcut"]),
            n_radial=int(config["n_radial"]),
            basis_type=str(config.get("basis_type", "bessel")),
            exponent=int(config.get("exponent", 7)),
            precision=precision,
        )
        if variables is not None:
            prec = PRECISION_DICT[precision.lower()]
            obj.adam_freqs = np.asarray(variables["adam_freqs"], dtype=prec)
        return obj
