# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Radial building blocks for the SeZM descriptor.

This module defines the cutoff envelope, inner-distance clamp, radial basis,
and radial multilayer perceptron used by SeZM.
"""

from __future__ import (
    annotations,
)

import math
from typing import (
    Any,
)

import torch
import torch.nn as nn
from einops import (
    rearrange,
)

from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.utils import (
    ActivationFn,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .norm import (
    RMSNorm,
)
from .utils import (
    np_safe,
    safe_numpy_to_tensor,
)


class RadialMLP(nn.Module):
    """
    Radial MLP with optional channel RMSNorm and configurable activation.

    Parameters
    ----------
    mlp_layers : list[int]
        Layer sizes including input and output dimensions.
        E.g., [in_dim, hidden1, hidden2, out_dim].
    activation_function : str
        Activation function name (e.g., "silu", "tanh", "gelu").
    dtype : torch.dtype
        Floating point dtype for the linear layers.
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
    The input ``edge_rbf`` carries the C^3 cutoff envelope and therefore
    vanishes at ``rcut``; the RMSNorm divides that envelope out, and its ``eps``
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
        dtype: torch.dtype = torch.float32,
        trainable: bool = True,
        radial_norm: bool = True,
        seed: int | list[int] | None = None,
    ) -> None:
        super().__init__()
        if len(mlp_layers) < 2:
            raise ValueError("`mlp_layers` must have at least 2 elements")
        self.mlp_layers = list(mlp_layers)
        self.activation_function = str(activation_function)
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[self.dtype]
        self.trainable = bool(trainable)
        self.radial_norm = bool(radial_norm)

        modules: list[nn.Module] = []
        n_layers = len(mlp_layers)
        for i in range(n_layers - 1):
            linear = MLPLayer(
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
                            dtype=self.dtype,
                            trainable=trainable,
                        )
                    )
                modules.append(ActivationFn(self.activation_function))

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (..., mlp_layers[0]).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (..., mlp_layers[-1]).
        """
        return self.net(x)

    def serialize(self) -> dict[str, Any]:
        """Serialize the RadialMLP to a dict."""
        state = self.net.state_dict()
        return {
            "@class": "RadialMLP",
            "@version": 1,
            "mlp_layers": self.mlp_layers.copy(),
            "activation_function": self.activation_function,
            "dtype": RESERVED_PRECISION_DICT[self.dtype],
            "trainable": self.trainable,
            "radial_norm": self.radial_norm,
            "@variables": {k: np_safe(v) for k, v in state.items()},
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
        data["dtype"] = PRECISION_DICT[data["dtype"]]
        obj = cls(**data)
        state = {
            k: safe_numpy_to_tensor(v, device=env.DEVICE, dtype=obj.dtype)
            for k, v in variables.items()
        }
        obj.net.load_state_dict(state)
        return obj


class C3CutoffEnvelope(torch.nn.Module):
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
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        if exponent <= 0:
            raise ValueError("`exponent` must be positive")
        self.rcut = float(rcut)
        self.p = int(exponent)
        self.dtype = dtype
        self.device = env.DEVICE
        self._series_coefficients = tuple(
            float(math.comb(k + 3, 3)) for k in range(self.p)
        )
        self.register_buffer(
            "rcut_tensor",
            torch.tensor(self.rcut, dtype=self.dtype, device=self.device),
            persistent=False,
        )

    def forward(self, dst: torch.Tensor) -> torch.Tensor:
        """Compute the envelope value for given distances."""
        u = ((self.rcut_tensor - dst) / self.rcut_tensor).clamp(min=0.0, max=1.0)
        x = 1.0 - u
        series = torch.full_like(x, self._series_coefficients[-1])
        for coefficient in reversed(self._series_coefficients[:-1]):
            series = coefficient + x * series
        return u.pow(4) * series


class InnerClamp(nn.Module):
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
        super().__init__()
        if r_inner <= 0 or r_outer <= 0:
            raise ValueError("r_inner and r_outer must be positive")
        if r_inner >= r_outer:
            raise ValueError(f"r_inner ({r_inner}) must be < r_outer ({r_outer})")
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply inner distance clamping.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (...) or (..., 1) in Å.

        Returns
        -------
        torch.Tensor
            Clamped distances r̃ with the same shape as input.
        """
        t = ((r - self.r_inner) / (self.r_outer - self.r_inner)).clamp(0.0, 1.0)
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
        # so torch.where preserves C3 continuity here.
        return torch.where(r >= self.r_outer, r, interpolated)


class BridgingSwitch(nn.Module):
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
        super().__init__()
        if r_inner <= 0 or r_outer <= 0:
            raise ValueError("r_inner and r_outer must be positive")
        if r_inner >= r_outer:
            raise ValueError(f"r_inner ({r_inner}) must be < r_outer ({r_outer})")
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the C3 switching amplitude.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (...) or (..., 1) in Å.

        Returns
        -------
        torch.Tensor
            Switching amplitudes in ``[0, 1]`` with the same shape as input.
        """
        t = ((r - self.r_inner) / (self.r_outer - self.r_inner)).clamp(0.0, 1.0)
        t2 = t * t
        t4 = t2 * t2
        # h(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7  (Horner form).
        # Degree-7 smootherstep: the unique polynomial of this degree that
        # hits ``w(r_inner)=0, w(r_outer)=1`` together with C3 flatness at
        # both radii.
        return t4 * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)))


class RadialBasis(nn.Module):
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
    dtype : torch.dtype
        Floating-point dtype for the radial basis frequencies and outputs.
    exponent : int, optional
        Exponent for the C^3 cutoff envelope polynomial. Default is 7.
    """

    def __init__(
        self,
        rcut: float,
        basis_type: str = "bessel",
        n_radial: int = 10,
        dtype: torch.dtype = torch.float32,
        exponent: int = 7,
    ) -> None:
        super().__init__()
        self.rcut = float(rcut)
        if self.rcut <= 0.0:
            raise ValueError("`rcut` must be positive")
        self.n_radial = int(n_radial)
        if self.n_radial <= 0:
            raise ValueError("`n_radial` must be positive")
        self.basis_type = str(basis_type).lower()
        if self.basis_type not in ("bessel", "gaussian"):
            raise ValueError("`basis_type` must be either 'bessel' or 'gaussian'")
        self.dtype = dtype
        self.device = env.DEVICE
        self.precision = RESERVED_PRECISION_DICT[self.dtype]
        self.exponent = int(exponent)
        self.register_buffer(
            "pi_tensor",
            torch.tensor(math.pi, dtype=self.dtype, device=self.device),
            persistent=False,
        )

        # Frequencies: n*π/rcut, n=1..n_radial
        # Shape: (1, n_radial), stored as trainable nn.Parameter.
        if self.basis_type == "bessel":
            freqs = torch.arange(
                1,
                self.n_radial + 1,
                device=self.device,
                dtype=self.dtype,
            ) * (math.pi / self.rcut)
        else:
            freqs = torch.linspace(
                0.0,
                self.rcut,
                self.n_radial,
                device=self.device,
                dtype=self.dtype,
            )
        self.adam_freqs = nn.Parameter(
            rearrange(freqs, "n_radial -> 1 n_radial"), requires_grad=True
        )
        gaussian_width = self.rcut / max(self.n_radial - 1, 1)
        self.register_buffer(
            "gaussian_coeff",
            torch.tensor(
                -0.5 / (gaussian_width * gaussian_width),
                dtype=self.dtype,
                device=self.device,
            ),
            persistent=False,
        )

        self.envelope = C3CutoffEnvelope(
            rcut=self.rcut,
            exponent=self.exponent,
            dtype=self.dtype,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute radial basis functions.

        Parameters
        ----------
        r : torch.Tensor
            Pair distances with shape (N, 1) in Å, where N is the number of pairs.

        Returns
        -------
        torch.Tensor
            Radial basis multiplied by C^3 cutoff envelope with shape (N, n_rbf).
            The output is smoothly truncated to zero at r = rcut.
        """
        # === Step 1. Radial basis ===
        # Shape: (N, 1) * (1, n_radial) -> (N, n_radial)
        if self.basis_type == "bessel":
            # phi_n(r) = w_n * sinc(w_n * r / π)
            x = r * self.adam_freqs  # (N, n_rbf)
            raw = self.adam_freqs * torch.sinc(x / self.pi_tensor)  # (N, n_rbf)
        else:
            dr = r - self.adam_freqs  # (N, n_rbf)
            raw = torch.exp(dr * dr * self.gaussian_coeff)  # (N, n_rbf)

        # === Step 2. Apply C^3 envelope for smooth cutoff ===
        envelope = self.envelope(r)  # (N, 1)
        return raw * envelope

    def serialize(self) -> dict[str, Any]:
        """Serialize RadialBasis including trainable frequencies."""
        state = self.state_dict()
        return {
            "@class": "RadialBasis",
            "@version": 1,
            "config": {
                "rcut": self.rcut,
                "basis_type": self.basis_type,
                "n_radial": self.n_radial,
                "exponent": self.exponent,
                "precision": RESERVED_PRECISION_DICT[self.dtype],
            },
            "@variables": {key: np_safe(value) for key, value in state.items()},
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
        precision = config["precision"]
        dtype = PRECISION_DICT[precision]
        obj = cls(
            rcut=float(config["rcut"]),
            n_radial=int(config["n_radial"]),
            basis_type=str(config.get("basis_type", "bessel")),
            exponent=int(config.get("exponent", 7)),
            dtype=dtype,
        )
        if variables is not None:
            template = obj.state_dict()
            state = {
                key: safe_numpy_to_tensor(
                    value, device=template[key].device, dtype=template[key].dtype
                )
                for key, value in variables.items()
            }
            obj.load_state_dict(state)
        return obj
