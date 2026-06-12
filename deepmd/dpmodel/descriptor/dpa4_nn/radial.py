# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Radial building blocks for the DPA4/SeZM descriptor.

This module is the dpmodel port of ``deepmd.pt.model.descriptor.sezm_nn.radial``.
It defines the cutoff envelope, radial basis, and radial multilayer perceptron
used by the DPA4 descriptor. ``InnerClamp`` and ``BridgingSwitch`` are ported
by later tasks together with the modules that consume them.

Serialization contract: the ``@variables`` keys of each class match the
``state_dict`` key names of its pt counterpart, so pt ``serialize()`` output
deserializes directly into the dpmodel classes (and vice versa).
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
    seed : int | list[int] | None
        Random seed for the layer initialization.

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
        self.mlp_layers = [int(d) for d in mlp_layers]
        self.activation_function = str(activation_function)
        self.precision = precision
        self.trainable = bool(trainable)

        n_layers = len(self.mlp_layers)
        self.layers: list[NativeLayer] = []
        self.norms: list[RMSNorm] = []
        for i in range(n_layers - 1):
            self.layers.append(
                NativeLayer(
                    self.mlp_layers[i],
                    self.mlp_layers[i + 1],
                    bias=False,
                    activation_function=None,
                    precision=self.precision,
                    seed=child_seed(seed, i),
                    trainable=self.trainable,
                )
            )
            # Last layer: no RMSNorm/activation
            if i < n_layers - 2:
                self.norms.append(
                    RMSNorm(
                        channels=self.mlp_layers[i + 1],
                        precision=self.precision,
                        trainable=self.trainable,
                    )
                )

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
        n_hidden = len(self.norms)
        for i, layer in enumerate(self.layers):
            x = layer.call(x)
            if i < n_hidden:
                x = self.norms[i].call(x)
                fn = get_activation_fn(self.activation_function)
                x = fn(x)
        return x

    def serialize(self) -> dict[str, Any]:
        """Serialize the RadialMLP to a dict.

        The ``@variables`` keys follow the pt ``net.state_dict()`` naming:
        ``{3*i}.matrix`` for the i-th linear layer and ``{3*i+1}.adam_scale``
        for the i-th RMSNorm (activation modules are parameter-free).
        """
        variables: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            variables[f"{3 * i}.matrix"] = to_numpy_array(layer.w)
            if i < len(self.norms):
                variables[f"{3 * i + 1}.adam_scale"] = to_numpy_array(
                    self.norms[i].adam_scale
                )
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
        precision = str(data.pop("dtype"))
        obj = cls(
            data.pop("mlp_layers"),
            activation_function=str(data.pop("activation_function")),
            precision=precision,
            trainable=bool(data.pop("trainable")),
        )
        prec = PRECISION_DICT[precision.lower()]
        expected_keys = {f"{3 * i}.matrix" for i in range(len(obj.layers))} | {
            f"{3 * i + 1}.adam_scale" for i in range(len(obj.norms))
        }
        if set(variables) != expected_keys:
            raise ValueError(
                f"variable keys {sorted(variables)} do not match the expected "
                f"keys {sorted(expected_keys)}"
            )
        for key, value in variables.items():
            idx_s, _, name = key.partition(".")
            idx = int(idx_s)
            value = np.asarray(value, dtype=prec)
            if name == "matrix":
                layer = obj.layers[idx // 3]
                if value.shape != layer.w.shape:
                    raise ValueError(
                        f"shape of {key} {value.shape} does not match "
                        f"the layer shape {layer.w.shape}"
                    )
                layer.w = value
            else:
                norm = obj.norms[idx // 3]
                norm.adam_scale = value.reshape(norm.adam_scale.shape)
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
    precision : str
        Floating point precision label (kept for config parity with pt; the
        computation follows the input dtype).
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

    def serialize(self) -> dict[str, Any]:
        """Serialize the C3CutoffEnvelope to a dict (config only, no state)."""
        return {
            "@class": "C3CutoffEnvelope",
            "@version": 1,
            "config": {
                "rcut": self.rcut,
                "exponent": self.p,
                "precision": np.dtype(PRECISION_DICT[self.precision]).name,
            },
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> C3CutoffEnvelope:
        """Deserialize a C3CutoffEnvelope from a dict."""
        data = data.copy()
        data_cls = data.pop("@class")
        if data_cls != "C3CutoffEnvelope":
            raise ValueError(f"Invalid class for C3CutoffEnvelope: {data_cls}")
        version = int(data.pop("@version"))
        check_version_compatibility(version, 1, 1)
        config = data.pop("config")
        return cls(
            rcut=float(config["rcut"]),
            exponent=int(config["exponent"]),
            precision=str(config["precision"]),
        )


class RadialBasis(NativeOP):
    """
    Radial basis with C^3 cutoff envelope.

    The trainable radial parameters are stored in ``adam_freqs`` so HybridMuon
    routes them to Adam without weight decay.

    Notes
    -----
    The Bessel basis uses the normalized sinc function for numerical
    stability::

        phi_n(r) = w_n * sinc(w_n * r / π)

    where ``sinc(z) = sin(π*z) / (π*z)`` with ``sinc(0) = 1`` (same convention
    as ``torch.sinc`` and ``np.sinc``). This is mathematically equivalent to
    the standard form ``sin(w_n * r) / r``, but sinc handles the r->0 limit,
    providing continuous gradients without explicit epsilon clamping.

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
    basis_type : str, optional
        Radial basis type. Supported values are ``"bessel"`` and ``"gaussian"``.
    n_radial : int
        Number of basis functions.
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
            Pair distances with shape (N, 1) in Å, where N is the number of
            pairs.

        Returns
        -------
        Array
            Radial basis multiplied by C^3 cutoff envelope with shape
            (N, n_rbf). The output is smoothly truncated to zero at r = rcut.
        """
        xp = array_api_compat.array_namespace(r)
        freqs = xp.asarray(self.adam_freqs, device=array_api_compat.device(r))
        # === Step 1. Radial basis ===
        # Shape: (N, 1) * (1, n_radial) -> (N, n_radial)
        if self.basis_type == "bessel":
            # phi_n(r) = w_n * sinc(w_n * r / π)
            x = r * freqs  # (N, n_rbf)
            # normalized sinc, mirroring torch.sinc(x / π):
            # sinc(z) = sin(π*z) / (π*z), with sinc(0) = 1.
            # The zero branch is selected through a safe denominator so that
            # gradients stay finite at r = 0.
            z = x / math.pi
            pz = math.pi * z
            zero = z == 0.0
            safe_pz = xp.where(zero, xp.ones_like(pz), pz)
            sinc = xp.where(zero, xp.ones_like(pz), xp.sin(safe_pz) / safe_pz)
            raw = freqs * sinc  # (N, n_rbf)
        else:
            dr = r - freqs  # (N, n_rbf)
            raw = xp.exp(dr * dr * self.gaussian_coeff)  # (N, n_rbf)

        # === Step 2. Apply C^3 envelope for smooth cutoff ===
        envelope = self.envelope.call(r)  # (N, 1)
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
            basis_type=str(config.get("basis_type", "bessel")),
            n_radial=int(config["n_radial"]),
            precision=precision,
            exponent=int(config.get("exponent", 7)),
        )
        if variables is not None:
            prec = PRECISION_DICT[precision.lower()]
            adam_freqs = np.asarray(variables["adam_freqs"], dtype=prec)
            if adam_freqs.shape != obj.adam_freqs.shape:
                raise ValueError(
                    f"adam_freqs shape {adam_freqs.shape} does not match "
                    f"the expected shape {obj.adam_freqs.shape}"
                )
            obj.adam_freqs = adam_freqs
        return obj
