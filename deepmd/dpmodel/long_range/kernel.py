# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Pair interaction kernels of the latent-charge long-range modules.

A kernel is a function :math:`K(r)` of the distance between two charges, in
Å⁻¹, so that charges in units of :math:`e` interact with the energy
:math:`k_e q_i q_j K(r_{ij})` in eV, where :math:`k_e` is the Coulomb
constant. A kernel owns its parameters; widths are stored through their
logarithm, which keeps them positive however training moves them.
"""

from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    Array,
    xp_asarray_nodetach,
    xp_erf,
)
from deepmd.dpmodel.common import (
    PRECISION_DICT,
    NativeOP,
    to_numpy_array,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

COULOMB_CONSTANT = 14.3996454784255
r"""Coulomb constant :math:`e^2 / (4 \pi \varepsilon_0)` in eV Å."""


class BaseInteractionKernel(NativeOP, make_plugin_registry("interaction kernel")):
    """Pair interaction kernel :math:`K(r)` in Å⁻¹ with its own parameters.

    Parameters
    ----------
    trainable
        Whether the parameters of the kernel are trained.
    precision
        The precision in which the parameters are stored; the kernel is
        evaluated in the precision of its input.
    """

    PARAMETER_NAMES: tuple[str, ...] = ()
    """Attributes holding the trainable parameters of the kernel."""

    def __init__(self, trainable: bool = True, precision: str = "default") -> None:
        self.trainable = bool(trainable)
        self.precision = precision
        self.prec = PRECISION_DICT[precision.lower()]

    def call(self, r_sq: Array) -> Array:
        """Evaluate the kernel on squared pair distances.

        Parameters
        ----------
        r_sq
            Squared distances with shape ``(P,)``, in Å².

        Returns
        -------
        Array
            The kernel values with shape ``(P,)``, in Å⁻¹.
        """
        return self.evaluate(r_sq, self.coefficients(r_sq))

    def coefficients(self, like: Array) -> dict[str, Array]:
        """The coefficients of the kernel formula, derived from its parameters.

        Parameters
        ----------
        like
            An array giving the namespace, dtype and device of the coefficients.

        Returns
        -------
        dict[str, Array]
            Every coefficient with its own shape, differentiable with respect
            to the parameters.
        """
        raise NotImplementedError

    def evaluate(self, r_sq: Array, coefficients: dict[str, Array]) -> Array:
        """Evaluate the kernel formula with given coefficients.

        Parameters
        ----------
        r_sq
            Squared distances with shape ``(P,)``, in Å².
        coefficients
            The coefficients of :meth:`coefficients`, each with its own shape
            or with a leading pair axis of length ``P``.

        Returns
        -------
        Array
            The kernel values with shape ``(P,)``, in Å⁻¹.
        """
        raise NotImplementedError

    def kspace(self, k_sq: Array) -> Array:
        r"""Fourier transform of the kernel on squared wave numbers.

        :math:`\hat K(k) = \int K(r) e^{-i \mathbf{k} \cdot \mathbf{r}} d^3r`,
        so that the energy of charges in a periodic cell of volume ``V`` reads
        :math:`(k_e / 2V) \sum_{\mathbf{k} \neq 0} \hat K(k) |\hat\rho(\mathbf{k})|^2`.

        Parameters
        ----------
        k_sq
            Squared wave numbers with shape ``(nk,)`` or ``(rows, nk)``, in
            Å⁻²; every entry must be positive.

        Returns
        -------
        Array
            The transformed kernel with the shape of ``k_sq``, in Å³.
        """
        raise NotImplementedError

    def contact_value(self, like: Array) -> Array:
        r"""The kernel at contact, :math:`K(0)`, in Å⁻¹.

        The k-space lattice sum of a periodic system contains the self terms,
        which the evaluation subtracts as :math:`\frac{1}{2} K(0) \sum_i q_i^2`.

        Parameters
        ----------
        like
            An array giving the namespace, dtype and device of the result.
        """
        raise NotImplementedError

    def kspace_cutoff(self, tolerance: float = 1e-8) -> float:
        """Wave number at which the kernel weight falls below ``tolerance``.

        The Gaussian tail of the Fourier-transformed kernel makes the k-space
        truncation exponentially convergent; the k grid of a periodic
        evaluation ends at this wave number.

        Parameters
        ----------
        tolerance
            The relative weight of the slowest-decaying mode at the cutoff.
        """
        raise NotImplementedError

    def _parameter(self, name: str, like: Array) -> Array:
        """A parameter in the namespace, dtype and device of ``like``, autograd kept."""
        xp = array_api_compat.array_namespace(like)
        return xp_asarray_nodetach(
            xp,
            getattr(self, name),
            dtype=like.dtype,
            device=array_api_compat.device(like),
        )

    @staticmethod
    def _row_values(value: Array, k_sq: Array) -> Array:
        """A scalar broadcast to shape ``(..., 1)`` over the leading axes of ``k_sq``.

        The gradient of a parameter of the transform sums over every wave
        number. Broadcast over the rows, it sums along every row first, a
        reduction a compiled graph spreads over the rows, and then over the
        rows, rather than over every wave number in one block.
        """
        xp = array_api_compat.array_namespace(k_sq)
        return xp.broadcast_to(value, (*k_sq.shape[:-1], 1))

    def serialize(self) -> dict:
        return {
            "@class": "InteractionKernel",
            "type": self.TYPE,
            "@version": 1,
            "trainable": self.trainable,
            "precision": self.precision,
            "@variables": {
                name: to_numpy_array(getattr(self, name))
                for name in self.PARAMETER_NAMES
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "BaseInteractionKernel":
        if cls is BaseInteractionKernel:
            return cls.get_class_by_type(data["type"]).deserialize(data)
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        data.pop("type")
        variables = data.pop("@variables")
        obj = cls(**data, **cls._shape_kwargs(variables))
        for name in cls.PARAMETER_NAMES:
            setattr(obj, name, np.asarray(variables[name], dtype=obj.prec))
        return obj

    @classmethod
    def _shape_kwargs(cls, variables: dict[str, np.ndarray]) -> dict[str, Any]:
        """Constructor arguments that size the parameters stored in ``variables``."""
        return {}


# Taylor coefficients of erf(x) / x in powers of x^2, 2 / sqrt(pi) (-1)^n /
# (n! (2n + 1)), and the square of x below which LESKernel evaluates them: there
# the nine terms reach the float64 rounding of erf(x) / x. Above it the closed
# form keeps the relative error of its radial derivative within 2e-14 in
# float64 and 1e-5 in float32.
_ERF_OVER_X_SERIES = (
    1.1283791670955126,
    -0.37612638903183754,
    0.11283791670955126,
    -0.026866170645131252,
    0.005223977625442188,
    -0.0008548327023450852,
    0.00012055332981789664,
    -1.492565035840625e-05,
    1.6462114365889246e-06,
)
_ERF_SERIES_X_SQ = 0.0625


@BaseInteractionKernel.register("les")
class LESKernel(BaseInteractionKernel):
    r"""Latent Ewald summation kernel :math:`K(r) = \mathrm{erf}(\alpha r) / r`.

    The kernel is the interaction of two Gaussian charge clouds of width
    :math:`1/(\sqrt{2}\alpha)`: it reaches :math:`1/r` beyond a few
    :math:`1/\alpha` and stays finite, :math:`2\alpha/\sqrt{\pi}`, at contact.

    Parameters
    ----------
    alpha
        The inverse width :math:`\alpha > 0`, in Å⁻¹.
    trainable
        Whether :math:`\alpha` is trained.
    precision
        The precision in which :math:`\log \alpha` is stored.
    """

    TYPE = "les"
    PARAMETER_NAMES = ("log_alpha",)

    def __init__(
        self, alpha: float = 1.0, trainable: bool = True, precision: str = "default"
    ) -> None:
        super().__init__(trainable, precision)
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"`alpha` must be finite and positive, got {alpha}")
        self.log_alpha = np.array(np.log(alpha), dtype=self.prec)

    def coefficients(self, like: Array) -> dict[str, Array]:
        xp = array_api_compat.array_namespace(like)
        return {"alpha": xp.exp(self._parameter("log_alpha", like))}

    def evaluate(self, r_sq: Array, coefficients: dict[str, Array]) -> Array:
        xp = array_api_compat.array_namespace(r_sq)
        alpha = coefficients["alpha"]
        x_sq = alpha**2 * r_sq
        # K = alpha erf(x) / x with x = alpha r. The derivative of the closed
        # form subtracts two terms that agree to second order in x, so near
        # contact erf(x) / x comes from its Taylor series in x^2. Both are
        # evaluated on every pair, which keeps the shapes static; each reads a
        # value of its own domain where the other is selected, which keeps the
        # derivatives of both finite.
        near = x_sq < _ERF_SERIES_X_SQ
        x_far = xp.sqrt(xp.where(near, xp.ones_like(x_sq), x_sq))
        x_sq_near = xp.where(near, x_sq, xp.zeros_like(x_sq))
        series = xp.full_like(x_sq, _ERF_OVER_X_SERIES[-1])
        for coefficient in reversed(_ERF_OVER_X_SERIES[:-1]):
            series = series * x_sq_near + coefficient
        return alpha * xp.where(near, series, xp_erf(x_far) / x_far)

    def kspace(self, k_sq: Array) -> Array:
        xp = array_api_compat.array_namespace(k_sq)
        # erf(alpha r)/r is the interaction of two Gaussian charge clouds of
        # width 1/(sqrt(2) alpha); its transform is the damped Coulomb
        # 4 pi exp(-k^2 / 4 alpha^2) / k^2.
        alpha = self._row_values(xp.exp(self._parameter("log_alpha", k_sq)), k_sq)
        return 4.0 * np.pi * xp.exp(-k_sq / (4.0 * alpha * alpha)) / k_sq

    def contact_value(self, like: Array) -> Array:
        xp = array_api_compat.array_namespace(like)
        alpha = xp.exp(self._parameter("log_alpha", like))
        return 2.0 * alpha / np.sqrt(np.pi)

    def kspace_cutoff(self, tolerance: float = 1e-8) -> float:
        # exp(-k^2 / 4 alpha^2) < tolerance
        alpha = float(np.exp(to_numpy_array(self.log_alpha).astype(np.float64)))
        return 2.0 * alpha * np.sqrt(np.log(1.0 / tolerance))


@BaseInteractionKernel.register("sog")
class SOGKernel(BaseInteractionKernel):
    r"""Sum-of-Gaussians kernel :math:`K(r) = \sum_k a_k \exp(-r^2 / 2\sigma_k^2)`.

    Parameters
    ----------
    amplitude
        The amplitudes :math:`a_k`, in Å⁻¹, of either sign: a scalar shared by
        every Gaussian or one value per Gaussian. Required with ``width``;
        with the default widths it defaults to
        :math:`a_k = \sqrt{2/\pi} \ln b / \sigma_k`, the trapezoidal
        discretization in :math:`\ln s` of
        :math:`1/r = \sqrt{2/\pi} \int_0^\infty e^{-r^2/2s^2} s^{-2} \, ds`.
        The default kernel then follows :math:`1/r`, less a nearly constant
        offset, from a few :math:`\sigma` to a fraction of
        :math:`\sigma b^{M-1}`, and stays finite at contact. The offset
        exerts no force between fixed charges; through charges that respond
        to the positions it does.
    width
        The widths :math:`\sigma_k > 0`, in Å. Defaults to the geometric
        sequence :math:`\sigma_k = \sigma b^k`, :math:`k = 0, \dots, M - 1`.
    n_gaussian
        The number :math:`M` of Gaussians of the default widths.
    base_width
        The width :math:`\sigma` of the first default Gaussian, in Å.
    width_ratio
        The ratio :math:`b > 1` of consecutive default widths.
    trainable
        Whether the amplitudes and widths are trained.
    precision
        The precision in which the amplitudes and log-widths are stored.
    """

    TYPE = "sog"
    PARAMETER_NAMES = ("amplitude", "log_width")

    def __init__(
        self,
        amplitude: float | list[float] | None = None,
        width: list[float] | None = None,
        n_gaussian: int = 12,
        base_width: float = 2.180230445405648,
        width_ratio: float = 1.62976708826776469,
        trainable: bool = True,
        precision: str = "default",
    ) -> None:
        super().__init__(trainable, precision)
        if width is None:
            if n_gaussian < 1:
                raise ValueError(f"`n_gaussian` must be positive, got {n_gaussian}")
            if not np.isfinite(base_width) or base_width <= 0.0:
                raise ValueError(f"`base_width` must be positive, got {base_width}")
            if not np.isfinite(width_ratio) or width_ratio <= 1.0:
                raise ValueError(
                    f"`width_ratio` must be greater than one, got {width_ratio}"
                )
            width = base_width * np.power(
                width_ratio, np.arange(n_gaussian, dtype=np.float64)
            )
            if amplitude is None:
                amplitude = np.sqrt(2.0 / np.pi) * np.log(width_ratio) / width
        elif amplitude is None:
            raise ValueError("`amplitude` is required with explicit `width`")
        width = np.asarray(width, dtype=np.float64).reshape(-1)
        if width.size < 1 or not np.all(np.isfinite(width)) or np.any(width <= 0.0):
            raise ValueError("`width` must hold finite positive values")
        amplitude = np.asarray(amplitude, dtype=np.float64).reshape(-1)
        if amplitude.size == 1:
            amplitude = np.full_like(width, amplitude[0])
        if amplitude.shape != width.shape or not np.all(np.isfinite(amplitude)):
            raise ValueError(
                "`amplitude` must be a finite scalar or hold one value per Gaussian"
            )
        self.amplitude = amplitude.astype(self.prec)
        self.log_width = np.log(width).astype(self.prec)

    def coefficients(self, like: Array) -> dict[str, Array]:
        xp = array_api_compat.array_namespace(like)
        return {
            "amplitude": self._parameter("amplitude", like),
            "inv_two_width_sq": 0.5 * xp.exp(-2.0 * self._parameter("log_width", like)),
        }

    def evaluate(self, r_sq: Array, coefficients: dict[str, Array]) -> Array:
        xp = array_api_compat.array_namespace(r_sq)
        gaussian = xp.exp(-r_sq[:, None] * coefficients["inv_two_width_sq"])
        return xp.sum(coefficients["amplitude"] * gaussian, axis=-1)

    def kspace(self, k_sq: Array) -> Array:
        xp = array_api_compat.array_namespace(k_sq)
        amplitude = self._parameter("amplitude", k_sq)
        width = xp.exp(self._parameter("log_width", k_sq))
        # FT of a_k exp(-r^2 / 2 sigma_k^2) is a_k (2 pi)^{3/2} sigma_k^3
        # exp(-sigma_k^2 k^2 / 2); for the default amplitudes
        # a_k = sqrt(2/pi) ln b / sigma_k the coefficient is 4 pi ln b sigma_k^2,
        # whose ln-sigma trapezoid sum approaches 4 pi / k^2. The mode loop is
        # unrolled: a (nk, M) intermediate breaks inductor's double backward of
        # the k-space virial (NaN), while per-mode (nk,) terms compile fine.
        terms = []
        for m in range(width.shape[0]):
            weight = self._row_values(amplitude[m] * width[m] ** 3, k_sq)
            rate = self._row_values(0.5 * width[m] ** 2, k_sq)
            terms.append((2.0 * np.pi) ** 1.5 * weight * xp.exp(-rate * k_sq))
        return sum(terms[1:], terms[0])

    def contact_value(self, like: Array) -> Array:
        xp = array_api_compat.array_namespace(like)
        amplitude = self._parameter("amplitude", like)
        return xp.sum(amplitude)

    def kspace_cutoff(self, tolerance: float = 1e-8) -> float:
        # the slowest-decaying mode is the NARROWEST Gaussian:
        # exp(-sigma_min^2 k^2 / 2) < tolerance
        sigma_min = float(
            np.exp(np.min(to_numpy_array(self.log_width).astype(np.float64)))
        )
        return np.sqrt(2.0 * np.log(1.0 / tolerance)) / sigma_min

    @classmethod
    def _shape_kwargs(cls, variables: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"n_gaussian": int(np.asarray(variables["log_width"]).size)}
