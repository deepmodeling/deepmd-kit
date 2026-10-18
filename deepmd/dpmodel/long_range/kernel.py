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

    def _parameter(self, name: str, like: Array) -> Array:
        """A parameter in the namespace, dtype and device of ``like``, autograd kept."""
        xp = array_api_compat.array_namespace(like)
        return xp_asarray_nodetach(
            xp,
            getattr(self, name),
            dtype=like.dtype,
            device=array_api_compat.device(like),
        )

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
        # The softening keeps the derivative finite on a coincident pair.
        r = xp.sqrt(r_sq + 1e-12)
        return xp_erf(coefficients["alpha"] * r) / r


@BaseInteractionKernel.register("sog")
class SOGKernel(BaseInteractionKernel):
    r"""Sum-of-Gaussians kernel :math:`K(r) = \sum_k a_k \exp(-r^2 / 2\sigma_k^2)`.

    Parameters
    ----------
    amplitude
        The amplitudes :math:`a_k`, in Å⁻¹, of either sign: a scalar shared by
        every Gaussian or one value per Gaussian. Required with ``width``;
        with the default widths it defaults to :math:`4\pi \ln b` for every
        Gaussian.
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
                amplitude = 4.0 * np.pi * np.log(width_ratio)
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

    @classmethod
    def _shape_kwargs(cls, variables: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"n_gaussian": int(np.asarray(variables["log_width"]).size)}
