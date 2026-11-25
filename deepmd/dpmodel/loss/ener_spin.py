# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.reduction import (
    masked_atom_mean,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class EnergySpinLoss(EnergyLoss):
    r"""Loss on energy, real force, magnetic force and virial for spin models.

    For mean-squared error, the objective is a weighted sum

    .. math::

        L = p_E\left\langle\frac{(\Delta E)^2}{N^q}\right\rangle
        +p_{F_r}\left\langle(\Delta F^r_{i\alpha})^2
        \right\rangle_{i,\alpha}
        +p_{F_m}\left\langle(\Delta F^m_{i\alpha})^2
        \right\rangle_{i\in\mathcal M,\alpha}
        +p_\Xi\left\langle\frac{(\Delta\Xi_{\alpha\beta})^2}{N^q}
        \right\rangle_{\alpha,\beta}
        +p_{E_i}\langle(\Delta E_i)^2\rangle,

    where :math:`q=2` for intensive energy/virial normalization and :math:`q=1`
    for the legacy normalization, and :math:`\mathcal M` is the set selected by
    ``mask_mag``.  Thus force and virial errors are componentwise means rather
    than means of squared vector or tensor norms.  In MAE mode the squared
    component differences are replaced by absolute differences and extensive
    terms use :math:`1/N`.  Every prefactor is interpolated using the current
    learning rate,

    .. math::
        p_X(\eta)=p_X^{\mathrm{limit}}+
        \left(p_X^{\mathrm{start}}-p_X^{\mathrm{limit}}\right)
        \frac{\eta}{\eta_0}.

    Parameters
    ----------
    starter_learning_rate : float
        The learning rate at the start of the training.
    start_pref_e : float
        The prefactor of energy loss at the start of the training.
    limit_pref_e : float
        The prefactor of energy loss at the end of the training.
    start_pref_fr : float
        The prefactor of real force loss at the start of the training.
    limit_pref_fr : float
        The prefactor of real force loss at the end of the training.
    start_pref_fm : float
        The prefactor of magnetic force loss at the start of the training.
    limit_pref_fm : float
        The prefactor of magnetic force loss at the end of the training.
    start_pref_v : float
        The prefactor of virial loss at the start of the training.
    limit_pref_v : float
        The prefactor of virial loss at the end of the training.
    start_pref_ae : float
        The prefactor of atomic energy loss at the start of the training.
    limit_pref_ae : float
        The prefactor of atomic energy loss at the end of the training.
    enable_atom_ener_coeff : bool
        if true, the energy will be computed as \sum_i c_i E_i
    loss_func : str
        Loss function type: 'mse' or 'mae'.
    intensive_ener_virial : bool
        If true, the MSE energy and virial terms use intensive normalization,
        i.e. an additional normalization by the square of the number of atoms
        (1/N^2) instead of the legacy (1/N) behavior. This keeps those MSE loss
        terms consistent with per-atom RMSE reporting and less dependent on
        system size. This option does not change the MAE formulation, which is
        handled separately. The default is false for backward compatibility with
        models trained using deepmd-kit <= 3.1.3.
    symm_virial : bool
        Compare only the symmetric part of predicted and labeled virials.
    **kwargs
        Other keyword arguments.
    """

    force_metric = "fr"
    metric_order = (
        "rmse_e",
        "mae_e",
        "mae_e_all",
        "rmse_fr",
        "mae_fr",
        "rmse_fm",
        "mae_fm",
        "rmse_ae",
        "mae_ae",
        "rmse_v",
        "mae_v",
        "rmse",
    )

    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        start_pref_e: float = 0.0,
        limit_pref_e: float = 0.0,
        start_pref_fr: float = 0.0,
        limit_pref_fr: float = 0.0,
        start_pref_fm: float = 0.0,
        limit_pref_fm: float = 0.0,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        enable_atom_ener_coeff: bool = False,
        loss_func: str = "mse",
        intensive_ener_virial: bool = False,
        symm_virial: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            starter_learning_rate=starter_learning_rate,
            start_pref_e=start_pref_e,
            limit_pref_e=limit_pref_e,
            start_pref_f=start_pref_fr,
            limit_pref_f=limit_pref_fr,
            start_pref_v=start_pref_v,
            limit_pref_v=limit_pref_v,
            start_pref_ae=start_pref_ae,
            limit_pref_ae=limit_pref_ae,
            enable_atom_ener_coeff=enable_atom_ener_coeff,
            loss_func=loss_func,
            intensive_ener_virial=intensive_ener_virial,
            symm_virial=symm_virial,
        )
        self.start_pref_fr = start_pref_fr
        self.limit_pref_fr = limit_pref_fr
        self.start_pref_fm = start_pref_fm
        self.limit_pref_fm = limit_pref_fm
        self.has_fr = self.has_f
        self.has_fm = start_pref_fm != 0.0 or limit_pref_fm != 0.0

    def _atom_energy_coefficients(
        self, coefficients: Array, mask: Array | None
    ) -> Array:
        """Exclude masked atomic energies from the weighted spin-energy sum."""
        if mask is None:
            return coefficients
        xp = array_api_compat.array_namespace(coefficients)
        return coefficients * xp.reshape(mask, coefficients.shape)

    def _additional_loss(
        self,
        learning_rate: float,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool,
    ) -> tuple[Array | int, dict[str, Array]]:
        """Supervise magnetic forces independently of atomic-energy exclusions."""
        if not self.has_fm:
            return 0, {}
        prediction = model_dict["force_mag"]
        xp = array_api_compat.array_namespace(prediction)
        prediction = xp.reshape(prediction, (-1, 3))
        reference = xp.reshape(label_dict["force_mag"], (-1, 3))
        mask = xp.astype(xp.reshape(model_dict["mask_mag"], (-1,)), prediction.dtype)
        # The magnetic mask describes physical spin degrees of freedom, even
        # at atoms whose own energy is excluded from the atomic-energy sum.
        residual = (reference - prediction) * mask[:, None]
        error = xp.square(residual) if self.loss_func == "mse" else xp.abs(residual)
        mean_error = masked_atom_mean(error, mask, 3)
        find = label_dict.get("find_force_mag", 0.0)
        prefactor = self.limit_pref_fm + (self.start_pref_fm - self.limit_pref_fm) * (
            learning_rate / self.starter_learning_rate
        )
        metric = xp.sqrt(mean_error) if self.loss_func == "mse" else mean_error
        prefix = "rmse" if self.loss_func == "mse" else "mae"
        metrics = {f"{prefix}_fm": self.display_if_exist(metric, find)}
        if mae and self.loss_func == "mse":
            metrics["mae_fm"] = self.display_if_exist(
                masked_atom_mean(xp.abs(residual), mask, 3), find
            )
        return prefactor * find * mean_error, metrics

    @property
    def training_metric_names(self) -> tuple[str, ...]:
        """Return configured energy and real/magnetic force metrics."""
        prefix = "rmse" if self.loss_func == "mse" else "mae"
        names = ("rmse",)
        return names + tuple(
            f"{prefix}_{term}"
            for term in ("e", "fr", "fm", "v", "ae")
            if getattr(self, f"has_{term}")
        )

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        requirements = super().label_requirement
        if self.has_fm:
            requirements.append(
                DataRequirementItem(
                    "force_mag",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return requirements

    def serialize(self) -> dict:
        """Serialize the loss module."""
        data = {
            "@class": "EnergySpinLoss",
            "@version": 3 if self.symm_virial else 2,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_fr": self.start_pref_fr,
            "limit_pref_fr": self.limit_pref_fr,
            "start_pref_fm": self.start_pref_fm,
            "limit_pref_fm": self.limit_pref_fm,
            "start_pref_v": self.start_pref_v,
            "limit_pref_v": self.limit_pref_v,
            "start_pref_ae": self.start_pref_ae,
            "limit_pref_ae": self.limit_pref_ae,
            "enable_atom_ener_coeff": self.enable_atom_ener_coeff,
            "loss_func": self.loss_func,
            "intensive_ener_virial": self.intensive_ener_virial,
        }

        if self.symm_virial:
            data["symm_virial"] = True
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "EnergySpinLoss":
        """Deserialize the loss module."""
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 3, 1)
        data.pop("@class")
        # Backward compatibility: version 1 used legacy normalization
        if version < 2:
            data.setdefault("intensive_ener_virial", False)
        return cls(**data)
