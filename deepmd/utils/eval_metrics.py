# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

FULL_VALIDATION_WEIGHTED_METRIC_KEYS = {
    "energy_per_atom": ("mae_e_per_atom", "rmse_e_per_atom"),
    "force": ("mae_f", "rmse_f"),
    "virial_per_atom": ("mae_v_per_atom", "rmse_v_per_atom"),
}
DP_TEST_WEIGHTED_METRIC_KEYS = {
    "energy": ("mae_e", "rmse_e"),
    "energy_per_atom": ("mae_ea", "rmse_ea"),
    "force": ("mae_f", "rmse_f"),
    "virial": ("mae_v", "rmse_v"),
    "virial_per_atom": ("mae_va", "rmse_va"),
}
DP_TEST_SPIN_WEIGHTED_METRIC_KEYS = {
    "force_real": ("mae_fr", "rmse_fr"),
    "force_magnetic": ("mae_fm", "rmse_fm"),
}
DP_TEST_WEIGHTED_FORCE_METRIC_KEYS = ("mae_fw", "rmse_fw")
DP_TEST_HESSIAN_METRIC_KEYS = ("mae_h", "rmse_h")


def mae(diff: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return float(np.mean(np.abs(diff)))


def rmse(diff: np.ndarray) -> float:
    """Calculate root mean square error."""
    return float(np.sqrt(np.mean(diff * diff)))


@dataclass(frozen=True)
class ErrorStat:
    """One weighted MAE/RMSE pair."""

    mae: float
    rmse: float
    weight: float

    def as_weighted_average_errors(
        self,
        mae_key: str,
        rmse_key: str,
    ) -> dict[str, tuple[float, float]]:
        """Convert one metric pair into `weighted_average` inputs."""
        return {
            mae_key: (self.mae, self.weight),
            rmse_key: (self.rmse, self.weight),
        }


@dataclass(frozen=True)
class EnergyTypeEvalMetrics:
    """Shared energy-type metrics for one evaluation batch or system."""

    energy: ErrorStat | None = None
    energy_per_atom: ErrorStat | None = None
    force: ErrorStat | None = None
    virial: ErrorStat | None = None
    virial_per_atom: ErrorStat | None = None

    def as_weighted_average_errors(
        self,
        metric_keys: dict[str, tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        """Project shared metrics into caller-specific error dict keys."""
        errors: dict[str, tuple[float, float]] = {}
        for metric_name, (mae_key, rmse_key) in metric_keys.items():
            metric = getattr(self, metric_name)
            if metric is not None:
                errors.update(metric.as_weighted_average_errors(mae_key, rmse_key))
        return errors


@dataclass(frozen=True)
class SpinForceEvalMetrics:
    """Shared spin-force metrics for one evaluation batch or system."""

    force_real: ErrorStat | None = None
    force_magnetic: ErrorStat | None = None

    def as_weighted_average_errors(
        self,
        metric_keys: dict[str, tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        """Project shared spin metrics into caller-specific error dict keys."""
        errors: dict[str, tuple[float, float]] = {}
        for metric_name, (mae_key, rmse_key) in metric_keys.items():
            metric = getattr(self, metric_name)
            if metric is not None:
                errors.update(metric.as_weighted_average_errors(mae_key, rmse_key))
        return errors


def compute_error_stat(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    scale: float = 1.0,
) -> ErrorStat:
    """Compute one MAE/RMSE pair from aligned prediction and reference arrays."""
    diff = prediction - reference
    return ErrorStat(
        mae=mae(diff) * scale,
        rmse=rmse(diff) * scale,
        weight=float(diff.size),
    )


def compute_weighted_error_stat(
    prediction: np.ndarray,
    reference: np.ndarray,
    weight: np.ndarray,
) -> ErrorStat:
    """Compute weighted MAE/RMSE from aligned prediction and reference arrays."""
    diff = prediction - reference
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0.0:
        return ErrorStat(mae=0.0, rmse=0.0, weight=weight_sum)
    return ErrorStat(
        mae=float(np.sum(np.abs(diff) * weight) / weight_sum),
        rmse=float(np.sqrt(np.sum(diff * diff * weight) / weight_sum)),
        weight=weight_sum,
    )


def compute_energy_type_metrics(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    natoms: int,
    has_pbc: bool,
) -> EnergyTypeEvalMetrics:
    """Compute shared energy-type metrics for one evaluation dataset."""
    energy = None
    energy_per_atom = None
    force = None
    virial = None
    virial_per_atom = None

    if bool(test_data.get("find_energy", 0.0)):
        energy = compute_error_stat(
            prediction["energy"].reshape(-1, 1),
            test_data["energy"].reshape(-1, 1),
        )
        energy_per_atom = compute_error_stat(
            prediction["energy"].reshape(-1, 1),
            test_data["energy"].reshape(-1, 1),
            scale=1.0 / natoms,
        )

    if bool(test_data.get("find_force", 0.0)):
        force = compute_error_stat(
            prediction["force"].reshape(-1),
            test_data["force"].reshape(-1),
        )

    if has_pbc and bool(test_data.get("find_virial", 0.0)):
        virial = compute_error_stat(
            prediction["virial"].reshape(-1, 9),
            test_data["virial"].reshape(-1, 9),
        )
        virial_per_atom = compute_error_stat(
            prediction["virial"].reshape(-1, 9),
            test_data["virial"].reshape(-1, 9),
            scale=1.0 / natoms,
        )

    return EnergyTypeEvalMetrics(
        energy=energy,
        energy_per_atom=energy_per_atom,
        force=force,
        virial=virial,
        virial_per_atom=virial_per_atom,
    )


def compute_spin_force_metrics(
    force_real_prediction: np.ndarray,
    force_real_reference: np.ndarray,
    *,
    force_magnetic_prediction: np.ndarray | None = None,
    force_magnetic_reference: np.ndarray | None = None,
) -> SpinForceEvalMetrics:
    """Compute spin-aware force metrics from aligned real and magnetic forces."""
    force_real = compute_error_stat(force_real_prediction, force_real_reference)
    force_magnetic = None
    if force_magnetic_prediction is not None or force_magnetic_reference is not None:
        if force_magnetic_prediction is None or force_magnetic_reference is None:
            raise ValueError(
                "Spin magnetic force metrics require both prediction and reference."
            )
        force_magnetic = compute_error_stat(
            force_magnetic_prediction,
            force_magnetic_reference,
        )
    return SpinForceEvalMetrics(
        force_real=force_real,
        force_magnetic=force_magnetic,
    )


def _spin_force_metrics_from_prediction(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
) -> SpinForceEvalMetrics:
    """Align predicted and reference forces into real and magnetic subsets.

    Real forces cover all atoms, while magnetic forces are restricted to the
    magnetic atoms selected by the boolean ``mask_mag`` of shape
    ``(nframes, natoms)``. The magnetic term is produced only when
    ``find_force_mag`` is set and both prediction and reference magnetic
    forces are present, matching the ``dp test`` spin convention.

    Parameters
    ----------
    prediction : dict[str, np.ndarray]
        Model predictions containing ``force`` and, for the magnetic term,
        ``force_mag`` and ``mask_mag``.
    test_data : dict[str, np.ndarray]
        Reference labels and ``find_*`` availability flags for one system.

    Returns
    -------
    SpinForceEvalMetrics
        The real-atom and (optionally) magnetic-atom force errors.
    """
    force_real_prediction = prediction["force"].reshape(-1, 3)
    force_real_reference = test_data["force"].reshape(-1, 3)
    has_force_mag = (
        bool(test_data.get("find_force_mag", 0.0))
        and "force_mag" in prediction
        and "force_mag" in test_data
    )
    if not has_force_mag:
        return compute_spin_force_metrics(
            force_real_prediction=force_real_prediction,
            force_real_reference=force_real_reference,
        )
    magnetic_mask = prediction["mask_mag"].reshape(-1).astype(bool)
    return compute_spin_force_metrics(
        force_real_prediction=force_real_prediction,
        force_real_reference=force_real_reference,
        force_magnetic_prediction=prediction["force_mag"].reshape(-1, 3)[magnetic_mask],
        force_magnetic_reference=test_data["force_mag"].reshape(-1, 3)[magnetic_mask],
    )


def compute_full_validation_energy_metrics(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    natoms: int,
    has_pbc: bool,
) -> dict[str, tuple[float, float]]:
    """Compute energy-type full validation metrics for one system.

    Parameters
    ----------
    prediction : dict[str, np.ndarray]
        Model predictions containing ``energy``, ``force`` and optionally
        ``virial``.
    test_data : dict[str, np.ndarray]
        Reference labels and ``find_*`` availability flags for one system.
    natoms : int
        The number of atoms per frame, used for per-atom normalization.
    has_pbc : bool
        Whether the system is periodic, gating the virial metrics.

    Returns
    -------
    dict[str, tuple[float, float]]
        Weighted-average-ready ``(value, weight)`` pairs keyed by metric.
    """
    metrics = compute_energy_type_metrics(prediction, test_data, natoms, has_pbc)
    return metrics.as_weighted_average_errors(FULL_VALIDATION_WEIGHTED_METRIC_KEYS)


def compute_full_validation_spin_metrics(
    prediction: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    natoms: int,
    has_pbc: bool,
) -> dict[str, tuple[float, float]]:
    """Compute spin-energy full validation metrics for one system.

    The energy term reuses per-atom energy errors. Forces are split into a
    real-atom term over all atoms and a magnetic term over the magnetic atoms
    selected by ``mask_mag``. Spin models do not report virial, so no virial
    metric is produced.

    Parameters
    ----------
    prediction : dict[str, np.ndarray]
        Model predictions containing ``energy``, ``force``, ``force_mag`` and
        the boolean ``mask_mag``.
    test_data : dict[str, np.ndarray]
        Reference labels and ``find_*`` availability flags for one system.
    natoms : int
        The number of atoms per frame, used for per-atom normalization.
    has_pbc : bool
        Unused; spin full validation never reports virial. Present to keep a
        uniform profile signature.

    Returns
    -------
    dict[str, tuple[float, float]]
        Weighted-average-ready ``(value, weight)`` pairs keyed by metric.
    """
    errors: dict[str, tuple[float, float]] = {}
    if bool(test_data.get("find_energy", 0.0)):
        energy_per_atom = compute_error_stat(
            prediction["energy"].reshape(-1, 1),
            test_data["energy"].reshape(-1, 1),
            scale=1.0 / natoms,
        )
        errors.update(
            energy_per_atom.as_weighted_average_errors(
                "mae_e_per_atom", "rmse_e_per_atom"
            )
        )
    if bool(test_data.get("find_force", 0.0)):
        spin_metrics = _spin_force_metrics_from_prediction(prediction, test_data)
        errors.update(
            spin_metrics.as_weighted_average_errors(DP_TEST_SPIN_WEIGHTED_METRIC_KEYS)
        )
    return errors


@dataclass(frozen=True)
class FullValidationMetricProfile:
    """Metric family definition for one full validation model class.

    Bundles every aspect that differs between energy-type and spin-energy full
    validation so the validator stays data-driven instead of branching on the
    model class:

    - ``column_order`` defines the ``val.log`` table layout as
      ``(header_label, metric_key)`` pairs.
    - ``metric_key_map`` maps a normalized ``validation_metric`` token (such as
      ``"e:mae"``) to an internal metric key (such as ``"mae_e_per_atom"``).
    - ``metric_family_by_key`` maps an internal metric key back to its family,
      used for display-unit lookup.
    - ``unit_by_family`` maps a family to its ``(display_unit, scale)``.
    - ``prefactor_by_metric`` maps a metric token to the loss prefactor keys
      that must both be active for the metric to be trainable.
    - ``needs_spin`` indicates whether the model consumes a spin input and
      emits magnetic forces.
    - ``log_header_note`` is the one-line table legend written to ``val.log``.
    - ``compute_system_metrics`` turns one system's prediction and reference
      into weighted ``(value, weight)`` metric pairs.

    Attributes
    ----------
    name : str
        Profile identifier, either ``"energy"`` or ``"spin"``.
    column_order : tuple[tuple[str, str], ...]
        Ordered ``(header_label, metric_key)`` pairs for the log table.
    metric_key_map : dict[str, str]
        Normalized metric token to internal metric key.
    metric_family_by_key : dict[str, str]
        Internal metric key to family identifier.
    unit_by_family : dict[str, tuple[str, float]]
        Family identifier to ``(display_unit, scale)``.
    prefactor_by_metric : dict[str, tuple[str, str]]
        Normalized metric token to ``(start_pref_key, limit_pref_key)``.
    needs_spin : bool
        Whether the profile requires spin input and magnetic-force outputs.
    log_header_note : str
        One-line legend describing the metric columns.
    compute_system_metrics : Callable
        Routine computing weighted metric pairs for one system, with signature
        ``(prediction, test_data, natoms, has_pbc) -> dict``.
    """

    name: str
    column_order: tuple[tuple[str, str], ...]
    metric_key_map: dict[str, str]
    metric_family_by_key: dict[str, str]
    unit_by_family: dict[str, tuple[str, float]]
    prefactor_by_metric: dict[str, tuple[str, str]]
    needs_spin: bool
    log_header_note: str
    compute_system_metrics: Callable[
        [dict[str, np.ndarray], dict[str, np.ndarray], int, bool],
        dict[str, tuple[float, float]],
    ]


ENERGY_FULL_VALIDATION_PROFILE = FullValidationMetricProfile(
    name="energy",
    column_order=(
        ("E_MAE", "mae_e_per_atom"),
        ("E_RMSE", "rmse_e_per_atom"),
        ("F_MAE", "mae_f"),
        ("F_RMSE", "rmse_f"),
        ("V_MAE", "mae_v_per_atom"),
        ("V_RMSE", "rmse_v_per_atom"),
    ),
    metric_key_map={
        "e:mae": "mae_e_per_atom",
        "e:rmse": "rmse_e_per_atom",
        "f:mae": "mae_f",
        "f:rmse": "rmse_f",
        "v:mae": "mae_v_per_atom",
        "v:rmse": "rmse_v_per_atom",
    },
    metric_family_by_key={
        "mae_e_per_atom": "e",
        "rmse_e_per_atom": "e",
        "mae_f": "f",
        "rmse_f": "f",
        "mae_v_per_atom": "v",
        "rmse_v_per_atom": "v",
    },
    unit_by_family={
        "e": ("meV/atom", 1000.0),
        "f": ("meV/Å", 1000.0),
        "v": ("meV/atom", 1000.0),
    },
    prefactor_by_metric={
        "e:mae": ("start_pref_e", "limit_pref_e"),
        "e:rmse": ("start_pref_e", "limit_pref_e"),
        "f:mae": ("start_pref_f", "limit_pref_f"),
        "f:rmse": ("start_pref_f", "limit_pref_f"),
        "v:mae": ("start_pref_v", "limit_pref_v"),
        "v:rmse": ("start_pref_v", "limit_pref_v"),
    },
    needs_spin=False,
    log_header_note=(
        "# E uses per-atom energy, F uses component-wise force errors, "
        "and V uses virial normalized by natoms.\n"
    ),
    compute_system_metrics=compute_full_validation_energy_metrics,
)

SPIN_FULL_VALIDATION_PROFILE = FullValidationMetricProfile(
    name="spin",
    column_order=(
        ("E_MAE", "mae_e_per_atom"),
        ("E_RMSE", "rmse_e_per_atom"),
        ("FR_MAE", "mae_fr"),
        ("FR_RMSE", "rmse_fr"),
        ("FM_MAE", "mae_fm"),
        ("FM_RMSE", "rmse_fm"),
    ),
    metric_key_map={
        "e:mae": "mae_e_per_atom",
        "e:rmse": "rmse_e_per_atom",
        "fr:mae": "mae_fr",
        "fr:rmse": "rmse_fr",
        "fm:mae": "mae_fm",
        "fm:rmse": "rmse_fm",
    },
    metric_family_by_key={
        "mae_e_per_atom": "e",
        "rmse_e_per_atom": "e",
        "mae_fr": "fr",
        "rmse_fr": "fr",
        "mae_fm": "fm",
        "rmse_fm": "fm",
    },
    unit_by_family={
        "e": ("meV/atom", 1000.0),
        "fr": ("meV/Å", 1000.0),
        "fm": ("meV/μB", 1000.0),
    },
    prefactor_by_metric={
        "e:mae": ("start_pref_e", "limit_pref_e"),
        "e:rmse": ("start_pref_e", "limit_pref_e"),
        "fr:mae": ("start_pref_fr", "limit_pref_fr"),
        "fr:rmse": ("start_pref_fr", "limit_pref_fr"),
        "fm:mae": ("start_pref_fm", "limit_pref_fm"),
        "fm:rmse": ("start_pref_fm", "limit_pref_fm"),
    },
    needs_spin=True,
    log_header_note=(
        "# E uses per-atom energy, FR uses component-wise real-atom force "
        "errors, and FM uses magnetic-atom force errors.\n"
    ),
    compute_system_metrics=compute_full_validation_spin_metrics,
)

FULL_VALIDATION_PROFILES: dict[str, FullValidationMetricProfile] = {
    ENERGY_FULL_VALIDATION_PROFILE.name: ENERGY_FULL_VALIDATION_PROFILE,
    SPIN_FULL_VALIDATION_PROFILE.name: SPIN_FULL_VALIDATION_PROFILE,
}
